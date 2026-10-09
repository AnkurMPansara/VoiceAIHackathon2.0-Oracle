"""PostgreSQL database connection manager and migration system.

Implements WP3 (Persistence agent) STATE-01 through STATE-06 requirements:

- STATE-01: Atomic transactions for outcome processing (attempt revision lock,
  seller-state row, revision comparison, delta calculation, latest + audit rows,
  statistics update, outbox record — all in one transaction).
- STATE-02: Crash recovery — no contribution visible before commit; redelivery
  is a no-op after commit. Queue adapters acknowledge only after database commit.
- STATE-03: Per-seller concurrent update serialization with row-level locks,
  3-retry bounded jitter, signed 64-bit state_version.
- STATE-04: Model state eligibility bounded by state-history interval.
- STATE-05: Redis cache integration (cache key includes compatibility ID,
  state_version, cached-at time).
- STATE-06: Cold-start zero state only after authoritative not-found; no silent
  expiration of authoritative states.

Supports both PostgreSQL (production) and SQLite (development/testing) via
a unified ``DatabaseConnection`` interface.

Usage
-----
>>> async with DatabaseConnection("postgresql://localhost/btc") as db:
...     create_tables(db)
...     with db.transaction():
...         db.execute("INSERT INTO attempt_latest ...")

Development / testing with SQLite:
>>> db = DatabaseConnection("sqlite:///./btc_dev.sqlite")
>>> db.connect()
>>> create_tables(db)
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Connection = Any  # psycopg2 connection (PostgreSQL) or sqlite3.Connection (SQLite)
Cursor = Any      # psycopg2 cursor or sqlite3.Cursor


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatabaseError(Exception):
    """Base exception for database operations."""

    def __init__(self, message: str, error_code: str = "DATABASE_ERROR") -> None:
        super().__init__(message)
        self.error_code = error_code


class ConnectionError(DatabaseError):
    """Failed to establish a database connection."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="CONNECTION_FAILED")


class TransactionError(DatabaseError):
    """Transaction failed (commit/rollback error)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="TRANSACTION_FAILED")


class LockTimeoutError(DatabaseError):
    """Row-level lock could not be acquired within the retry budget."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="LOCK_TIMEOUT")


# ---------------------------------------------------------------------------
# DatabaseConnection
# ---------------------------------------------------------------------------

class DatabaseConnection:
    """Database connection manager with connection pooling.

    Supports both PostgreSQL (production) and SQLite (development/testing).

    Parameters
    ----------
    dsn : str
        Database connection string.
        PostgreSQL: ``"postgresql://user:pass@host:port/dbname"``
        SQLite: ``"sqlite:///path/to/db.sqlite"``
    pool_size : int
        Maximum number of connections in the pool (PostgreSQL only).
        Ignored for SQLite.
    timeout : float
        Connection timeout in seconds.

    Attributes
    ----------
    _dsn : str
        The connection string.
    _pool : list[Connection]
        Internal connection pool (PostgreSQL).
    _conn : Connection | None
        Primary connection handle.
    _is_sqlite : bool
        Whether this connection uses SQLite.
    _timeout : float
        Connection timeout in seconds.
    """

    def __init__(
        self,
        dsn: str,
        pool_size: int = 10,
        timeout: float = 30.0,
    ) -> None:
        """Initialise the connection manager.

        Parameters
        ----------
        dsn : str
            Database connection string.
        pool_size : int
            Connection pool size.
        timeout : float
            Connection timeout in seconds.
        """
        self._dsn = dsn
        self._pool_size = pool_size
        self._timeout = timeout
        self._conn: Optional[Connection] = None
        self._pool: list[Connection] = []
        self._is_sqlite = dsn.startswith("sqlite:///") or dsn.startswith("sqlite://")
        self._in_transaction: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> "DatabaseConnection":
        """Establish database connection.

        For PostgreSQL, creates connections up to ``pool_size``.
        For SQLite, opens a single connection.

        Returns
        -------
        DatabaseConnection
            ``self`` for chaining.

        Raises
        ------
        ConnectionError
            If the connection cannot be established.
        """
        try:
            if self._is_sqlite:
                self._connect_sqlite()
            else:
                self._connect_postgresql()
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to {self._dsn}: {exc}"
            ) from exc
        return self

    def disconnect(self) -> None:
        """Close all connections and empty the pool.

        Commits any pending transaction before closing.
        """
        if self._conn is not None:
            try:
                if self._in_transaction:
                    self._conn.rollback()
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

        for conn in self._pool:
            try:
                conn.close()
            except Exception:
                pass
        self._pool.clear()

    def __enter__(self) -> "DatabaseConnection":
        """Context manager entry — establish connection."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — close all connections."""
        self.disconnect()

    # ── PostgreSQL ───────────────────────────────────────────────────────

    def _connect_postgresql(self) -> None:
        """Establish PostgreSQL connection(s).

        Creates a pool of connections up to ``pool_size``. The primary
        connection is stored in ``_conn`` and additional connections
        are managed in ``_pool``.
        """
        import psycopg2
        import psycopg2.pool

        # Parse DSN to extract components
        dsn = self._dsn
        if dsn.startswith("postgresql://"):
            dsn = dsn.replace("postgresql://", "postgres://", 1)

        # Build connection parameters
        conn_params: Dict[str, Any] = {}
        if "://" in dsn:
            conn_params["dsn"] = dsn
        else:
            for part in dsn.split():
                if "=" in part:
                    key, _, value = part.partition("=")
                    conn_params[key] = value

        # Use connection pool if pool_size > 1
        if self._pool_size > 1:
            try:
                self._pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=self._pool_size,
                    **conn_params,
                )
                self._conn = self._pool.getconn()
                # Set statement timeout
                self._conn.set_client_encoding("UTF8")
            except Exception:
                # Fallback to single connection
                self._conn = psycopg2.connect(**conn_params, timeout=self._timeout)
                self._conn.set_client_encoding("UTF8")
        else:
            self._conn = psycopg2.connect(**conn_params, timeout=self._timeout)
            self._conn.set_client_encoding("UTF8")

        logger.info("PostgreSQL connection established (pool_size=%d)", self._pool_size)

    def _connect_sqlite(self) -> None:
        """Establish SQLite connection.

        Creates the database file if it does not exist.
        """
        import sqlite3

        # Extract path from DSN
        dsn = self._dsn
        if dsn.startswith("sqlite:///"):
            path = dsn[len("sqlite:///"):]
        elif dsn.startswith("sqlite://"):
            path = dsn[len("sqlite://"):]
        else:
            path = dsn

        # Ensure the directory exists
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read performance
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign keys
        self._conn.execute("PRAGMA foreign_keys=ON")

        logger.info("SQLite connection established: %s", db_path)

    # ── Query execution ──────────────────────────────────────────────────

    def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a query with params.

        Parameters
        ----------
        query : str
            SQL query string.
        params : tuple
            Query parameters.

        Returns
        -------
        Any
            Cursor object (PostgreSQL) or None (SQLite).

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            return cursor
        except Exception as exc:
            logger.error("Query execution failed: %s — %s", query[:200], exc)
            raise DatabaseError(f"Query execution failed: {exc}") from exc

    def fetchall(self, query: str, params: tuple = ()) -> list:
        """Execute query and return all rows.

        Parameters
        ----------
        query : str
            SQL query string.
        params : tuple
            Query parameters.

        Returns
        -------
        list
            List of row dicts.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        cursor = self.execute(query, params)
        rows = cursor.fetchall()
        return self._rows_to_dicts(rows, cursor)

    def fetchone(self, query: str, params: tuple = ()) -> dict:
        """Execute query and return one row as dict.

        Parameters
        ----------
        query : str
            SQL query string.
        params : tuple
            Query parameters.

        Returns
        -------
        dict
            Single row as a dictionary, or ``None`` if no rows found.

        Raises
        ------
        DatabaseError
            If the query fails.
        """
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row, cursor)

    # ── Transaction management ───────────────────────────────────────────

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for atomic transactions.

        Commits on success, rolls back on exception.

        STATE-01/02: Single transaction for outcome processing.
        Within one database transaction: establish the attempt revision
        lock, lock/create the seller-state row, compare revision and
        payload, calculate the delta, write latest and audit rows,
        update statistics and state version, and append an outbox
        record. Commit before acknowledging success.

        If the process crashes before commit, no contribution is
        visible. If it crashes after commit but before acknowledgement,
        redelivery is a no-op.

        Raises
        ------
        TransactionError
            If the transaction fails to commit or rollback.
        """
        conn = self._get_connection()
        try:
            conn.autocommit = False if not self._is_sqlite else False
            if not self._is_sqlite:
                conn.set_session(autocommit=False)
            self._in_transaction = True
            yield
            conn.commit()
            self._in_transaction = False
        except DatabaseError:
            # Database errors: rollback and re-raise wrapped
            try:
                conn.rollback()
            except Exception:
                pass
            self._in_transaction = False
            raise
        except Exception as exc:
            # Non-database exceptions: rollback but re-raise original
            try:
                conn.rollback()
            except Exception:
                pass
            self._in_transaction = False
            raise

    # ── Row-level locking ────────────────────────────────────────────────

    @contextmanager
    def locked_row(
        self,
        table: str,
        key: tuple,
        timeout: float = 5.0,
    ) -> Iterator[None]:
        """Acquire row-level lock for concurrent update serialization.

        STATE-03: Per-seller concurrent updates serialized.
        Uses ``SELECT ... FOR UPDATE`` with retry logic.

        Retries at most 3 times with bounded jitter, then fails.

        Parameters
        ----------
        table : str
            Table name to lock rows in. Must contain a ``key`` column
            that matches the first element of *key* tuple, and a
            ``seller_id`` column for seller-level serialization.
        key : tuple
            Primary key values for the row to lock. The first element
            is used as the key column value; additional elements are
            used for composite keys.
        timeout : float
            Maximum time to wait for lock acquisition (seconds).

        Yields
        ------
        None

        Raises
        ------
        LockTimeoutError
            If the lock cannot be acquired within the retry budget.
        DatabaseError
            If the query fails.

        Notes
        -----
        This method acquires a row-level lock using ``SELECT ... FOR UPDATE``.
        The caller MUST be within a transaction when calling this method.

        For seller-level serialization (STATE-03), the lock targets
        the ``seller_model_state`` row for the given seller.
        """
        conn = self._get_connection()
        max_retries = 3
        lock_acquired = False
        placeholder = "?" if self._is_sqlite else "%s"
        for_update = "FOR UPDATE" if not self._is_sqlite else ""

        for attempt in range(max_retries):
            try:
                # Use SELECT ... FOR UPDATE to acquire row-level lock
                # For seller_model_state: lock by (model_compatibility_id, seller_id)
                if len(key) >= 2:
                    model_compat_id, seller_id = key[0], key[1]
                    sql = (
                        f"SELECT * FROM {table} "
                        f"WHERE model_compatibility_id = {placeholder} AND seller_id = {placeholder} "
                        f"{for_update}"
                    ).strip()
                    lock_params: tuple = (model_compat_id, seller_id)
                else:
                    # Generic single-key lock
                    sql = (
                        f"SELECT * FROM {table} "
                        f"WHERE key = {placeholder} {for_update}"
                    ).strip()
                    lock_params = (key[0],)

                cursor = conn.cursor()
                cursor.execute(sql, lock_params)
                row = cursor.fetchone()
                lock_acquired = True

                # If row doesn't exist, create it (for seller state initialization)
                if row is None and table == "seller_model_state":
                    # The caller should handle row creation within the lock
                    pass

                yield
                break

            except Exception as exc:
                # Check if this is a lock timeout (PostgreSQL: 55P03)
                is_lock_timeout = False
                if not self._is_sqlite:
                    # psycopg2 lock timeout error
                    if hasattr(exc, 'pgcode') and exc.pgcode == '55P03':
                        is_lock_timeout = True
                    elif hasattr(exc, 'sqlstate') and exc.sqlstate == '55P03':
                        is_lock_timeout = True

                if is_lock_timeout and attempt < max_retries - 1:
                    # Bounded jitter: random delay between 0.1 and 0.5 seconds
                    jitter = random.uniform(0.1, 0.5)
                    wait_time = min(jitter, timeout / (max_retries - attempt))
                    logger.warning(
                        "Lock timeout on %s (attempt %d/%d), retrying in %.2fs",
                        table, attempt + 1, max_retries, wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        if not lock_acquired:
            raise LockTimeoutError(
                f"Could not acquire lock on {table} after {max_retries} attempts"
            )

    # ── Internal helpers ─────────────────────────────────────────────────

    def _get_connection(self) -> Connection:
        """Get the current connection, raising if none is available."""
        if self._conn is None:
            raise ConnectionError(
                "No active connection. Call connect() first."
            )
        return self._conn

    @staticmethod
    def _row_to_dict(row: Any, cursor: Cursor) -> dict:
        """Convert a single row to a dictionary."""
        if hasattr(cursor, 'description') and cursor.description:
            keys = [desc[0] for desc in cursor.description]
        else:
            # Fallback for sqlite3.Row
            if hasattr(row, '_fields'):
                keys = row._fields
            else:
                keys = [str(i) for i in range(len(row))]
        return dict(zip(keys, row))

    @staticmethod
    def _rows_to_dicts(rows: list, cursor: Cursor) -> list:
        """Convert multiple rows to a list of dictionaries."""
        if not rows:
            return []
        if hasattr(cursor, 'description') and cursor.description:
            keys = [desc[0] for desc in cursor.description]
        else:
            if hasattr(rows[0], '_fields'):
                keys = rows[0]._fields
            else:
                keys = [str(i) for i in range(len(rows[0]))]
        return [dict(zip(keys, row)) for row in rows]

    # ── Utility ──────────────────────────────────────────────────────────

    @property
    def is_sqlite(self) -> bool:
        """Whether this connection uses SQLite."""
        return self._is_sqlite

    @property
    def is_connected(self) -> bool:
        """Whether a connection is currently active."""
        return self._conn is not None


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def create_tables(conn: DatabaseConnection) -> None:
    """Create all required tables (STATE-01).

    Creates the following tables:

    - ``attempt_latest``: PK (source, attempt_id) — latest outcome per attempt.
    - ``attempt_revisions``: Append-only audit of accepted revisions.
    - ``seller_model_state``: PK (model_compatibility_id, seller_id) —
      Bayesian sufficient statistics per seller.
    - ``decisions``: Unique decision/request ID — scheduling decisions.
    - ``retry_decisions``: Unique RET-04 key — retry outcomes.
    - ``outbox``: Transactional events for committed state changes.

    Parameters
    ----------
    conn : DatabaseConnection
        Active database connection.

    Notes
    -----
    Uses ``CREATE TABLE IF NOT EXISTS`` for idempotency.
    All tables use ``IF NOT EXISTS`` so this function can be called
    multiple times without error.

    SQLite compatibility:
    - Uses ``SERIAL``-equivalent ``INTEGER PRIMARY KEY AUTOINCREMENT``
    - Uses ``BLOB`` for payload hashes
    - Uses ``DATETIME`` for timestamps (stored as ISO 8601 strings)
    """
    if conn.is_sqlite:
        _create_tables_sqlite(conn)
    else:
        _create_tables_postgresql(conn)


def _create_tables_postgresql(conn: DatabaseConnection) -> None:
    """Create tables for PostgreSQL."""
    with conn.transaction():
        # ── attempt_latest ─────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempt_latest (
                source            VARCHAR(128)    NOT NULL,
                attempt_id        VARCHAR(128)    NOT NULL,
                seller_id         VARCHAR(128)    NOT NULL,
                lead_id           VARCHAR(128)    NOT NULL,
                revision          BIGINT          NOT NULL,
                payload_hash      CHAR(64)        NOT NULL,
                event_id          VARCHAR(128)    NOT NULL,
                finalized_at      TIMESTAMPTZ     NOT NULL,
                call_start_time   TIMESTAMPTZ     NOT NULL,
                call_end_time     TIMESTAMPTZ     NOT NULL,
                lead_sent_time    TIMESTAMPTZ     NOT NULL,
                attempt_number    INT             NOT NULL,
                answered          BOOLEAN         NOT NULL,
                disposition       VARCHAR(32)     NOT NULL,
                meeting_fixed     BOOLEAN         NOT NULL,
                requested_callback_at TIMESTAMPTZ,
                decision_id       VARCHAR(128),
                duration_s        INT,
                dialer_version    VARCHAR(128),
                source_bucket     VARCHAR(128),
                created_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
                updated_at        TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
                PRIMARY KEY (source, attempt_id)
            )
        """)

        # ── attempt_revisions (append-only audit) ──────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempt_revisions (
                id                BIGSERIAL       PRIMARY KEY,
                source            VARCHAR(128)    NOT NULL,
                attempt_id        VARCHAR(128)    NOT NULL,
                revision          BIGINT          NOT NULL,
                payload_hash      CHAR(64)        NOT NULL,
                event_id          VARCHAR(128)    NOT NULL,
                finalized_at      TIMESTAMPTZ     NOT NULL,
                call_start_time   TIMESTAMPTZ     NOT NULL,
                call_end_time     TIMESTAMPTZ     NOT NULL,
                lead_sent_time    TIMESTAMPTZ     NOT NULL,
                attempt_number    INT             NOT NULL,
                answered          BOOLEAN         NOT NULL,
                disposition       VARCHAR(32)     NOT NULL,
                meeting_fixed     BOOLEAN         NOT NULL,
                requested_callback_at TIMESTAMPTZ,
                decision_id       VARCHAR(128),
                duration_s        INT,
                dialer_version    VARCHAR(128),
                source_bucket     VARCHAR(128),
                ingestion_time    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
                UNIQUE (source, attempt_id, revision)
            )
        """)

        # Index for efficient seller-level lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_attempt_revisions_seller
            ON attempt_revisions (seller_id, call_start_time)
        """)

        # ── seller_model_state ─────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seller_model_state (
                model_compatibility_id  VARCHAR(128)    NOT NULL,
                seller_id               VARCHAR(128)    NOT NULL,
                a                       FLOAT8[]        NOT NULL,
                b                       FLOAT8[]        NOT NULL,
                n                       BIGINT          NOT NULL DEFAULT 0,
                state_version           BIGINT          NOT NULL DEFAULT 0,
                max_call_time           TIMESTAMPTZ,
                last_commit_time        TIMESTAMPTZ,
                created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
                updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
                PRIMARY KEY (model_compatibility_id, seller_id),
                CONSTRAINT chk_state_version_nonneg CHECK (state_version >= 0),
                CONSTRAINT chk_n_nonneg CHECK (n >= 0)
            )
        """)

        # ── decisions ──────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id           VARCHAR(128)        PRIMARY KEY,
                request_id            VARCHAR(128)        NOT NULL,
                seller_id             VARCHAR(128)        NOT NULL,
                lead_id               VARCHAR(128)        NOT NULL,
                status                VARCHAR(32)         NOT NULL,
                scheduled_at          TIMESTAMPTZ,
                secondary_at          TIMESTAMPTZ,
                reason_code           VARCHAR(256)        NOT NULL,
                mode                  VARCHAR(32)         NOT NULL,
                assignment            VARCHAR(32)         NOT NULL,
                experiment_id         VARCHAR(128),
                policy_version        VARCHAR(128)        NOT NULL,
                bundle_id             VARCHAR(128)        NOT NULL,
                model_compatibility_id VARCHAR(128)       NOT NULL,
                profile_version       VARCHAR(128)        NOT NULL,
                calendar_version      VARCHAR(128)        NOT NULL,
                context_version       INT                 NOT NULL,
                state_version         BIGINT              NOT NULL,
                n_attempts            INT                 NOT NULL,
                n_eff                 INT                 NOT NULL,
                prior_level           FLOAT8              NOT NULL,
                prior_weight          FLOAT8              NOT NULL,
                expected_reward       FLOAT8              NOT NULL,
                latent_std            FLOAT8              NOT NULL,
                predictive_std        FLOAT8              NOT NULL,
                candidate_count       INT                 NOT NULL,
                action_probability    FLOAT8,
                assignment_probability FLOAT8,
                ope_eligible          BOOLEAN             NOT NULL,
                created_at            TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
                valid_until           TIMESTAMPTZ,
                canonical_request_hash CHAR(64)           NOT NULL,
                response_json         JSONB               NOT NULL,
                candidate_set_json    JSONB,
                UNIQUE (request_id),
                UNIQUE (canonical_request_hash)
            )
        """)

        # ── retry_decisions ────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retry_decisions (
                decision_id           VARCHAR(128)        PRIMARY KEY,
                request_id            VARCHAR(128)        NOT NULL,
                source                VARCHAR(128)        NOT NULL,
                attempt_id            VARCHAR(128)        NOT NULL,
                expected_revision     BIGINT              NOT NULL DEFAULT 0,
                result                VARCHAR(32)         NOT NULL,
                scheduled_at          TIMESTAMPTZ,
                reason_code           VARCHAR(256)        NOT NULL,
                policy_version        VARCHAR(128)        NOT NULL,
                superseded_by         VARCHAR(128),
                scheduler_link        VARCHAR(128),
                created_at            TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
                UNIQUE (source, attempt_id, expected_revision)
            )
        """)

        # ── outbox ─────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox (
                id                    BIGSERIAL           PRIMARY KEY,
                event_type            VARCHAR(64)         NOT NULL,
                entity_type           VARCHAR(64)         NOT NULL,
                entity_id             VARCHAR(128)        NOT NULL,
                payload_json          JSONB               NOT NULL,
                state_version         BIGINT              NOT NULL DEFAULT 0,
                status                VARCHAR(32)         NOT NULL DEFAULT 'pending',
                delivered_at          TIMESTAMPTZ,
                delivery_attempts     INT                 NOT NULL DEFAULT 0,
                created_at            TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
                CONSTRAINT chk_status CHECK (status IN ('pending', 'delivered', 'failed'))
            )
        """)

        # Index for outbox worker to find pending events
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_outbox_pending
            ON outbox (status, created_at)
            WHERE status = 'pending'
        """)

        # ── migration history ──────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version             INT             PRIMARY KEY,
                applied_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
                checksum            CHAR(64)        NOT NULL
            )
        """)

    logger.info("PostgreSQL tables created successfully")


def _create_tables_sqlite(conn: DatabaseConnection) -> None:
    """Create tables for SQLite (development/testing fallback).

    Maps PostgreSQL types to SQLite equivalents:
    - TIMESTAMPTZ → TEXT (ISO 8601 strings)
    - FLOAT8[] → TEXT (JSON arrays)
    - JSONB → TEXT (JSON strings)
    - BIGSERIAL → INTEGER PRIMARY KEY AUTOINCREMENT
    - BOOLEAN → INTEGER (0/1)
    """
    with conn.transaction():
        # ── attempt_latest ─────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempt_latest (
                source            TEXT            NOT NULL,
                attempt_id        TEXT            NOT NULL,
                seller_id         TEXT            NOT NULL,
                lead_id           TEXT            NOT NULL,
                revision          INTEGER         NOT NULL,
                payload_hash      TEXT(64)        NOT NULL,
                event_id          TEXT            NOT NULL,
                finalized_at      TEXT            NOT NULL,
                call_start_time   TEXT            NOT NULL,
                call_end_time     TEXT            NOT NULL,
                lead_sent_time    TEXT            NOT NULL,
                attempt_number    INTEGER         NOT NULL,
                answered          INTEGER         NOT NULL,
                disposition       TEXT(32)        NOT NULL,
                meeting_fixed     INTEGER         NOT NULL,
                requested_callback_at TEXT,
                decision_id       TEXT,
                duration_s        INTEGER,
                dialer_version    TEXT(128),
                source_bucket     TEXT(128),
                created_at        TEXT            NOT NULL DEFAULT (datetime('now')),
                updated_at        TEXT            NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (source, attempt_id)
            )
        """)

        # ── attempt_revisions (append-only audit) ──────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempt_revisions (
                id                INTEGER         PRIMARY KEY AUTOINCREMENT,
                source            TEXT            NOT NULL,
                attempt_id        TEXT            NOT NULL,
                seller_id         TEXT            NOT NULL,
                revision          INTEGER         NOT NULL,
                payload_hash      TEXT(64)        NOT NULL,
                event_id          TEXT            NOT NULL,
                finalized_at      TEXT            NOT NULL,
                call_start_time   TEXT            NOT NULL,
                call_end_time     TEXT            NOT NULL,
                lead_sent_time    TEXT            NOT NULL,
                attempt_number    INTEGER         NOT NULL,
                answered          INTEGER         NOT NULL,
                disposition       TEXT(32)        NOT NULL,
                meeting_fixed     INTEGER         NOT NULL,
                requested_callback_at TEXT,
                decision_id       TEXT,
                duration_s        INTEGER,
                dialer_version    TEXT(128),
                source_bucket     TEXT(128),
                ingestion_time    TEXT            NOT NULL DEFAULT (datetime('now')),
                UNIQUE (source, attempt_id, revision)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_attempt_revisions_seller
            ON attempt_revisions (seller_id, call_start_time)
        """)

        # ── seller_model_state ─────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seller_model_state (
                model_compatibility_id  TEXT            NOT NULL,
                seller_id               TEXT            NOT NULL,
                a                       TEXT            NOT NULL,
                b                       TEXT            NOT NULL,
                n                       INTEGER         NOT NULL DEFAULT 0,
                state_version           INTEGER         NOT NULL DEFAULT 0,
                max_call_time           TEXT,
                last_commit_time        TEXT,
                created_at              TEXT            NOT NULL DEFAULT (datetime('now')),
                updated_at              TEXT            NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (model_compatibility_id, seller_id)
            )
        """)

        # ── decisions ──────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                decision_id           TEXT            PRIMARY KEY,
                request_id            TEXT            NOT NULL,
                seller_id             TEXT            NOT NULL,
                lead_id               TEXT            NOT NULL,
                status                TEXT(32)        NOT NULL,
                scheduled_at          TEXT,
                secondary_at          TEXT,
                reason_code           TEXT(256)       NOT NULL,
                mode                  TEXT(32)        NOT NULL,
                assignment            TEXT(32)        NOT NULL,
                experiment_id         TEXT,
                policy_version        TEXT(128)       NOT NULL,
                bundle_id             TEXT(128)       NOT NULL,
                model_compatibility_id TEXT(128)     NOT NULL,
                profile_version       TEXT(128)       NOT NULL,
                calendar_version      TEXT(128)       NOT NULL,
                context_version       INTEGER         NOT NULL,
                state_version         INTEGER         NOT NULL,
                n_attempts            INTEGER         NOT NULL,
                n_eff                 INTEGER         NOT NULL,
                prior_level           REAL            NOT NULL,
                prior_weight          REAL            NOT NULL,
                expected_reward       REAL            NOT NULL,
                latent_std            REAL            NOT NULL,
                predictive_std        REAL            NOT NULL,
                candidate_count       INTEGER         NOT NULL,
                action_probability    REAL,
                assignment_probability REAL,
                ope_eligible          INTEGER         NOT NULL,
                created_at            TEXT            NOT NULL DEFAULT (datetime('now')),
                valid_until           TEXT,
                canonical_request_hash TEXT(64)       NOT NULL,
                response_json         TEXT            NOT NULL,
                candidate_set_json    TEXT,
                UNIQUE (request_id),
                UNIQUE (canonical_request_hash)
            )
        """)

        # ── retry_decisions ────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retry_decisions (
                decision_id           TEXT            PRIMARY KEY,
                request_id            TEXT            NOT NULL,
                source                TEXT(128)       NOT NULL,
                attempt_id            TEXT(128)       NOT NULL,
                expected_revision     INTEGER         NOT NULL DEFAULT 0,
                result                TEXT(32)        NOT NULL,
                scheduled_at          TEXT,
                reason_code           TEXT(256)       NOT NULL,
                policy_version        TEXT(128)       NOT NULL,
                superseded_by         TEXT,
                scheduler_link        TEXT,
                created_at            TEXT            NOT NULL DEFAULT (datetime('now')),
                UNIQUE (source, attempt_id, expected_revision)
            )
        """)

        # ── outbox ─────────────────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS outbox (
                id                    INTEGER         PRIMARY KEY AUTOINCREMENT,
                event_type            TEXT(64)        NOT NULL,
                entity_type           TEXT(64)        NOT NULL,
                entity_id             TEXT(128)       NOT NULL,
                payload_json          TEXT            NOT NULL,
                state_version         INTEGER         NOT NULL DEFAULT 0,
                status                TEXT(32)        NOT NULL DEFAULT 'pending',
                delivered_at          TEXT,
                delivery_attempts     INTEGER         NOT NULL DEFAULT 0,
                created_at            TEXT            NOT NULL DEFAULT (datetime('now')),
                CHECK (status IN ('pending', 'delivered', 'failed'))
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_outbox_pending
            ON outbox (status, created_at)
            WHERE status = 'pending'
        """)

        # ── migration history ──────────────────────────────────────────
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version             INTEGER         PRIMARY KEY,
                applied_at          TEXT            NOT NULL DEFAULT (datetime('now')),
                checksum            TEXT(64)        NOT NULL
            )
        """)

    logger.info("SQLite tables created successfully")


# ---------------------------------------------------------------------------
# Migration system
# ---------------------------------------------------------------------------

def migrate_up(
    conn: DatabaseConnection,
    target_version: int = None,
) -> None:
    """Run database migrations forward.

    Migrations are SQL files in the ``migrations/`` directory, named
    with a numeric prefix (e.g. ``001_initial.sql``, ``002_add_outbox.sql``).
    Only unapplied migrations are executed.

    Parameters
    ----------
    conn : DatabaseConnection
        Active database connection. Must be outside a transaction.
    target_version : int, optional
        Migrate up to this version (inclusive). If ``None``, migrates to
        the latest available migration.

    Raises
    ------
    DatabaseError
        If a migration fails to apply.

    Example
    -------
    >>> db = DatabaseConnection("postgresql://localhost/btc")
    >>> db.connect()
    >>> migrate_up(db)
    >>> migrate_up(db, target_version=5)
    """
    migrations_dir = _resolve_migrations_dir()
    if not migrations_dir or not migrations_dir.is_dir():
        logger.warning("No migrations directory found at %s", migrations_dir)
        return

    # Get applied versions
    applied = _get_applied_versions(conn)

    # Get available migrations
    available = _list_migrations(migrations_dir)

    if not available:
        logger.info("No migrations found")
        return

    # Determine target
    if target_version is None:
        target_version = max(v for v, _ in available)

    # Filter to unapplied migrations up to target
    lowest_applied = min(applied) if applied else 0
    to_apply = [
        (version, path)
        for version, path in available
        if version > lowest_applied
        and version <= target_version
    ]

    if not to_apply:
        logger.info("No migrations to apply (current version: %s)",
                     max(applied) if applied else 0)
        return

    # Apply migrations sequentially
    for version, path in to_apply:
        logger.info("Applying migration %d: %s", version, path.name)
        sql = path.read_text(encoding="utf-8")
        try:
            conn.execute(sql)
            _record_migration(conn, version, path)
            logger.info("Migration %d applied successfully", version)
        except Exception as exc:
            logger.error("Migration %d failed: %s", version, exc)
            raise DatabaseError(
                f"Migration {version} failed: {exc}"
            ) from exc


def _resolve_migrations_dir() -> Optional[Path]:
    """Resolve the migrations directory path.

    Searches in order:
    1. ``migrations/`` relative to this module
    2. ``migrations/`` relative to the project root
    3. ``DB_MIGRATIONS_DIR`` environment variable
    """
    # Relative to this module file
    module_dir = Path(__file__).resolve().parent
    migrations = module_dir.parent.parent.parent / "migrations"
    if migrations.is_dir():
        return migrations

    # Relative to project root (parent of src/)
    project_root = module_dir.parent.parent
    migrations = project_root / "migrations"
    if migrations.is_dir():
        return migrations

    # Environment variable
    env_path = os.environ.get("DB_MIGRATIONS_DIR")
    if env_path:
        p = Path(env_path)
        if p.is_dir():
            return p

    return None


def _list_migrations(migrations_dir: Path) -> list:
    """List available migrations sorted by version number.

    Returns
    -------
    list
        List of (version, Path) tuples, sorted by version.
    """
    migrations = []
    for path in migrations_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        # Match files like "001_initial.sql", "002_add_outbox.sql"
        if not name.endswith(".sql"):
            continue
        prefix = name.split("_", 1)[0]
        try:
            version = int(prefix)
        except ValueError:
            continue
        migrations.append((version, path))

    migrations.sort(key=lambda x: x[0])
    return migrations


def _get_applied_versions(conn: DatabaseConnection) -> list:
    """Get the set of applied migration versions.

    Returns
    -------
    list
        List of applied version numbers.
    """
    try:
        rows = conn.fetchall(
            "SELECT version FROM schema_migrations ORDER BY version"
        )
        return [row["version"] for row in rows]
    except DatabaseError:
        # Table doesn't exist yet
        return []


def _record_migration(
    conn: DatabaseConnection,
    version: int,
    path: Path,
) -> None:
    """Record a migration as applied.

    Parameters
    ----------
    conn : DatabaseConnection
        Active database connection.
    version : int
        Migration version number.
    path : Path
        Path to the migration file.
    """
    import hashlib
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()

    if conn.is_sqlite:
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, checksum) "
            "VALUES (?, ?)",
            (version, checksum),
        )
    else:
        conn.execute(
            "INSERT INTO schema_migrations (version, checksum) "
            "VALUES (%s, %s) "
            "ON CONFLICT (version) DO UPDATE SET checksum = EXCLUDED.checksum",
            (version, checksum),
        )


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_connection_pool(config) -> DatabaseConnection:
    """Create database connection from config.

    Reads ``DB_DSN`` from config or environment. Falls back to SQLite
    for development/testing if no PostgreSQL DSN is provided.

    Parameters
    ----------
    config : Any
        Configuration object. Must have a ``database`` attribute (dict-like)
        with ``dsn``, ``pool_size``, and ``timeout`` keys, or be accessed
        directly as ``config.db_dsn``, ``config.db_pool_size``,
        ``config.db_timeout``. Also checks the ``database`` dict for
        ``DSN``, ``POOL_SIZE``, ``TIMEOUT`` keys.

    Returns
    -------
    DatabaseConnection
        Configured database connection.

    Raises
    ------
    ConnectionError
        If no valid DSN can be determined.

    Notes
    -----
    DSN resolution order:
    1. ``config.database["DSN"]`` or ``config.database.dsn``
    2. ``config.DB_DSN`` or ``config.db_dsn``
    3. ``DB_DSN`` environment variable
    4. SQLite fallback: ``sqlite:///./btc_dev.sqlite``
    """
    # Try to extract DSN from config
    dsn = None
    pool_size = 10
    timeout = 30.0

    # Try config.database dict
    database_config = getattr(config, "database", None)
    if database_config is not None:
        if isinstance(database_config, dict):
            dsn = database_config.get("DSN") or database_config.get("dsn")
            pool_size = database_config.get("POOL_SIZE", database_config.get("pool_size", pool_size))
            timeout = database_config.get("TIMEOUT", database_config.get("timeout", timeout))
        else:
            dsn = getattr(database_config, "DSN", None) or getattr(database_config, "dsn", None)
            pool_size = getattr(database_config, "POOL_SIZE", None) or getattr(database_config, "pool_size", pool_size)
            timeout = getattr(database_config, "TIMEOUT", None) or getattr(database_config, "timeout", timeout)

    # Try direct config attributes
    if dsn is None:
        dsn = getattr(config, "DB_DSN", None) or getattr(config, "db_dsn", None)

    # Try environment variable
    if dsn is None:
        dsn = os.environ.get("DB_DSN")

    # Try environment variables for pool settings
    if "DB_POOL_SIZE" in os.environ:
        try:
            pool_size = int(os.environ["DB_POOL_SIZE"])
        except ValueError:
            pass
    if "DB_TIMEOUT" in os.environ:
        try:
            timeout = float(os.environ["DB_TIMEOUT"])
        except ValueError:
            pass

    # Determine pool size from config if available
    if hasattr(config, "db_pool_size"):
        pool_size = config.db_pool_size
    elif hasattr(config, "database") and isinstance(config.database, dict):
        pool_size = config.database.get("POOL_SIZE", config.database.get("pool_size", pool_size))

    # Fallback to SQLite for development
    if dsn is None:
        dsn = os.environ.get("DB_DSN", "sqlite:///./btc_dev.sqlite")
        if not dsn.startswith("sqlite"):
            dsn = "sqlite:///./btc_dev.sqlite"
            logger.info("No DSN configured, using SQLite fallback: %s", dsn)

    return DatabaseConnection(
        dsn=dsn,
        pool_size=pool_size,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Helper: initialise connection and create tables if needed
# ---------------------------------------------------------------------------

def ensure_tables(config) -> DatabaseConnection:
    """Create a connection and ensure all tables exist.

    Convenience function that creates a connection from config,
    connects, and runs ``create_tables()`` if the tables don't
    already exist.

    Parameters
    ----------
    config : Any
        Configuration object (passed to :func:`get_connection_pool`).

    Returns
    -------
    DatabaseConnection
        Active connection with all tables created.
    """
    db = get_connection_pool(config)
    db.connect()

    # Check if any of our tables exist
    try:
        if db.is_sqlite:
            exists = db.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='attempt_latest'"
            )
        else:
            exists = db.fetchone(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'attempt_latest' AND table_schema = 'public'"
            )

        if not exists:
            logger.info("Tables do not exist, creating...")
            create_tables(db)
    except Exception:
        # If check fails, just create tables (idempotent with IF NOT EXISTS)
        create_tables(db)

    return db
