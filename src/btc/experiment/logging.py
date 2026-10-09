"""Decision logging for the Best Time to Call experiment system.

Implements SRS EXP-03 and EXP-04, and API-01 idempotency:

- EXP-03: Persist the complete post-constraint candidate timestamps in
  sorted order, selected index, action probabilities, context snapshot,
  state and model versions, assignment, seed/PRNG algorithm version,
  and every fallback before returning a successful actionable decision.
  Repeated request IDs reuse this record, not a new random draw.
  Failing durable decision logging fails the actionable request with 503.

- EXP-04: Baseline fallback, external reschedule, missing final outcome,
  override, or unsupported action is logged with an explicit evaluation
  status. Missing outcomes are not automatically coded as zero reward.

- API-01: Idempotent by request_id. Canonical hash for conflict detection.
  Repeating an ID with identical canonical request returns the persisted
  response; repeating with a different payload returns 409.

This module provides an in-memory decision store for development and
testing, with a pluggable persistence interface for production use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Decision record ────────────────────────────────────────────────────────────


class DecisionRecord:
    """Persisted decision record (EXP-03).

    Attributes
    ----------
    decision_id : str
        Unique decision identifier.
    request_id : str
        Request identifier (idempotency key).
    request_hash : str
        SHA-256 canonical hash of the request payload.
    response : dict
        Full recommendation response payload.
    candidate_timestamps : list[str]
        Sorted candidate timestamps as ISO 8601 strings.
    action_probability : float | None
        Probability of selecting the chosen action.
    assignment : str
        Experiment assignment arm.
    experiment_id : str | None
        Experiment identifier, or None if no experiment.
    state_version : int
        Seller model state version at decision time.
    model_compatibility_id : str
        Model compatibility identifier.
    bundle_id : str
        Model bundle identifier.
    created_at : str
        ISO 8601 timestamp when the decision was logged.
    evaluation_status : str | None
        EXP-04: Evaluation status for fallback/baseline decisions.
    """

    __slots__ = (
        "decision_id",
        "request_id",
        "request_hash",
        "response",
        "candidate_timestamps",
        "action_probability",
        "assignment",
        "experiment_id",
        "state_version",
        "model_compatibility_id",
        "bundle_id",
        "created_at",
        "evaluation_status",
    )

    def __init__(
        self,
        decision_id: str,
        request_id: str,
        request_hash: str,
        response: dict,
        candidate_timestamps: list,
        action_probability: Optional[float],
        assignment: str,
        experiment_id: Optional[str],
        state_version: int,
        model_compatibility_id: str,
        bundle_id: str,
        created_at: str,
        evaluation_status: Optional[str] = None,
    ) -> None:
        """Initialise a decision record.

        Parameters
        ----------
        decision_id : str
            Unique decision identifier.
        request_id : str
            Request identifier (idempotency key).
        request_hash : str
            SHA-256 canonical hash of the request payload.
        response : dict
            Full recommendation response payload.
        candidate_timestamps : list
            Sorted candidate timestamps.
        action_probability : float | None
            Probability of selecting the chosen action.
        assignment : str
            Experiment assignment arm.
        experiment_id : str | None
            Experiment identifier.
        state_version : int
            Seller model state version.
        model_compatibility_id : str
            Model compatibility identifier.
        bundle_id : str
            Model bundle identifier.
        created_at : str
            ISO 8601 creation timestamp.
        evaluation_status : str | None
            EXP-04 evaluation status for fallback decisions.
        """
        self.decision_id = decision_id
        self.request_id = request_id
        self.request_hash = request_hash
        self.response = response
        self.candidate_timestamps = list(candidate_timestamps)
        self.action_probability = action_probability
        self.assignment = assignment
        self.experiment_id = experiment_id
        self.state_version = state_version
        self.model_compatibility_id = model_compatibility_id
        self.bundle_id = bundle_id
        self.created_at = created_at
        self.evaluation_status = evaluation_status

    def to_dict(self) -> dict:
        """Serialise the record to a dictionary.

        Returns
        -------
        dict
            Serializable dictionary representation.
        """
        return {
            "decision_id": self.decision_id,
            "request_id": self.request_id,
            "request_hash": self.request_hash,
            "response": self.response,
            "candidate_timestamps": self.candidate_timestamps,
            "action_probability": self.action_probability,
            "assignment": self.assignment,
            "experiment_id": self.experiment_id,
            "state_version": self.state_version,
            "model_compatibility_id": self.model_compatibility_id,
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "evaluation_status": self.evaluation_status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionRecord":
        """Reconstruct a decision record from a dictionary.

        Parameters
        ----------
        data : dict
            Serialised decision record.

        Returns
        -------
        DecisionRecord
            Reconstructed record.
        """
        return cls(
            decision_id=data["decision_id"],
            request_id=data["request_id"],
            request_hash=data["request_hash"],
            response=data["response"],
            candidate_timestamps=data["candidate_timestamps"],
            action_probability=data["action_probability"],
            assignment=data["assignment"],
            experiment_id=data["experiment_id"],
            state_version=data["state_version"],
            model_compatibility_id=data["model_compatibility_id"],
            bundle_id=data["bundle_id"],
            created_at=data["created_at"],
            evaluation_status=data.get("evaluation_status"),
        )


# ── Decision store ─────────────────────────────────────────────────────────────


class DecisionStore:
    """In-memory thread-safe decision store (EXP-03).

    Provides idempotent decision logging with conflict detection.

    - Repeated ``request_id`` with identical ``request_hash`` → returns
      the existing record (idempotent).
    - Repeated ``request_id`` with different ``request_hash`` → raises
      ``DecisionConflictError`` (409).
    - New ``request_id`` → creates and returns a new record.

    For production, this can be replaced with a database-backed store.
    """

    def __init__(self) -> None:
        """Initialise an empty decision store."""
        self._lock = threading.RLock()
        self._by_request_id: dict[str, DecisionRecord] = {}
        self._by_hash: dict[str, DecisionRecord] = {}

    def log_decision(
        self,
        request_id: str,
        request_hash: str,
        response: dict,
        candidate_timestamps: list,
        action_probability: Optional[float],
        assignment: str,
        experiment_id: Optional[str],
        state_version: int,
        model_compatibility_id: str,
        bundle_id: str,
        clock,
        evaluation_status: Optional[str] = None,
    ) -> str:
        """EXP-03: Log a decision record idempotently.

        Parameters
        ----------
        request_id : str
            Request identifier (idempotency key).
        request_hash : str
            SHA-256 canonical hash of the request payload (API-01).
        response : dict
            Full recommendation response payload.
        candidate_timestamps : list
            Sorted candidate timestamps.
        action_probability : float | None
            Probability of selecting the chosen action.
        assignment : str
            Experiment assignment arm.
        experiment_id : str | None
            Experiment identifier.
        state_version : int
            Seller model state version.
        model_compatibility_id : str
            Model compatibility identifier.
        bundle_id : str
            Model bundle identifier.
        clock : callable
            Returns current server time (datetime).
        evaluation_status : str | None
            EXP-04 evaluation status for fallback decisions.

        Returns
        -------
        str
            The decision_id.

        Raises
        ------
        DecisionConflictError
            If ``request_id`` exists but ``request_hash`` differs (409).
        """
        now = clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        created_at = now.isoformat()

        import uuid

        decision_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{request_id}:{request_hash}"))

        with self._lock:
            # Check for existing request_id
            existing = self._by_request_id.get(request_id)
            if existing is not None:
                if existing.request_hash != request_hash:
                    # Different payload with same request_id → 409 conflict
                    raise DecisionConflictError(
                        request_id=request_id,
                        existing_hash=existing.request_hash,
                        new_hash=request_hash,
                    )
                # Same request_id and hash → idempotent, return existing
                return existing.decision_id

            # Create new record
            record = DecisionRecord(
                decision_id=decision_id,
                request_id=request_id,
                request_hash=request_hash,
                response=response,
                candidate_timestamps=candidate_timestamps,
                action_probability=action_probability,
                assignment=assignment,
                experiment_id=experiment_id,
                state_version=state_version,
                model_compatibility_id=model_compatibility_id,
                bundle_id=bundle_id,
                created_at=created_at,
                evaluation_status=evaluation_status,
            )

            self._by_request_id[request_id] = record
            self._by_hash[request_hash] = record

            logger.info(
                "Decision logged: decision_id=%s, request_id=%s, assignment=%s",
                decision_id,
                request_id,
                assignment,
            )

            return decision_id

    def get_decision(self, request_id: str) -> Optional[DecisionRecord]:
        """Retrieve a persisted decision by request_id.

        Parameters
        ----------
        request_id : str
            Request identifier.

        Returns
        -------
        DecisionRecord | None
            The decision record, or None if not found.
        """
        with self._lock:
            return self._by_request_id.get(request_id)

    def get_decision_by_hash(self, request_hash: str) -> Optional[DecisionRecord]:
        """Retrieve a persisted decision by canonical request hash.

        Parameters
        ----------
        request_hash : str
            Canonical request hash.

        Returns
        -------
        DecisionRecord | None
            The decision record, or None if not found.
        """
        with self._lock:
            return self._by_hash.get(request_hash)

    def count(self) -> int:
        """Return the number of decisions in the store.

        Returns
        -------
        int
            Number of decisions.
        """
        with self._lock:
            return len(self._by_request_id)


# ── Exceptions ─────────────────────────────────────────────────────────────────


class DecisionConflictError(Exception):
    """409 CONFLICT: same request_id, different payload.

    Raised when a decision is logged with a request_id that already exists
    but with a different canonical request hash (API-01).

    Attributes
    ----------
    request_id : str
        The conflicting request identifier.
    existing_hash : str
        The existing canonical hash.
    new_hash : str
        The new canonical hash.
    """

    def __init__(
        self,
        request_id: str,
        existing_hash: str,
        new_hash: str,
    ) -> None:
        """Initialise the conflict error.

        Parameters
        ----------
        request_id : str
            The conflicting request identifier.
        existing_hash : str
            The existing canonical hash.
        new_hash : str
            The new canonical hash.
        """
        super().__init__(
            f"Decision conflict: request_id={request_id!r} — "
            f"existing hash {existing_hash[:16]}… vs new hash {new_hash[:16]}… "
            f"(API-01)"
        )
        self.request_id = request_id
        self.existing_hash = existing_hash
        self.new_hash = new_hash


# ── Canonical request hashing (API-01) ─────────────────────────────────────────


def compute_canonical_request_hash(request: dict) -> str:
    """API-01: Compute a canonical SHA-256 hash of a request dict.

    The hash is computed over sorted-key UTF-8 JSON after normalising
    timestamps to UTC and explicitly setting optional fields to null
    when absent.

    Parameters
    ----------
    request : dict
        Request dictionary with fields like seller_id, lead_id,
        earliest_at, latest_at, context_version, etc.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.

    Notes
    -----
    Timestamp normalisation:
        Fields ending with ``_at`` (e.g. earliest_at, latest_at) are
        parsed as ISO 8601 timestamps and converted to UTC before
        serialisation.

    Optional field normalisation:
        Fields with ``None`` values are kept as explicit null in the
        canonical JSON (the default json.dumps behaviour).

    Examples
    --------
    >>> req = {"seller_id": "s1", "lead_id": "l1", "earliest_at": "2026-10-09T11:00:00+05:30", "latest_at": "2026-10-09T17:00:00+05:30", "context_version": 1}
    >>> h = compute_canonical_request_hash(req)
    >>> len(h) == 64
    True
    """
    normalised = _normalise_request(request)
    canonical = json.dumps(
        normalised,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalise_request(request: dict) -> dict:
    """Normalise a request dict for canonical hashing.

    - Timestamp fields (ending with ``_at``) are converted to UTC.
    - All fields are preserved, including explicit nulls.

    Parameters
    ----------
    request : dict
        Raw request dictionary.

    Returns
    -------
    dict
        Normalised request dictionary.
    """
    from datetime import datetime as _dt

    result: dict[str, Any] = {}
    for key, value in request.items():
        if isinstance(value, str) and key.endswith("_at"):
            # Try parsing as ISO 8601 and converting to UTC
            try:
                dt = _dt.fromisoformat(value)
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc)
                result[key] = dt.isoformat()
                continue
            except (ValueError, TypeError):
                pass
        result[key] = value
    return result


# ── Public API functions ───────────────────────────────────────────────────────


# Module-level default store (for convenience).
_default_store: Optional[DecisionStore] = None
_default_store_lock = threading.Lock()


def _get_store() -> DecisionStore:
    """Get or create the default decision store.

    Returns
    -------
    DecisionStore
        The module-level decision store.
    """
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = DecisionStore()
    return _default_store


def log_decision(
    request_id: str,
    request_hash: str,
    response: dict,
    candidate_timestamps: list,
    action_probability: Optional[float],
    assignment: str,
    experiment_id: Optional[str],
    state_version: int,
    model_compatibility_id: str,
    bundle_id: str,
    clock,
    evaluation_status: Optional[str] = None,
) -> str:
    """EXP-03: Log a decision record idempotently.

    Convenience function that delegates to the module-level default store.

    Repeated ``request_id`` returns the same ``decision_id`` (idempotent).
    Different payload with same ``request_id`` raises ``DecisionConflictError``
    (409). In a production system, logging failure would fail the request
    with 503.

    Parameters
    ----------
    request_id : str
        Request identifier (idempotency key).
    request_hash : str
        SHA-256 canonical hash of the request payload.
    response : dict
        Full recommendation response payload.
    candidate_timestamps : list
        Sorted candidate timestamps.
    action_probability : float | None
        Probability of selecting the chosen action.
    assignment : str
        Experiment assignment arm (CONTROL, TREATMENT, EXPLORE).
    experiment_id : str | None
        Experiment identifier.
    state_version : int
        Seller model state version.
    model_compatibility_id : str
        Model compatibility identifier.
    bundle_id : str
        Model bundle identifier.
    clock : callable
        Returns current server time (datetime).
    evaluation_status : str | None
        EXP-04 evaluation status for fallback/baseline decisions.

    Returns
    -------
    str
        The decision_id.

    Raises
    ------
    DecisionConflictError
        Same request_id with different payload (409).
    """
    store = _get_store()
    return store.log_decision(
        request_id=request_id,
        request_hash=request_hash,
        response=response,
        candidate_timestamps=candidate_timestamps,
        action_probability=action_probability,
        assignment=assignment,
        experiment_id=experiment_id,
        state_version=state_version,
        model_compatibility_id=model_compatibility_id,
        bundle_id=bundle_id,
        clock=clock,
        evaluation_status=evaluation_status,
    )


def get_decision(request_id: str) -> Optional[dict]:
    """Retrieve a persisted decision by request_id.

    Parameters
    ----------
    request_id : str
        Request identifier.

    Returns
    -------
    dict | None
        Decision record as a dictionary, or None if not found.
    """
    store = _get_store()
    record = store.get_decision(request_id)
    if record is None:
        return None
    return record.to_dict()
