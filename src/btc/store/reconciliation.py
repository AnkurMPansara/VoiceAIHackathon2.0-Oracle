"""Seller state reconciliation from audit ledger.

SRS STATE-05/08: Periodic reconciliation rebuilds seller state from the
current finalized attempt ledger to bound numerical drift. Reconciles
all sellers in a namespace by replaying attempt_revisions audit log
entries in commit order.

References
----------
SRS STATE-05, STATE-08, MOD-04, MOD-05.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from btc.model.stats import (
    SellerState,
    apply_contribution,
    zero_state,
)

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of a namespace reconciliation."""

    namespace: str
    sellers_processed: int
    sellers_rebuilt: int
    sellers_unchanged: int
    errors: List[str]
    dry_run: bool


def reconcile_namespace(
    config: Any,
    namespace: str,
    dry_run: bool = False,
    db_conn: Any = None,
    state_store: Any = None,
) -> ReconciliationResult:
    """Rebuild seller states from the attempt_revisions audit ledger.

    Implements periodic reconciliation (STATE-05):
    1. Query all attempt_revisions for the namespace
    2. Group by seller_id
    3. For each seller, replay revisions in commit order
    4. Compare rebuilt state with current state
    5. Update if drifted or create if missing

    Parameters
    ----------
    config : Any
        Application config with model parameters.
    namespace : str
        Namespace to reconcile.
    dry_run : bool
        If True, report what would happen without writing.
    db_conn : Any, optional
        Database connection. Required for production reconciliation.
    state_store : Any, optional
        In-memory state store for testing.

    Returns
    -------
    ReconciliationResult
        Summary of reconciliation operation.

    References
    ----------
    SRS STATE-05, STATE-08.
    """
    errors: List[str] = []
    sellers_processed = 0
    sellers_rebuilt = 0
    sellers_unchanged = 0

    k = getattr(getattr(config, "model_config", None), "k", 4)
    sigma2 = getattr(
        getattr(config, "reward_config", None), "sigma2", 1.0
    )
    d = 2 * k + 1
    compatibility_id = namespace  # namespace == compatibility_id

    if db_conn is not None:
        try:
            # Query all revisions for this namespace, ordered by commit
            query = """
                SELECT seller_id, attempt_id, source, revision,
                       phi_features, reward_value, commit_time
                FROM attempt_revisions
                WHERE model_compatibility_id = %s
                ORDER BY commit_time ASC, revision ASC
            """
            revisions = db_conn.fetchall(query, (compatibility_id,))

            # Group by seller
            seller_revisions: Dict[str, List[Dict]] = {}
            for rev in revisions:
                sid = str(rev.get("seller_id", ""))
                if not sid:
                    continue
                if sid not in seller_revisions:
                    seller_revisions[sid] = []
                seller_revisions[sid].append(rev)

            sellers_processed = len(seller_revisions)

            # Rebuild state for each seller
            for seller_id, seller_revs in seller_revisions.items():
                state = zero_state(d, state_version=0)

                for rev in seller_revs:
                    phi_data = rev.get("phi_features")
                    reward_val = rev.get("reward_value")

                    if phi_data is None or reward_val is None:
                        continue

                    # Parse phi features (stored as JSON array)
                    import json

                    try:
                        phi_list = json.loads(phi_data)
                        phi = np.array(phi_list, dtype=np.float64)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    # Apply contribution
                    state = apply_contribution(
                        state, phi, float(reward_val), sigma2
                    )

                # Compare with current state
                if state_store is not None:
                    current = state_store.get(
                        compatibility_id, seller_id
                    )
                    if current is None:
                        # Seller missing — create
                        if not dry_run:
                            state_store.set(
                                compatibility_id, seller_id, state
                            )
                        sellers_rebuilt += 1
                    else:
                        # Compare A, b, n
                        if (
                            np.allclose(current.A, state.A, atol=1e-9)
                            and np.allclose(current.b, state.b, atol=1e-9)
                            and current.n == state.n
                        ):
                            sellers_unchanged += 1
                        else:
                            # Drifted — update
                            if not dry_run:
                                state.state_version = (
                                    current.state_version + 1
                                )
                                state_store.set(
                                    compatibility_id, seller_id, state
                                )
                            sellers_rebuilt += 1

        except Exception as exc:
            errors.append(f"Reconciliation failed: {exc}")
    else:
        if not dry_run:
            errors.append(
                "db_conn required for production reconciliation; "
                "use dry_run=True for simulation"
            )

    return ReconciliationResult(
        namespace=namespace,
        sellers_processed=sellers_processed,
        sellers_rebuilt=sellers_rebuilt,
        sellers_unchanged=sellers_unchanged,
        errors=errors,
        dry_run=dry_run,
    )
