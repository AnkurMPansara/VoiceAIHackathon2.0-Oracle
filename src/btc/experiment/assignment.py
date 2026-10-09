"""Experiment assignment for the Best Time to Call system.

Implements SRS EXP-01: seller-level experiment arm assignment using
SHA-256 based deterministic hashing.

Assignment fractions (release 1):
    CONTROL  [0.00, 0.45)  → 45%
    TREATMENT [0.45, 0.95) → 50%
    EXPLORE  [0.95, 1.00) → 5%

Assignment algorithm (EXP-01):
    1. Canonical JSON: ``json.dumps([experiment_id, salt, seller_id],
       sort_keys=False, separators=(",", ":"), ensure_ascii=True)``
    2. SHA-256 hash of the UTF-8 encoded canonical JSON.
    3. First 8 bytes interpreted as unsigned big-endian integer.
    4. Fraction = integer / 2^64.
    5. Fraction in [0, 0.45)  → CONTROL
       Fraction in [0.45, 0.95) → TREATMENT
       Fraction in [0.95, 1.0)  → EXPLORE

Fractions and salt are immutable within an experiment. Changing them
requires a new experiment ID.

Assignment is per-seller, not per-call: the same seller always maps to
the same arm within an experiment.
"""

from __future__ import annotations

import hashlib
import json
import logging
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AssignmentArm(str, Enum):
    """Experiment assignment arm (EXP-01).

    Attributes
    ----------
    CONTROL
        Control arm — receives approved baseline behaviour.
    TREATMENT
        Treatment arm — deterministic policy (EXPLOIT).
    EXPLORE
        Exploration arm — uniform sampling over candidates.
    """

    CONTROL = "CONTROL"
    TREATMENT = "TREATMENT"
    EXPLORE = "EXPLORE"


# ── Constants ──────────────────────────────────────────────────────────────────

_CONTROL_BOUNDARY = 0.45
"""Upper bound (exclusive) for the CONTROL arm."""

_TREATMENT_BOUNDARY = 0.95
"""Upper bound (exclusive) for the TREATMENT arm; lower bound (inclusive) for EXPLORE."""

_MAX_INT_64 = 2 ** 64
"""Divisor for normalising the hash to [0, 1)."""


def assign_seller(experiment_id: str, salt: str, seller_id: str) -> str:
    """EXP-01: Assign a seller to an experiment arm.

    Computes a deterministic SHA-256 hash over the canonical JSON
    representation of ``[experiment_id, salt, seller_id]`` and maps
    the resulting fraction to CONTROL, TREATMENT, or EXPLORE.

    The same ``(experiment_id, salt, seller_id)`` triple always yields
    the same arm.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier.
    salt : str
        Immutable salt for the experiment. Changing the salt requires
        a new experiment ID.
    seller_id : str
        Seller identifier to assign.

    Returns
    -------
    str
        One of ``"CONTROL"``, ``"TREATMENT"``, or ``"EXPLORE"``.

    Raises
    ------
    ValueError
        If any argument is empty or not a string.

    Examples
    --------
    >>> assign_seller("exp-1", "salt", "seller-42")
    'CONTROL'
    >>> assign_seller("exp-1", "salt", "seller-42") == assign_seller("exp-1", "salt", "seller-42")
    True
    """
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be a non-empty string")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")
    if not isinstance(seller_id, str) or not seller_id:
        raise ValueError("seller_id must be a non-empty string")

    # Step 1: Canonical JSON — list with three elements, no whitespace.
    canonical = json.dumps(
        [experiment_id, salt, seller_id],
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    # Step 2: SHA-256 hash.
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()

    # Step 3: First 8 bytes as unsigned big-endian integer.
    hash_int = int.from_bytes(digest[:8], byteorder="big")

    # Step 4: Normalise to [0, 1).
    fraction = hash_int / _MAX_INT_64

    # Step 5: Map to arm.
    if fraction < _CONTROL_BOUNDARY:
        arm = AssignmentArm.CONTROL
    elif fraction < _TREATMENT_BOUNDARY:
        arm = AssignmentArm.TREATMENT
    else:
        arm = AssignmentArm.EXPLORE

    logger.debug(
        "Assigned seller %r to arm %s (fraction=%.6f, experiment=%r)",
        seller_id,
        arm.value,
        fraction,
        experiment_id,
    )

    return arm.value


def assign_seller_with_probability(
    experiment_id: str, salt: str, seller_id: str
) -> tuple[str, float]:
    """EXP-01: Assign a seller and return the arm assignment probability.

    The assignment probability is the arm fraction: 0.45 for CONTROL,
    0.50 for TREATMENT, 0.05 for EXPLORE.

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier.
    salt : str
        Immutable salt for the experiment.
    seller_id : str
        Seller identifier to assign.

    Returns
    -------
    tuple[str, float]
        (arm, assignment_probability) where arm is one of CONTROL,
        TREATMENT, EXPLORE and assignment_probability is the arm fraction.
    """
    arm = assign_seller(experiment_id, salt, seller_id)

    if arm == AssignmentArm.CONTROL:
        prob = _CONTROL_BOUNDARY
    elif arm == AssignmentArm.TREATMENT:
        prob = _TREATMENT_BOUNDARY - _CONTROL_BOUNDARY
    else:
        prob = 1.0 - _TREATMENT_BOUNDARY

    return arm, prob


def is_experiment_enabled(config: Any) -> bool:
    """Check if the experiment is enabled in the configuration.

    Looks for ``experiment_enabled`` on the config object. Returns
    ``False`` if the attribute is absent or falsy.

    Parameters
    ----------
    config : Any
        Configuration object. Must have an ``experiment_enabled``
        attribute (bool).

    Returns
    -------
    bool
        ``True`` if the experiment is enabled, ``False`` otherwise.

    Examples
    --------
    >>> class _Cfg:
    ...     experiment_enabled = True
    >>> is_experiment_enabled(_Cfg())
    True
    >>> is_experiment_enabled(object())
    False
    """
    enabled = getattr(config, "experiment_enabled", False)
    return bool(enabled)


def resolve_assignment(
    config: Any,
    experiment_id: str,
    salt: str,
    seller_id: str,
) -> str:
    """Resolve the experiment assignment for a seller.

    If the experiment is disabled in the config, returns ``CONTROL``
    (SRS §10.2: when experiment is disabled in live mode, return
    approved baseline behaviour as CONTROL with null experiment ID).

    Parameters
    ----------
    config : Any
        Configuration object.
    experiment_id : str
        Experiment identifier.
    salt : str
        Experiment salt.
    seller_id : str
        Seller identifier.

    Returns
    -------
    str
        One of CONTROL, TREATMENT, or EXPLORE.
    """
    if not is_experiment_enabled(config):
        return AssignmentArm.CONTROL.value

    return assign_seller(experiment_id, salt, seller_id)
