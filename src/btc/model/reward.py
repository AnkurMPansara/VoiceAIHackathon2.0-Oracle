"""Reward computation for the Best Time to Call prediction system.

Implements SRS MOD-01: utility score computation for finalised call attempts.

This module provides pure functions for computing reward values based on
outcome labels (answered, meeting_fixed, disposition) and configurable
weights. The reward is a MOD-01 utility score (not a probability) — values
can be negative and the constant dial cost does not alter ranking of candidates.

Utility score formula:
    y = w_meeting * meeting_fixed + w_answered * answered - c_dial - w_not_interested * I(NOT_INTERESTED)

DATA-04 cross-field constraints are enforced via ``validate_outcome_consistency``.
"""

from __future__ import annotations

from btc.config import RewardConfig


# Valid disposition labels recognised by this module.
_VALID_DISPOSITIONS = frozenset({
    "MEETING_FIXED",
    "NOT_INTERESTED",
    "GENERAL",
    "CALL_LATER_BUSY",
    "NOT_ANSWERED",
    "UNKNOWN",
})


def validate_outcome_consistency(
    answered: bool,
    meeting_fixed: bool,
    disposition: str,
) -> tuple[bool, str]:
    """Validate cross-field consistency of outcome labels.

    DATA-04: Enforces semantic constraints between answered, meeting_fixed,
    and disposition fields.

    Parameters
    ----------
    answered : bool
        Whether the call was answered.
    meeting_fixed : bool
        Whether a meeting was fixed.
    disposition : str
        The disposition label.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
        is_valid=True means no constraints violated.
        is_valid=False means violation found; error_message explains.

    Examples
    --------
    >>> validate_outcome_consistency(True, True, 'MEETING_FIXED')
    (True, '')
    >>> validate_outcome_consistency(False, True, 'MEETING_FIXED')
    (False, 'meeting_fixed=True requires answered=True')
    >>> validate_outcome_consistency(True, False, 'NOT_ANSWERED')
    (False, 'NOT_ANSWERED disposition requires answered=False')
    """
    if meeting_fixed and not answered:
        return (False, "meeting_fixed=True requires answered=True")

    if meeting_fixed and disposition != "MEETING_FIXED":
        return (False, "meeting_fixed=True requires disposition='MEETING_FIXED'")

    if disposition == "NOT_ANSWERED" and answered:
        return (False, "NOT_ANSWERED disposition requires answered=False")

    if disposition == "MEETING_FIXED" and not answered:
        return (False, "disposition='MEETING_FIXED' requires answered=True")

    return (True, "")


def compute_reward(
    answered: bool,
    meeting_fixed: bool,
    disposition: str,
    reward_config: RewardConfig,
) -> float:
    """Compute the MOD-01 utility score for a single finalised attempt.

    MOD-01 Formula:
        y = w_meeting * meeting_fixed
          + w_answered * answered
          - c_dial
          - w_not_interested * I(disposition == NOT_INTERESTED)

    This is a utility score (NOT a probability). Values can be negative.
    The constant -c_dial does not affect ranking of candidate actions.

    Parameters
    ----------
    answered : bool
        Whether the call was answered by the seller.
    meeting_fixed : bool
        Whether a meeting was fixed during this call.
    disposition : str
        The disposition label. Must be one of:
        'MEETING_FIXED', 'NOT_INTERESTED', 'GENERAL',
        'CALL_LATER_BUSY', 'NOT_ANSWERED', 'UNKNOWN'
    reward_config : RewardConfig
        Configuration with w_meeting, w_answered, c_dial, w_not_interested.

    Returns
    -------
    float
        MOD-01 utility score (NOT a probability, can be negative).
        Examples with defaults (w_meeting=1.0, w_answered=0.1, c_dial=0.02, w_not_interested=0.0):
        - Unanswered call: -0.02
        - Answered, no meeting: 0.08
        - Meeting fixed: 1.08

    Raises
    ------
    ValueError
        If disposition is invalid, or if cross-field constraints violated:
        - meeting_fixed=True requires answered=True and disposition='MEETING_FIXED'
        - answered=False implies disposition cannot be 'MEETING_FIXED'

    Notes
    -----
    - Constant cost (-c_dial) does not alter ranking of candidates
    - Reward changes create new model compatibility ID (handled elsewhere)
    - All reward changes require rebuilding statistics
    - This is a Gaussian working approximation for discrete bounded outcomes
    """
    # Validate disposition is a recognised label.
    if disposition not in _VALID_DISPOSITIONS:
        raise ValueError(
            f"Invalid disposition {disposition!r}. "
            f"Must be one of: {sorted(_VALID_DISPOSITIONS)}"
        )

    # Enforce DATA-04 cross-field constraints.
    is_valid, error_msg = validate_outcome_consistency(answered, meeting_fixed, disposition)
    if not is_valid:
        raise ValueError(error_msg)

    # MOD-01 utility score computation.
    reward = (
        reward_config.w_meeting * (1.0 if meeting_fixed else 0.0)
        + reward_config.w_answered * (1.0 if answered else 0.0)
        - reward_config.c_dial
    )

    # Apply NOT_INTERESTED penalty indicator.
    if disposition == "NOT_INTERESTED":
        reward -= reward_config.w_not_interested

    return float(reward)


def batch_compute_rewards(
    outcomes: list[dict],
    reward_config: RewardConfig,
) -> list[float]:
    """Compute rewards for a batch of outcomes.

    Parameters
    ----------
    outcomes : list[dict]
        Each dict must contain: 'answered', 'meeting_fixed', 'disposition'
    reward_config : RewardConfig
        Reward configuration.

    Returns
    -------
    list[float]
        Computed rewards for each outcome.

    Raises
    ------
    ValueError
        If any outcome has invalid or inconsistent labels.
        Raises on first invalid outcome encountered.
    """
    rewards: list[float] = []
    for i, outcome in enumerate(outcomes):
        try:
            answered = outcome["answered"]
            meeting_fixed = outcome["meeting_fixed"]
            disposition = outcome["disposition"]
        except KeyError as exc:
            raise ValueError(
                f"Outcome at index {i} is missing required field: {exc}"
            ) from exc

        reward = compute_reward(answered, meeting_fixed, disposition, reward_config)
        rewards.append(reward)

    return rewards
