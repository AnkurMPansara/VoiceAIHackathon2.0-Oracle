"""Dynamic retry decision engine for the Best Time to Call system.

Implements SRS RET-01 through RET-04:

- RET-01: Precedence order for retry decisions (suppression > terminal
  disposition > expiry > attempt cap > seller cap > callback > disposition
  rule > calendar projection).
- RET-02: Retry intervals are anchored to ``call_end_time``. Decision
  table covers all disposition combinations.
- RET-03: Stale proposal handling and unsupported bin projection.
- RET-04: Idempotent retry decisions keyed by ``(source, attempt_id, revision)``.

All functions are PURE — ``clock`` and ``rng`` are injected.
No network or database access.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from btc.config import Config
from btc.schemas import Disposition, RetryResult
from btc.model.posterior import Posterior

from btc.retry.calendar import (
    BusinessCalendar,
    find_next_available_slot,
    is_working_day,
    is_within_call_window,
    next_working_day,
)


# ── Constants ──────────────────────────────────────────────────────────────────

_POLICY_VERSION = "1.0.0"
"""Current retry policy version."""

_STALE_CALLBACK_HOURS = 24
"""Hours after which a seller-requested callback is considered stale."""


# ── RetryPlan ──────────────────────────────────────────────────────────────────


@dataclass
class RetryPlan:
    """Retry decision result.

    Attributes
    ----------
    result : str
        One of ``STOP``, ``MANUAL_REVIEW``, ``SUPERSEDED``, ``SCHEDULED``.
    scheduled_at : datetime or None
        Proposed retry timestamp, or None when no automatic retry.
    reason_code : str
        Machine-readable reason for the decision.
    policy_version : str
        Version of the retry policy used.
    requested_callback_at : datetime or None
        Echo of the seller-requested callback if present.
    """

    result: str
    scheduled_at: Optional[datetime]
    reason_code: str
    policy_version: str = _POLICY_VERSION
    requested_callback_at: Optional[datetime] = None


# ── Terminal disposition check ─────────────────────────────────────────────────


def is_terminal_disposition(disposition: str) -> bool:
    """RET-01: Check if disposition produces no automatic retry.

    Terminal dispositions are those that indicate the lead is
    effectively closed: the seller has agreed to a meeting,
    declined contact, or given a generic disposition.

    Parameters
    ----------
    disposition : str
        Disposition string (e.g. ``"MEETING_FIXED"``, ``"NOT_INTERESTED"``).

    Returns
    -------
    bool
        True if no automatic retry should be scheduled.
    """
    terminal = {
        Disposition.MEETING_FIXED.value,
        Disposition.NOT_INTERESTED.value,
        Disposition.GENERAL.value,
    }
    return disposition in terminal


# ── Main retry planning ───────────────────────────────────────────────────────


def plan_retry(
    latest_outcome: dict,
    context: dict,
    config: Config,
    calendar: BusinessCalendar,
    clock,
    rng,
) -> RetryPlan:
    """RET-01/02: Main retry planning function.

    Precedence order (RET-01):
    1. Suppression/opt-out or inactive lead
    2. Terminal disposition (Meeting Fixed, Not Interested, General)
    3. Lead expiry
    4. Total-attempt limit
    5. Seller daily cap / reservations
    6. Explicit callback
    7. Disposition rule
    8. Calendar/support/deadline projection

    Decision table (RET-02):
    - Any + suppressed/terminal/expired/cap → STOP
    - Meeting Fixed / Not Interested / General → STOP
    - Unknown → MANUAL_REVIEW
    - Call Later / Busy + valid callback → callback timestamp
    - Call Later / Busy + stale callback → MANUAL_REVIEW
    - Call Later / Busy + no callback → search curve peaks or call_end+120m
    - Not Answered (attempt 1) → call_end+15m (or double-call random)
    - Not Answered (attempt >=2) → next eligible working day at peak

    Parameters
    ----------
    latest_outcome : dict
        The most recent attempt outcome, containing:
        - ``disposition``: str, the disposition code.
        - ``call_end_time``: datetime, when the call ended.
        - ``attempt_number``: int, current attempt number.
        - ``answered``: bool, whether the call was answered.
        - ``requested_callback_at``: datetime or None.
        - ``meeting_fixed``: bool, whether a meeting was fixed.
        - ``seller_id``: str, seller identifier.
        - ``lead_id``: str, lead identifier.
        - ``lead_sent_time``: datetime or None.
    context : dict
        Execution context containing:
        - ``suppressed``: bool, whether the lead is suppressed.
        - ``lead_expired``: bool, whether the lead has expired.
        - ``lead_expiry``: datetime or None, expiry timestamp.
        - ``total_attempts``: int, total attempts for this lead.
        - ``seller_daily_calls``: int, calls today for this seller.
        - ``posterior``: Posterior or None, seller posterior.
        - ``sigma2``: float, noise variance.
        - ``segment_data``: list[dict], segment attempt data.
        - ``segment_key``: str, segment identifier.
    config : Config
        System configuration.
    calendar : BusinessCalendar
        Business calendar configuration.
    clock : callable
        Returns current server time (datetime). Injected for testing.
    rng : numpy.random.Generator
        Random number generator (injected for reproducibility).

    Returns
    -------
    RetryPlan
        Retry decision with result, scheduled time, and reason code.
    """
    disposition = latest_outcome.get("disposition", "")
    call_end_time = latest_outcome.get("call_end_time")
    attempt_number = latest_outcome.get("attempt_number", 1)
    answered = latest_outcome.get("answered", False)
    requested_callback_at = latest_outcome.get("requested_callback_at")
    lead_sent_time = latest_outcome.get("lead_sent_time")

    now = clock()

    # ── 1. Suppression / opt-out ─────────────────────────────────────────

    if context.get("suppressed", False):
        return RetryPlan(
            result=RetryResult.STOP,
            scheduled_at=None,
            reason_code="LEAD_SUPPRESSED",
        )

    # ── 2. Terminal disposition ──────────────────────────────────────────

    if is_terminal_disposition(disposition):
        return RetryPlan(
            result=RetryResult.STOP,
            scheduled_at=None,
            reason_code=f"TERMINAL_DISPOSITION_{disposition}",
        )

    # ── 3. Lead expiry ───────────────────────────────────────────────────

    lead_expiry = context.get("lead_expiry")
    if lead_expiry is not None:
        if lead_expiry.tzinfo is None:
            lead_expiry = lead_expiry.replace(tzinfo=calendar.timezone and timezone.utc or timezone.utc)
        if lead_expiry <= now:
            return RetryPlan(
                result=RetryResult.STOP,
                scheduled_at=None,
                reason_code="LEAD_EXPIRED",
            )

    if context.get("lead_expired", False):
        return RetryPlan(
            result=RetryResult.STOP,
            scheduled_at=None,
            reason_code="LEAD_EXPIRED",
        )

    # ── 4. Total-attempt limit ───────────────────────────────────────────

    max_attempts = config.max_attempts_per_lead
    total_attempts = context.get("total_attempts", 0)
    if total_attempts >= max_attempts:
        return RetryPlan(
            result=RetryResult.STOP,
            scheduled_at=None,
            reason_code="MAX_ATTEMPTS_REACHED",
        )

    # ── 5. Seller daily cap / reservations ───────────────────────────────

    max_daily = config.max_calls_per_seller_per_day
    seller_daily_calls = context.get("seller_daily_calls", 0)
    if seller_daily_calls >= max_daily:
        return RetryPlan(
            result=RetryResult.STOP,
            scheduled_at=None,
            reason_code="SELLER_DAILY_CAP_REACHED",
        )

    # ── 6. Explicit callback ─────────────────────────────────────────────

    if requested_callback_at is not None:
        return resolve_callback(
            requested_callback_at=requested_callback_at,
            context=context,
            config=config,
            calendar=calendar,
            clock=clock,
        )

    # ── 7. Disposition rule ──────────────────────────────────────────────

    if disposition == Disposition.UNKNOWN.value:
        return RetryPlan(
            result=RetryResult.MANUAL_REVIEW,
            scheduled_at=None,
            reason_code="UNKNOWN_DISPOSITION",
        )

    if disposition in (Disposition.CALL_LATER_BUSY.value,):
        return _handle_call_later_busy(
            call_end_time=call_end_time,
            context=context,
            config=config,
            calendar=calendar,
            clock=clock,
            rng=rng,
        )

    if disposition == Disposition.NOT_ANSWERED.value:
        return _handle_not_answered(
            call_end_time=call_end_time,
            attempt_number=attempt_number,
            context=context,
            config=config,
            calendar=calendar,
            clock=clock,
            rng=rng,
        )

    # Fallback for any unrecognized disposition
    return RetryPlan(
        result=RetryResult.MANUAL_REVIEW,
        scheduled_at=None,
        reason_code=f"UNRECOGNIZED_DISPOSITION_{disposition}",
    )


# ── Disposition handlers ───────────────────────────────────────────────────────


def _handle_call_later_busy(
    call_end_time,
    context: dict,
    config: Config,
    calendar: BusinessCalendar,
    clock,
    rng,
) -> RetryPlan:
    """Handle CALL_LATER_BUSY disposition.

    Decision table (RET-02):
    - Has valid callback → schedule at callback time
    - Has stale callback → MANUAL_REVIEW
    - No callback → search curve peaks or call_end+120m

    Parameters
    ----------
    call_end_time : datetime
        When the call ended.
    context : dict
        Execution context.
    config : Config
        System configuration.
    calendar : BusinessCalendar
        Business calendar.
    clock : callable
        Server time.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    RetryPlan
        Retry decision.
    """
    requested_callback_at = context.get("requested_callback_at")
    if requested_callback_at is None:
        requested_callback_at = None  # Explicit None for clarity

    # Check if there's a callback request in the outcome
    outcome_callback = None
    # Check context for the callback from the latest outcome
    # The callback would have been checked in step 6 of plan_retry,
    # but we need to handle the case where it's in the outcome dict
    outcome = context.get("_latest_outcome", {})
    if isinstance(outcome, dict):
        outcome_callback = outcome.get("requested_callback_at")

    effective_callback = requested_callback_at or outcome_callback

    if effective_callback is not None:
        # Validate the callback is not stale
        now = clock()
        if effective_callback.tzinfo is None:
            effective_callback = effective_callback.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        delta = effective_callback - now
        if delta.total_seconds() > _STALE_CALLBACK_HOURS * 3600:
            return RetryPlan(
                result=RetryResult.MANUAL_REVIEW,
                scheduled_at=None,
                reason_code="STALE_CALLBACK",
                requested_callback_at=effective_callback,
            )

        # Project to eligible slot
        projected = project_to_eligible(
            proposed_time=effective_callback,
            context=context,
            config=config,
            calendar=calendar,
            clock=clock,
        )
        if projected is not None:
            return RetryPlan(
                result=RetryResult.SCHEDULED,
                scheduled_at=projected,
                reason_code="CALLBACK_SCHEDULED",
                requested_callback_at=effective_callback,
            )
        else:
            return RetryPlan(
                result=RetryResult.MANUAL_REVIEW,
                scheduled_at=None,
                reason_code="CALLBACK_NO_ELIGIBLE_SLOT",
                requested_callback_at=effective_callback,
            )

    # No callback: search curve peaks or call_end+120m
    if call_end_time is None:
        return RetryPlan(
            result=RetryResult.MANUAL_REVIEW,
            scheduled_at=None,
            reason_code="NO_CALL_END_TIME",
        )

    posterior = context.get("posterior")
    sigma2 = context.get("sigma2", 0.06)

    if posterior is not None:
        peak = search_curve_peaks(
            call_end_time=call_end_time,
            context=context,
            config=config,
            calendar=calendar,
            posterior=posterior,
            sigma2=sigma2,
        )
        if peak is not None:
            return RetryPlan(
                result=RetryResult.SCHEDULED,
                scheduled_at=peak,
                reason_code="CURVE_PEAK_SCHEDULED",
            )

    # Fallback: call_end + 120 minutes
    fallback = call_end_time + timedelta(minutes=120)
    projected = project_to_eligible(
        proposed_time=fallback,
        context=context,
        config=config,
        calendar=calendar,
        clock=clock,
    )
    if projected is not None:
        return RetryPlan(
            result=RetryResult.SCHEDULED,
            scheduled_at=projected,
            reason_code="CALL_LATER_FALLBACK_120M",
        )

    return RetryPlan(
        result=RetryResult.MANUAL_REVIEW,
        scheduled_at=None,
        reason_code="CALL_LATER_NO_SLOT",
    )


def _handle_not_answered(
    call_end_time,
    attempt_number: int,
    context: dict,
    config: Config,
    calendar: BusinessCalendar,
    clock,
    rng,
) -> RetryPlan:
    """Handle NOT_ANSWERED disposition.

    Decision table (RET-02):
    - Attempt 1 → call_end+15m (or double-call random if enabled)
    - Attempt >=2 → next eligible working day at peak

    Parameters
    ----------
    call_end_time : datetime
        When the call ended.
    attempt_number : int
        Current attempt number.
    context : dict
        Execution context.
    config : Config
        System configuration.
    calendar : BusinessCalendar
        Business calendar.
    clock : callable
        Server time.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    RetryPlan
        Retry decision.
    """
    if call_end_time is None:
        return RetryPlan(
            result=RetryResult.MANUAL_REVIEW,
            scheduled_at=None,
            reason_code="NO_CALL_END_TIME",
        )

    # Attempt 1: quick retry (double-call)
    if attempt_number <= 1:
        if config.double_call_enabled:
            # Double-call: random offset within [1, 5] minutes
            jitter_minutes = rng.integers(1, 6)
            retry_time = call_end_time + timedelta(minutes=jitter_minutes)
        else:
            # Standard: call_end + 15 minutes
            retry_time = call_end_time + timedelta(minutes=15)

        projected = project_to_eligible(
            proposed_time=retry_time,
            context=context,
            config=config,
            calendar=calendar,
            clock=clock,
        )
        if projected is not None:
            return RetryPlan(
                result=RetryResult.SCHEDULED,
                scheduled_at=projected,
                reason_code="DOUBLE_CALL_RETRY" if config.double_call_enabled else "NOT_ANSWERED_RETRY_15M",
            )
        return RetryPlan(
            result=RetryResult.MANUAL_REVIEW,
            scheduled_at=None,
            reason_code="NOT_ANSWERED_NO_SLOT",
        )

    # Attempt >= 2: next eligible working day at peak
    posterior = context.get("posterior")
    sigma2 = context.get("sigma2", 0.06)

    # Search for curve peaks in the future
    if posterior is not None:
        peak = search_curve_peaks(
            call_end_time=call_end_time,
            context=context,
            config=config,
            calendar=calendar,
            posterior=posterior,
            sigma2=sigma2,
        )
        if peak is not None:
            return RetryPlan(
                result=RetryResult.SCHEDULED,
                scheduled_at=peak,
                reason_code="NOT_ANSWERED_PEAK_RETRY",
            )

    # Fallback: next working day at start_hour
    next_day = next_working_day(call_end_time, calendar)
    projected = project_to_eligible(
        proposed_time=next_day,
        context=context,
        config=config,
        calendar=calendar,
        clock=clock,
    )
    if projected is not None:
        return RetryPlan(
            result=RetryResult.SCHEDULED,
            scheduled_at=projected,
            reason_code="NOT_ANSWERED_NEXT_DAY",
        )

    return RetryPlan(
        result=RetryResult.MANUAL_REVIEW,
        scheduled_at=None,
        reason_code="NOT_ANSWERED_NO_SLOT",
    )


# ── Callback resolution ────────────────────────────────────────────────────────


def resolve_callback(
    requested_callback_at: datetime,
    context: dict,
    config: Config,
    calendar: BusinessCalendar,
    clock,
) -> RetryPlan:
    """Handle explicit seller-requested callback.

    Validates and projects the seller's requested callback time
    to the next eligible slot respecting calendar constraints.

    RET-03: If the requested time is in the past or unsupported,
    project to the next available slot.

    Parameters
    ----------
    requested_callback_at : datetime
        The callback time requested by the seller.
    context : dict
        Execution context.
    config : Config
        System configuration.
    calendar : BusinessCalendar
        Business calendar.
    clock : callable
        Server time.

    Returns
    -------
    RetryPlan
        Retry decision.
    """
    now = clock()

    # Normalize timezone
    if requested_callback_at.tzinfo is None:
        requested_callback_at = requested_callback_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Check staleness
    delta = requested_callback_at - now
    if delta.total_seconds() > _STALE_CALLBACK_HOURS * 3600:
        return RetryPlan(
            result=RetryResult.MANUAL_REVIEW,
            scheduled_at=None,
            reason_code="STALE_CALLBACK",
            requested_callback_at=requested_callback_at,
        )

    # Check if the callback is in the past
    if requested_callback_at <= now:
        # Project to next available slot
        projected = project_to_eligible(
            proposed_time=requested_callback_at,
            context=context,
            config=config,
            calendar=calendar,
            clock=clock,
        )
        if projected is not None:
            return RetryPlan(
                result=RetryResult.SCHEDULED,
                scheduled_at=projected,
                reason_code="PAST_CALLBACK_PROJECTED",
                requested_callback_at=requested_callback_at,
            )
        return RetryPlan(
            result=RetryResult.MANUAL_REVIEW,
            scheduled_at=None,
            reason_code="PAST_CALLBACK_NO_SLOT",
            requested_callback_at=requested_callback_at,
        )

    # Project to eligible slot
    projected = project_to_eligible(
        proposed_time=requested_callback_at,
        context=context,
        config=config,
        calendar=calendar,
        clock=clock,
    )
    if projected is not None:
        return RetryPlan(
            result=RetryResult.SCHEDULED,
            scheduled_at=projected,
            reason_code="CALLBACK_SCHEDULED",
            requested_callback_at=requested_callback_at,
        )

    return RetryPlan(
        result=RetryResult.MANUAL_REVIEW,
        scheduled_at=None,
        reason_code="CALLBACK_NO_ELIGIBLE_SLOT",
        requested_callback_at=requested_callback_at,
    )


# ── Curve peak search ──────────────────────────────────────────────────────────


def search_curve_peaks(
    call_end_time: datetime,
    context: dict,
    config: Config,
    calendar: BusinessCalendar,
    posterior,
    sigma2: float,
) -> Optional[datetime]:
    """Search for peaks in expected-curve within ``[call_end+60m, call_end+240m]``.

    Evaluates the posterior expected reward at 15-minute grid points
    within the search window and returns the time with the highest
    expected reward that falls on a working day within the call window.

    Parameters
    ----------
    call_end_time : datetime
        When the last call ended (anchor point).
    context : dict
        Execution context (may contain segment_data, segment_key).
    config : Config
        System configuration.
    calendar : BusinessCalendar
        Business calendar.
    posterior : Posterior
        Seller posterior for expected reward computation.
    sigma2 : float
        Working noise variance.

    Returns
    -------
    datetime or None
        The peak time if found, None otherwise.
    """
    from btc.features.fourier import fourier, time_to_hours
    from btc.model.posterior import predict_expected_reward

    # Search window: [call_end + 60min, call_end + 240min]
    search_start = call_end_time + timedelta(minutes=60)
    search_end = call_end_time + timedelta(minutes=240)

    # Generate 15-minute grid
    grid: list[datetime] = []
    current = search_start.replace(second=0, microsecond=0)
    minute = current.minute
    if minute % 15 != 0:
        next_minute = (minute // 15 + 1) * 15
        if next_minute >= 60:
            current = (current + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0,
            )
        else:
            current = current.replace(minute=next_minute, second=0, microsecond=0)

    while current <= search_end:
        grid.append(current)
        current += timedelta(minutes=15)

    if not grid:
        return None

    # Score each candidate
    best_time = None
    best_reward = float("-inf")

    for ts in grid:
        # Check calendar constraints
        if not is_working_day(ts, calendar):
            continue
        if not is_within_call_window(ts, calendar):
            continue

        # Compute expected reward
        try:
            hours = time_to_hours(ts)
            k = (posterior.d - 1) // 2
            phi = fourier([hours], k=k)[0]
            reward = predict_expected_reward(phi, posterior)
        except Exception:
            continue

        if not isinstance(reward, (int, float)):
            continue

        if reward > best_reward:
            best_reward = reward
            best_time = ts

    return best_time


# ── Eligible slot projection ───────────────────────────────────────────────────


def project_to_eligible(
    proposed_time: datetime,
    context: dict,
    config: Config,
    calendar: BusinessCalendar,
    clock,
) -> Optional[datetime]:
    """RET-03: Project proposed time to first eligible slot.

    If the proposed time is:
    - In the past → advance to next available slot.
    - On a non-working day → advance to next working day.
    - Outside call window → advance to next slot within window.
    - In an unsupported bin → advance to next supported bin.

    Parameters
    ----------
    proposed_time : datetime
        The initially proposed retry time.
    context : dict
        Execution context (may contain support_mask).
    config : Config
        System configuration.
    calendar : BusinessCalendar
        Business calendar.
    clock : callable
        Server time.

    Returns
    -------
    datetime or None
        The projected eligible time, or None if no slot exists.
    """
    now = clock()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if proposed_time.tzinfo is None:
        proposed_time = proposed_time.replace(tzinfo=timezone.utc)

    support_mask = context.get("support_mask", {})

    # If proposed time is in the past, find next available slot
    if proposed_time <= now:
        slot = find_next_available_slot(
            after=now,
            calendar=calendar,
            support_mask=support_mask,
            min_gap_minutes=config.minimum_inter_call_gap_minutes,
        )
        return slot

    # If not on a working day, advance to next working day
    if not is_working_day(proposed_time, calendar):
        next_day = next_working_day(proposed_time, calendar)
        slot = find_next_available_slot(
            after=next_day,
            calendar=calendar,
            support_mask=support_mask,
            min_gap_minutes=config.minimum_inter_call_gap_minutes,
        )
        return slot

    # If outside call window, find next slot within window
    if not is_within_call_window(proposed_time, calendar):
        # Move to start of next day if we're past the window
        local = proposed_time.astimezone(_KOLKATTA) if proposed_time.tzinfo else proposed_time
        if local.hour >= calendar.end_hour:
            next_day = proposed_time + timedelta(days=1)
            next_day = next_day.replace(
                hour=calendar.start_hour, minute=0, second=0, microsecond=0,
            )
            # Check if next day is a working day
            if not is_working_day(next_day, calendar):
                next_day = next_working_day(proposed_time, calendar)
            slot = find_next_available_slot(
                after=next_day,
                calendar=calendar,
                support_mask=support_mask,
                min_gap_minutes=config.minimum_inter_call_gap_minutes,
            )
            return slot
        else:
            # We're before the window, move to start of window
            kolkata_local = proposed_time.astimezone(_KOLKATTA) if proposed_time.tzinfo else proposed_time
            next_window = datetime(
                year=kolkata_local.year,
                month=kolkata_local.month,
                day=kolkata_local.day,
                hour=calendar.start_hour,
                minute=0,
                second=0,
                tzinfo=_KOLKATTA,
            )
            slot = find_next_available_slot(
                after=next_window,
                calendar=calendar,
                support_mask=support_mask,
                min_gap_minutes=config.minimum_inter_call_gap_minutes,
            )
            return slot

    # Within window and on working day — check support
    if support_mask:
        from btc.model.policy import _is_in_support_bin
        if not _is_in_support_bin(proposed_time, support_mask):
            slot = find_next_available_slot(
                after=proposed_time,
                calendar=calendar,
                support_mask=support_mask,
                min_gap_minutes=config.minimum_inter_call_gap_minutes,
            )
            return slot

    # All checks passed
    return proposed_time


# ── Idempotency wrapper ────────────────────────────────────────────────────────


def plan_retry_idempotent(
    source: str,
    attempt_id: str,
    revision: int,
    latest_outcome: dict,
    context: dict,
    config: Config,
    calendar: BusinessCalendar,
    clock,
    rng,
) -> RetryPlan:
    """RET-04: Idempotent retry decision wrapper.

    Produces deterministic results for the same ``(source, attempt_id, revision)``
    key. The ``decision_id`` in the result is derived from these inputs
    to enable deduplication.

    Parameters
    ----------
    source : str
        Producer identifier.
    attempt_id : str
        Attempt identifier.
    revision : int
        Revision number.
    latest_outcome : dict
        The most recent attempt outcome.
    context : dict
        Execution context.
    config : Config
        System configuration.
    calendar : BusinessCalendar
        Business calendar.
    clock : callable
        Server time.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    RetryPlan
        Retry decision with deterministic ``decision_id``.
    """
    # Compute deterministic decision ID for idempotency
    id_components = f"{source}:{attempt_id}:{revision}"
    decision_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_components))

    result = plan_retry(
        latest_outcome=latest_outcome,
        context=context,
        config=config,
        calendar=calendar,
        clock=clock,
        rng=rng,
    )

    # Attach decision_id to the result for tracking
    # (stored in a non-standard attribute for internal use)
    result._decision_id = decision_id  # type: ignore[attr-defined]

    return result
