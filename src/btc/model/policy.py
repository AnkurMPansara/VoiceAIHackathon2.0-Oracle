"""Recommendation policy: candidate generation, action selection, and secondary peak detection.

Implements SRS §10 (Policy agent) and POL-01 through POL-07:

- POL-01: All returned timestamps strictly in the future (server time).
- POL-02: Calendar windows start-inclusive, end-exclusive. Development:
  08:00–18:00 Asia/Kolkata, Mon–Sat, no holidays. 18:00 excluded.
- POL-03: Quarter-hour grid, max 7-day horizon, urgent earliest inclusion,
  deduplication.
- POL-04: Day with no capacity removed; no candidates → NO_ELIGIBLE_SLOT.
- POL-05: Deterministic selection maximises expected reward; ties within
  1e-12 break to earliest; cold start uses PRIOR_ONLY.
- POL-06: Exploration samples uniformly over candidates; probability = 1/n.
- POL-07: Secondary peak = local maxima on supported runs; plateau =
  earliest point; must be ≥2 h from primary; null if none.

All functions are PURE — clock and RNG are injected.
No network or database access.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import numpy.typing as npt

from btc.config import Calendar
from btc.features.fourier import fourier, time_to_hours
from btc.model.posterior import Posterior


# ── Constants ──────────────────────────────────────────────────────────────────

_QUARTER = timedelta(minutes=15)
"""Grid spacing for candidate generation (POL-03)."""

_MAX_HORIZON_DAYS = 7
"""Maximum forecast horizon in days (POL-03)."""

_TIE_TOLERANCE = 1e-12
"""Reward difference tolerance for tie-breaking (POL-05)."""

_MIN_GAP_HOURS = 2.0
"""Minimum hours between primary and secondary peak (POL-07)."""

_UTC = timezone.utc
"""UTC timezone for internal comparisons."""


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class CandidateSet:
    """Sorted list of eligible candidate timestamps with metadata.

    Attributes
    ----------
    timestamps : list[datetime]
        Sorted eligible candidates (quarter-hour grid).
    support_mask : list[bool]
        True if candidate falls in a supported time bin (TRAIN-05).
    scores : np.ndarray or None
        (n, 4) array from :func:`score_candidates`, or None if not scored.
    """

    timestamps: list[datetime] = field(default_factory=list)
    support_mask: list[bool] = field(default_factory=list)
    scores: npt.NDArray[np.float64] | None = None


@dataclass
class PolicyDecision:
    """SRS §10.2: Complete recommendation response.

    Attributes
    ----------
    decision_id : str
        Unique decision identifier.
    status : str
        RECOMMENDED, NO_ELIGIBLE_SLOT, STOP, MANUAL_REVIEW, SUPERSEDED.
    scheduled_at : datetime or None
        Recommended call timestamp (timezone-aware, Asia/Kolkata).
    secondary_at : datetime or None
        Secondary peak timestamp, or None.
    reason_code : str
        Reason for this decision.
    mode : str
        EXPLOIT, PRIOR_ONLY, UNIFORM_EXPLORE, BASELINE, NONE.
    assignment : str
        CONTROL, TREATMENT, EXPLORE, SHADOW.
    experiment_id : str or None
    candidate_count : int
        Number of eligible candidates considered.
    action_probability : float or None
        1/candidate_count for exploration, 1.0 for deterministic.
    assignment_probability : float or None
        Arm assignment probability.
    ope_eligible : bool
    n_attempts : int
        Number of finalised attempts for the seller.
    prior_weight : float
        Prior weight diagnostic.
    expected_reward : float
        Expected reward for the chosen action.
    latent_std : float
        Latent standard deviation (posterior uncertainty).
    predictive_std : float
        Predictive standard deviation (including noise).
    """

    decision_id: str = ""
    status: str = "NONE"
    scheduled_at: datetime | None = None
    secondary_at: datetime | None = None
    reason_code: str = ""
    mode: str = "NONE"
    assignment: str = "CONTROL"
    experiment_id: str | None = None
    candidate_count: int = 0
    action_probability: float | None = None
    assignment_probability: float | None = None
    ope_eligible: bool = False
    n_attempts: int = 0
    prior_weight: float = 1.0
    expected_reward: float = 0.0
    latent_std: float = 0.0
    predictive_std: float = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_utc(clock) -> datetime:
    """Return current server time from the injected clock, in UTC.

    Parameters
    ----------
    clock : callable
        Returns current server time (datetime).

    Returns
    -------
    datetime
        Current time in UTC.
    """
    now = clock()
    if now.tzinfo is None:
        return now.replace(tzinfo=_UTC)
    return now.astimezone(_UTC)


def _to_kolkata(dt: datetime) -> datetime:
    """Convert a datetime to Asia/Kolkata timezone.

    Parameters
    ----------
    dt : datetime
        Input datetime (may be naive or timezone-aware).

    Returns
    -------
    datetime
        Timezone-aware datetime in Asia/Kolkata.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(timezone(timedelta(hours=5, minutes=30)))


def _is_calendar_day(dt: datetime, calendar: Calendar) -> bool:
    """Check if a datetime falls on a calendar-allowed day.

    POL-02: Mon-Sat (ISO 0-5), excluding holidays.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.
    calendar : Calendar
        Business calendar config.

    Returns
    -------
    bool
        True if the day is allowed.
    """
    # ISO weekday: Monday=0 … Sunday=6
    iso_weekday = dt.weekday()
    if iso_weekday not in calendar.days_of_week:
        return False

    # Check holidays
    date_str = dt.strftime("%Y-%m-%d")
    if date_str in calendar.holidays:
        return False

    return True


def _is_in_calendar_window(dt: datetime, calendar: Calendar) -> bool:
    """Check if a datetime falls within the calendar's hour window.

    POL-02: start_hour inclusive, end_hour exclusive.
    18:00 is NOT permitted (end_hour is exclusive).

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime (should be in calendar's timezone).
    calendar : Calendar
        Business calendar config.

    Returns
    -------
    bool
        True if within [start_hour, end_hour).
    """
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    # Convert to minutes since midnight for precise comparison
    time_minutes = hour * 60 + minute + second / 60.0
    start_minutes = calendar.start_hour * 60
    end_minutes = calendar.end_hour * 60

    return start_minutes <= time_minutes < end_minutes


def _is_in_support_bin(dt: datetime, support_mask: dict) -> bool:
    """Check if a datetime falls in a supported time bin.

    POL-02 / TRAIN-05: support_mask maps hour_bin (integer) to is_supported.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.
    support_mask : dict
        {hour_bin: is_supported} mapping.

    Returns
    -------
    bool
        True if the hour bin is supported.
    """
    hour_bin = dt.hour
    return support_mask.get(hour_bin, False)


def _generate_grid(
    earliest: datetime,
    latest: datetime,
) -> list[datetime]:
    """Generate quarter-hour grid points in [earliest, latest].

    POL-03: Grid points at :, 00, :15, :30, :45.

    Parameters
    ----------
    earliest : datetime
        Earliest candidate time (inclusive).
    latest : datetime
        Latest candidate time (inclusive endpoint — we include up to
        the last quarter-hour <= latest).

    Returns
    -------
    list[datetime]
        Sorted list of quarter-hour aligned timestamps.
    """
    grid: list[datetime] = []
    current = earliest

    # Align to next quarter-hour if needed
    minute = current.minute
    if minute % 15 != 0:
        next_minute = (minute // 15 + 1) * 15
        if next_minute >= 60:
            # Push to next hour
            current = (current + timedelta(hours=1)).replace(
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            current = current.replace(
                minute=next_minute,
                second=0,
                microsecond=0,
            )
        # If alignment pushed past latest, nothing to generate
        if current > latest:
            return grid

    while current <= latest:
        grid.append(current)
        current += _QUARTER

    return grid


def _deduplicate(timestamps: list[datetime]) -> list[datetime]:
    """Remove exact duplicate timestamps, preserving order.

    POL-03: Deduplicate exact timestamps.

    Parameters
    ----------
    timestamps : list[datetime]
        Input timestamps (may contain duplicates).

    Returns
    -------
    list[datetime]
        Deduplicated timestamps.
    """
    seen: set[str] = set()
    result: list[datetime] = []
    for ts in timestamps:
        key = ts.isoformat()
        if key not in seen:
            seen.add(key)
            result.append(ts)
    return result


# ── POL-01/02/03/04: Candidate generation ─────────────────────────────────────


def generate_candidates(
    earliest_at: datetime,
    latest_at: datetime,
    calendar: Calendar,
    support_mask: dict,
    clock,
    max_calls_per_day: int = 3,
    calls_already_today: int = 0,
    lead_expiry: datetime = None,
    dispatch_lead_time_seconds: int = 5,
    initial_delay_minutes: int = 15,
    lead_sent_time: datetime = None,
) -> CandidateSet:
    """POL-01/02/03/04: Generate eligible candidate timestamps.

    Steps:
    1. Intersect earliest_at, latest_at, lead_expiry, calendar windows.
    2. Apply initial delay: lead_sent_time + max_initial_delay_minutes
       (if fresh lead).
    3. Generate quarter-hour grid points in [earliest, latest].
    4. Filter by calendar (Mon-Sat, 08:00-18:00, 18:00 excluded).
    5. Filter by support_mask (TRAIN-05: supported time bins).
    6. Apply daily cap: if calls_already_today >= max_calls, remove today.
    7. Max 7-day horizon.
    8. If urgent interval < 15 min, include earliest feasible timestamp.
    9. Deduplicate exact timestamps.
    10. Check support for each candidate.

    Parameters
    ----------
    earliest_at : datetime
        Earliest feasible call time.
    latest_at : datetime
        Latest feasible call time.
    calendar : Calendar
        Business calendar config.
    support_mask : dict
        Supported time bins from TRAIN-05.
    clock : callable
        Returns current server time (datetime). Injected for testing.
    max_calls_per_day : int
        Maximum calls per seller per day.
    calls_already_today : int
        Calls already placed today.
    lead_expiry : datetime or None
        Lead expiration time.
    dispatch_lead_time_seconds : int
        Minimum lead time for scheduler.
    initial_delay_minutes : int
        Minimum delay after lead sent.
    lead_sent_time : datetime or None
        When lead was sent to dialer.

    Returns
    -------
    CandidateSet
        Sorted eligible candidates with support mask.
        Empty if no eligible slots.
    """
    now = _now_utc(clock)

    # ── Step 1: Intersect constraints ────────────────────────────────────

    effective_earliest = earliest_at
    effective_latest = latest_at

    # Lead expiry: if provided, cap latest at expiry
    if lead_expiry is not None:
        if lead_expiry.tzinfo is None:
            lead_expiry = lead_expiry.replace(tzinfo=_UTC)
        if lead_expiry < now:
            # Already expired — no candidates
            return CandidateSet()
        if lead_expiry < effective_latest:
            effective_latest = lead_expiry

    # ── Step 2: Apply initial delay ──────────────────────────────────────

    if lead_sent_time is not None:
        if lead_sent_time.tzinfo is None:
            lead_sent_time = lead_sent_time.replace(tzinfo=_UTC)
        delay_end = lead_sent_time + timedelta(minutes=initial_delay_minutes)
        if delay_end > effective_earliest:
            effective_earliest = delay_end

    # ── POL-01: Ensure earliest is in the future ─────────────────────────

    # If the original interval is entirely in the past, return empty
    original_latest = latest_at
    if original_latest.tzinfo is None:
        original_latest = original_latest.replace(tzinfo=_UTC)
    if effective_latest <= now:
        return CandidateSet()

    if effective_earliest <= now:
        effective_earliest = now + timedelta(seconds=1)

    # ── POL-03: Max 7-day horizon ────────────────────────────────────────

    max_latest = now + timedelta(days=_MAX_HORIZON_DAYS)
    if effective_latest > max_latest:
        effective_latest = max_latest

    # ── Urgent: include earliest feasible if interval < 15 min ────────────

    urgent = False
    interval_minutes = (effective_latest - effective_earliest).total_seconds() / 60.0
    if interval_minutes < 15.0:
        urgent = True

    # ── Step 3: Generate quarter-hour grid ───────────────────────────────

    grid = _generate_grid(effective_earliest, effective_latest)

    # ── Step 8: If urgent, include earliest feasible timestamp ───────────

    if urgent and not grid:
        # Earliest feasible is just after now
        grid = [effective_earliest]

    # ── Step 4: Filter by calendar (day + hour window) ───────────────────

    filtered: list[datetime] = []
    for ts in grid:
        # Convert to calendar timezone for checks
        ts_local = _to_kolkata(ts)
        if not _is_calendar_day(ts_local, calendar):
            continue
        if not _is_in_calendar_window(ts_local, calendar):
            continue
        filtered.append(ts)

    # ── Step 9: Deduplicate ──────────────────────────────────────────────

    filtered = _deduplicate(filtered)

    # ── Step 6: Apply daily cap ──────────────────────────────────────────

    if calls_already_today >= max_calls_per_day:
        # Group by local date and remove all candidates from today
        if filtered:
            today_kolkata = _to_kolkata(now).date()
            remaining: list[datetime] = []
            for ts in filtered:
                ts_date = _to_kolkata(ts).date()
                if ts_date != today_kolkata:
                    remaining.append(ts)
            filtered = remaining

    # ── Step 5: Filter by support_mask ───────────────────────────────────

    supported: list[datetime] = []
    support_flags: list[bool] = []
    for ts in filtered:
        ts_local = _to_kolkata(ts)
        if _is_in_support_bin(ts_local, support_mask):
            supported.append(ts)
            support_flags.append(True)

    # If no supported candidates, fall back to keeping all (marked unsupported)
    if not supported:
        for ts in filtered:
            ts_local = _to_kolkata(ts)
            supported.append(ts)
            support_flags.append(False)

    # ── POL-04: Check for empty candidates ───────────────────────────────

    if not supported:
        return CandidateSet()

    return CandidateSet(
        timestamps=supported,
        support_mask=support_flags,
        scores=None,
    )


# ── POL-05: Deterministic action selection ────────────────────────────────────


def select_action_deterministic(
    candidates: CandidateSet,
    posterior: Posterior,
    sigma2: float,
) -> PolicyDecision:
    """POL-05: Select action by maximizing posterior expected reward.

    Ties within 1e-12 choose earliest timestamp.
    Cold start (n=0) uses PRIOR_ONLY mode.

    Parameters
    ----------
    candidates : CandidateSet
        Eligible candidate timestamps.
    posterior : Posterior
        Seller posterior.
    sigma2 : float
        Working noise variance.

    Returns
    -------
    PolicyDecision
        Selected action with scores.

    Raises
    ------
    ValueError
        If candidates is empty.
    """
    if not candidates.timestamps:
        raise ValueError("candidates is empty")

    # ── Cold start: PRIOR_ONLY mode ──────────────────────────────────────

    is_cold = posterior.is_cold_start

    # ── Score all candidates ─────────────────────────────────────────────

    if candidates.scores is None:
        candidates.scores = _score_from_posterior(candidates, posterior, sigma2)

    scores = candidates.scores  # (n, 4): [expected_reward, latent_std, predictive_std, prior_weight]

    # ── Find max expected reward ─────────────────────────────────────────

    expected_rewards = scores[:, 0]
    max_reward = float(np.max(expected_rewards))

    # ── POL-05: Tie-breaking — earliest among those within tolerance ─────

    best_idx = -1
    for i in range(len(expected_rewards)):
        if expected_rewards[i] >= max_reward - _TIE_TOLERANCE:
            best_idx = i
            break  # First one wins (earliest)

    best_timestamp = candidates.timestamps[best_idx]

    # ── Build decision ───────────────────────────────────────────────────

    decision = PolicyDecision(
        decision_id=str(uuid.uuid4()),
        status="RECOMMENDED",
        scheduled_at=best_timestamp,
        secondary_at=None,
        reason_code="PRIOR_ONLY" if is_cold else "MAX_EXPECTED_REWARD",
        mode="PRIOR_ONLY" if is_cold else "EXPLOIT",
        assignment="CONTROL",
        experiment_id=None,
        candidate_count=len(candidates.timestamps),
        action_probability=1.0,
        assignment_probability=None,
        ope_eligible=True,
        n_attempts=posterior.n,
        prior_weight=float(scores[best_idx, 3]),
        expected_reward=float(expected_rewards[best_idx]),
        latent_std=float(scores[best_idx, 1]),
        predictive_std=float(scores[best_idx, 2]),
    )

    return decision


def _score_from_posterior(
    candidates: CandidateSet,
    posterior: Posterior,
    sigma2: float,
) -> npt.NDArray[np.float64]:
    """Score candidates using posterior directly.

    Uses the posterior's mu and L to compute expected rewards and
    uncertainty for all candidates at once.

    Parameters
    ----------
    candidates : CandidateSet
        Candidate timestamps.
    posterior : Posterior
        Computed posterior.
    sigma2 : float
        Working noise variance.

    Returns
    -------
    np.ndarray, shape (n, 4)
        [expected_reward, latent_std, predictive_std, prior_weight].
    """
    n = len(candidates.timestamps)
    if n == 0:
        return np.empty((0, 4), dtype=np.float64)

    hours = np.array(
        [time_to_hours(ts) for ts in candidates.timestamps],
        dtype=np.float64,
    )
    k = (posterior.d - 1) // 2
    phi_matrix = fourier(hours, k=k)

    # Expected reward = phi @ mu
    expected_rewards = phi_matrix @ posterior.mu

    # Latent std = sqrt(diag(phi @ Lambda^{-1} @ phi^T))
    L_inv_phi_T = np.linalg.solve(posterior.L, phi_matrix.T)
    latent_vars = np.sum(L_inv_phi_T ** 2, axis=0)
    latent_stds = np.sqrt(np.maximum(0.0, latent_vars))

    # Predictive std = sqrt(latent_std^2 + sigma2)
    predictive_stds = np.sqrt(np.maximum(0.0, latent_stds ** 2 + sigma2))

    # Prior weight
    prior_weight = posterior.prior_weight

    scores = np.empty((n, 4), dtype=np.float64)
    scores[:, 0] = expected_rewards
    scores[:, 1] = latent_stds
    scores[:, 2] = predictive_stds
    scores[:, 3] = prior_weight

    return scores


# ── POL-06: Exploration action selection ──────────────────────────────────────


def select_action_explore(
    candidates: CandidateSet,
    rng,
) -> PolicyDecision:
    """POL-06: Select action by uniform random over candidates.

    Probability = 1/candidate_count.

    Parameters
    ----------
    candidates : CandidateSet
        Eligible candidate timestamps.
    rng : numpy.random.Generator
        Random number generator (injected for reproducibility).

    Returns
    -------
    PolicyDecision
        Selected action with uniform probability.

    Raises
    ------
    ValueError
        If candidates is empty.
    """
    if not candidates.timestamps:
        raise ValueError("candidates is empty")

    n = len(candidates.timestamps)
    prob = 1.0 / n

    # Uniform random selection
    idx = rng.integers(0, n)
    selected = candidates.timestamps[idx]

    return PolicyDecision(
        decision_id=str(uuid.uuid4()),
        status="RECOMMENDED",
        scheduled_at=selected,
        secondary_at=None,
        reason_code="UNIFORM_EXPLORE",
        mode="UNIFORM_EXPLORE",
        assignment="EXPLORE",
        experiment_id=None,
        candidate_count=n,
        action_probability=prob,
        assignment_probability=None,
        ope_eligible=True,
        n_attempts=0,
        prior_weight=1.0,
        expected_reward=0.0,
        latent_std=0.0,
        predictive_std=0.0,
    )


# ── POL-07: Secondary peak detection ──────────────────────────────────────────


def find_secondary_peak(
    expected_rewards: npt.NDArray[np.float64],
    support_mask: list,
    primary_index: int,
    min_gap_hours: float = _MIN_GAP_HOURS,
) -> int:
    """POL-07: Find secondary peak (local maximum) on supported runs.

    Rules:
    - Plateau = one peak at earliest point of plateau.
    - Endpoint can be peak if strictly better than its only neighbor.
    - Wholly flat run has no peak.
    - Must be at least min_gap_hours from primary.
    - Return -1 if no valid secondary peak.

    Parameters
    ----------
    expected_rewards : np.ndarray
        Expected rewards for each candidate.
    support_mask : list[bool]
        True for supported time bins.
    primary_index : int
        Index of primary peak.
    min_gap_hours : float
        Minimum hours between primary and secondary.

    Returns
    -------
    int
        Index of secondary peak, or -1 if none.
    """
    n = len(expected_rewards)
    if n == 0:
        return -1

    # Calculate candidate spacing in hours from grid
    # Grid is quarter-hour, so spacing = 0.25 hours
    grid_spacing_hours = 0.25

    # Compute minimum index gap corresponding to min_gap_hours
    min_gap_indices = max(1, int(np.ceil(min_gap_hours / grid_spacing_hours)))

    # Identify plateau regions: runs of consecutive equal values.
    # A plateau is defined by (start, end, value) where all indices
    # in [start, end] have the same reward value.
    plateaus: list[tuple[int, int, float]] = []
    i = 0
    while i < n:
        j = i
        while j < n and expected_rewards[j] == expected_rewards[i]:
            j += 1
        # Plateau from i to j-1 with value expected_rewards[i]
        plateaus.append((i, j - 1, float(expected_rewards[i])))
        i = j

    # Determine which plateaus are peaks.
    # A plateau is a peak if its value is strictly greater than its
    # neighbors (the plateau immediately before and after, or boundary).
    peak_indices: list[int] = []

    for p_idx, (p_start, p_end, p_val) in enumerate(plateaus):
        # Check if this plateau is supported (at least one point)
        is_supported = any(support_mask[k] for k in range(p_start, p_end + 1))
        if not is_supported:
            continue

        # Determine neighbor values
        left_val: float | None = None
        right_val: float | None = None

        if p_idx > 0:
            left_val = plateaus[p_idx - 1][2]
        if p_idx < len(plateaus) - 1:
            right_val = plateaus[p_idx + 1][2]

        is_peak = False

        if left_val is not None and right_val is not None:
            # Interior plateau: peak if strictly greater than both neighbors
            if p_val > left_val and p_val > right_val:
                is_peak = True
        elif left_val is not None:
            # Right endpoint plateau: peak if strictly greater than left neighbor
            if p_val > left_val:
                is_peak = True
        elif right_val is not None:
            # Left endpoint plateau: peak if strictly greater than right neighbor
            if p_val > right_val:
                is_peak = True
        else:
            # Single plateau (wholly flat run) — no peak
            is_peak = False

        if is_peak:
            # Plateau peak is at the earliest point
            peak_indices.append(p_start)

    if not peak_indices:
        return -1

    # Filter peaks by minimum distance from primary
    valid_peaks: list[int] = []
    for p in peak_indices:
        if p == primary_index:
            continue
        if abs(p - primary_index) >= min_gap_indices:
            valid_peaks.append(p)

    if not valid_peaks:
        return -1

    # Return the peak with highest expected reward
    # Ties broken by earliest index
    best_peak = valid_peaks[0]
    best_reward = expected_rewards[best_peak]

    for p in valid_peaks[1:]:
        if expected_rewards[p] > best_reward + _TIE_TOLERANCE:
            best_reward = expected_rewards[p]
            best_peak = p

    return best_peak


# ── Main recommendation entry point ───────────────────────────────────────────


def recommend(
    candidates: CandidateSet,
    posterior: Posterior,
    assignment: str,
    sigma2: float,
    rng,
    clock,
) -> PolicyDecision:
    """Main recommendation entry point.

    Resolves mode based on assignment and seller state:
    - EXPLORE assignment → select_action_explore
    - TREATMENT/CONTROL with cold start → PRIOR_ONLY
    - TREATMENT/CONTROL with data → EXPLOIT
    - SHADOW → same as TREATMENT but with ope_eligible=False

    POL-01: All timestamps strictly in the future.
    POL-04: No candidates → NO_ELIGIBLE_SLOT.

    Parameters
    ----------
    candidates : CandidateSet
        Eligible candidates.
    posterior : Posterior
        Seller posterior.
    assignment : str
        CONTROL, TREATMENT, EXPLORE, or SHADOW.
    sigma2 : float
        Working noise variance.
    rng : numpy.random.Generator
        Random number generator.
    clock : callable
        Server time.

    Returns
    -------
    PolicyDecision
        Complete recommendation.
    """
    now = _now_utc(clock)

    # ── POL-04: No candidates → NO_ELIGIBLE_SLOT ─────────────────────────

    if not candidates.timestamps:
        return PolicyDecision(
            decision_id=str(uuid.uuid4()),
            status="NO_ELIGIBLE_SLOT",
            scheduled_at=None,
            secondary_at=None,
            reason_code="NO_ELIGIBLE_SLOT",
            mode="NONE",
            assignment=assignment,
            experiment_id=None,
            candidate_count=0,
            action_probability=None,
            assignment_probability=None,
            ope_eligible=False,
            n_attempts=posterior.n,
            prior_weight=posterior.prior_weight,
            expected_reward=0.0,
            latent_std=0.0,
            predictive_std=0.0,
        )

    # ── Resolve mode based on assignment and seller state ────────────────

    is_cold = posterior.is_cold_start

    if assignment == "EXPLORE":
        # POL-06: Exploration arm
        decision = select_action_explore(candidates, rng)
        decision.assignment = assignment
        decision.assignment_probability = 1.0  # uniform arm assignment
    elif assignment == "SHADOW":
        # Shadow: same logic as TREATMENT but ope_eligible=False
        if is_cold:
            decision = select_action_deterministic(candidates, posterior, sigma2)
            decision.mode = "PRIOR_ONLY"
            decision.reason_code = "PRIOR_ONLY"
        else:
            decision = select_action_deterministic(candidates, posterior, sigma2)
            decision.mode = "EXPLOIT"
            decision.reason_code = "MAX_EXPECTED_REWARD"
        decision.assignment = assignment
        decision.ope_eligible = False
    elif assignment in ("TREATMENT", "CONTROL"):
        if is_cold:
            # POL-05: Cold start uses PRIOR_ONLY
            decision = select_action_deterministic(candidates, posterior, sigma2)
            decision.mode = "PRIOR_ONLY"
            decision.reason_code = "PRIOR_ONLY"
        else:
            # POL-05: Deterministic = max expected reward
            decision = select_action_deterministic(candidates, posterior, sigma2)
            decision.mode = "EXPLOIT"
            decision.reason_code = "MAX_EXPECTED_REWARD"
        decision.assignment = assignment
    else:
        # Unknown assignment — no action
        return PolicyDecision(
            decision_id=str(uuid.uuid4()),
            status="NO_ELIGIBLE_SLOT",
            scheduled_at=None,
            secondary_at=None,
            reason_code=f"UNKNOWN_ASSIGNMENT_{assignment}",
            mode="NONE",
            assignment=assignment,
            experiment_id=None,
            candidate_count=0,
            action_probability=None,
            assignment_probability=None,
            ope_eligible=False,
            n_attempts=posterior.n,
            prior_weight=posterior.prior_weight,
            expected_reward=0.0,
            latent_std=0.0,
            predictive_std=0.0,
        )

    # ── POL-07: Secondary peak detection ─────────────────────────────────

    if decision.scheduled_at is not None and candidates.scores is not None:
        scores = candidates.scores
        expected_rewards = scores[:, 0]

        # Find primary index (index of scheduled_at in candidates)
        primary_index = -1
        scheduled_iso = decision.scheduled_at.isoformat()
        for i, ts in enumerate(candidates.timestamps):
            if ts.isoformat() == scheduled_iso:
                primary_index = i
                break

        if primary_index >= 0:
            secondary_index = find_secondary_peak(
                expected_rewards=expected_rewards,
                support_mask=candidates.support_mask,
                primary_index=primary_index,
                min_gap_hours=_MIN_GAP_HOURS,
            )
            if secondary_index >= 0:
                decision.secondary_at = candidates.timestamps[secondary_index]

    # ── POL-01: Final check — all timestamps must be in the future ───────

    if decision.scheduled_at is not None:
        sched = decision.scheduled_at
        if sched.tzinfo is None:
            sched = sched.replace(tzinfo=_UTC)
        if sched <= now:
            # Should not happen if generate_candidates was correct,
            # but guard anyway
            decision.status = "NO_ELIGIBLE_SLOT"
            decision.scheduled_at = None
            decision.secondary_at = None
            decision.reason_code = "NO_ELIGIBLE_SLOT"
            decision.mode = "NONE"
            decision.candidate_count = 0
            decision.action_probability = None
            decision.expected_reward = 0.0
            decision.latent_std = 0.0
            decision.predictive_std = 0.0

    if decision.secondary_at is not None:
        sec = decision.secondary_at
        if sec.tzinfo is None:
            sec = sec.replace(tzinfo=_UTC)
        if sec <= now:
            decision.secondary_at = None

    return decision
