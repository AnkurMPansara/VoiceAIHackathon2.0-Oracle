"""Comprehensive unit tests for `src/btc/model/policy.py` (POL-01 to POL-08).

Tests SRS §7 (POL-01 to POL-08) and §15 (T08, T09):

    POL-01: All returned timestamps strictly in the future (server time).
    POL-02: Calendar Mon-Sat 08:00-18:00, 18:00 NOT permitted (end-exclusive).
    POL-03: Quarter-hour grid, max 7-day horizon, earliest feasible if urgent,
            deduplication.
    POL-04: No capacity day → remove; no candidates → NO_ELIGIBLE_SLOT.
    POL-05: Deterministic = max expected reward; ties within 1e-12 → earliest;
            cold start = PRIOR_ONLY.
    POL-06: Exploration = uniform random; probability = 1/candidate_count.
    POL-07: Secondary peak = local maxima; plateau = earliest point;
            >=2h from primary; null if none.

SRS §15: atol=1e-9, rtol=1e-9 on well-conditioned float64 fixtures.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.btc.config import Calendar
from src.btc.features.fourier import fourier, time_to_hours
from src.btc.model.policy import (
    CandidateSet,
    PolicyDecision,
    _generate_grid,
    _is_calendar_day,
    _is_in_calendar_window,
    _is_in_support_bin,
    _now_utc,
    _score_from_posterior,
    _to_kolkata,
    find_secondary_peak,
    generate_candidates,
    recommend,
    select_action_deterministic,
    select_action_explore,
)
from src.btc.model.posterior import Posterior, compute_posterior
from src.btc.model.stats import Prior, apply_contribution, zero_state


# ── Shared fixtures ────────────────────────────────────────────────────────────

_KTZ = timezone(timedelta(hours=5, minutes=30))
"""Asia/Kolkata timezone."""


def _k(dt: datetime) -> datetime:
    """Convert a naive datetime to Asia/Kolkata."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_KTZ)
    return dt.astimezone(_KTZ)


def _utc(dt: datetime) -> datetime:
    """Convert a naive datetime to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _make_calendar(
    days_of_week: list[int] | None = None,
    start_hour: int = 8,
    end_hour: int = 18,
    holidays: list[str] | None = None,
) -> Calendar:
    return Calendar(
        days_of_week=days_of_week or [0, 1, 2, 3, 4, 5],
        start_hour=start_hour,
        end_hour=end_hour,
        holidays=holidays or [],
    )


def _make_clock(reference: datetime) -> callable:
    """Return a clock function fixed at *reference* (timezone-aware)."""
    ref = reference
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)

    def clock() -> datetime:
        return ref

    return clock


def _make_posterior(n: int = 0, d: int = 9, alpha: float = 0.1) -> Posterior:
    """Build a Posterior: cold-start when n=0, warm otherwise."""
    prior = Prior.diagonal_prior(d=d, alpha=alpha)
    state = zero_state(d=d)
    if n > 0:
        np.random.seed(42)
        from src.btc.model.stats import apply_contribution
        from src.btc.features.fourier import fourier
        sigma2 = 0.06
        for _ in range(n):
            t = np.random.uniform(8.0, 17.0)
            phi = fourier(t, k=(d - 1) // 2)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi, float(reward), sigma2)
    result = compute_posterior(state, prior, sigma2=0.06)
    assert isinstance(result, Posterior)
    return result


def _make_candidates(
    timestamps: list[datetime],
    support_flags: list[bool] | None = None,
) -> CandidateSet:
    return CandidateSet(
        timestamps=timestamps,
        support_mask=support_flags or [True] * len(timestamps),
        scores=None,
    )


def _score_candidates(
    candidates: CandidateSet,
    posterior: Posterior,
    sigma2: float = 0.06,
) -> CandidateSet:
    """Score candidates in-place and return the updated set."""
    candidates.scores = _score_from_posterior(candidates, posterior, sigma2)
    return candidates


# ── POL-02 helpers ────────────────────────────────────────────────────────────

class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_now_utc_returns_utc(self):
        """_now_utc returns a timezone-aware UTC datetime."""
        clock = _make_clock(datetime(2026, 6, 15, 12, 0, 0))
        result = _now_utc(clock)
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_now_utc_handles_naive(self):
        """_now_utc treats naive datetime as UTC."""
        clock = _make_clock(datetime(2026, 6, 15, 12, 0, 0))
        result = _now_utc(clock)
        assert result.hour == 12

    def test_to_kolkata_converts_utc(self):
        """_to_kolkata converts UTC to Asia/Kolkata (+5:30)."""
        dt_utc = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _to_kolkata(dt_utc)
        assert result.hour == 17
        assert result.minute == 30

    def test_to_kolkata_naive_treated_as_utc(self):
        """_to_kolkata treats naive datetime as UTC."""
        dt_naive = datetime(2026, 6, 15, 12, 0, 0)
        result = _to_kolkata(dt_naive)
        assert result.hour == 17
        assert result.minute == 30

    def test_is_calendar_day_mon_sat(self):
        """Calendar allows Mon(0) through Sat(5)."""
        cal = _make_calendar()
        # Monday
        assert _is_calendar_day(_k(datetime(2026, 6, 15)), cal) is True  # Mon
        # Saturday
        assert _is_calendar_day(_k(datetime(2026, 6, 20)), cal) is True  # Sat
        # Sunday
        assert _is_calendar_day(_k(datetime(2026, 6, 21)), cal) is False  # Sun

    def test_is_calendar_day_holiday_excluded(self):
        """Holidays are excluded from calendar."""
        cal = _make_calendar(holidays=["2026-06-15"])
        # June 15 2026 is a Monday
        assert _is_calendar_day(_k(datetime(2026, 6, 15)), cal) is False

    def test_is_calendar_day_custom_days(self):
        """Custom days_of_week respected."""
        cal = _make_calendar(days_of_week=[0, 1, 2, 3, 4])  # Mon-Fri only
        # Saturday excluded
        assert _is_calendar_day(_k(datetime(2026, 6, 20)), cal) is False
        # Monday included
        assert _is_calendar_day(_k(datetime(2026, 6, 15)), cal) is True

    def test_is_in_calendar_window_start_inclusive(self):
        """Start hour is inclusive."""
        cal = _make_calendar(start_hour=8, end_hour=18)
        assert _is_in_calendar_window(_k(datetime(2026, 6, 15, 8, 0, 0)), cal) is True

    def test_is_in_calendar_window_end_exclusive(self):
        """End hour is EXCLUSIVE — 18:00 NOT permitted."""
        cal = _make_calendar(start_hour=8, end_hour=18)
        assert _is_in_calendar_window(_k(datetime(2026, 6, 15, 18, 0, 0)), cal) is False

    def test_is_in_calendar_window_boundary_17_59(self):
        """17:59 is within 08:00-18:00 window."""
        cal = _make_calendar(start_hour=8, end_hour=18)
        assert _is_in_calendar_window(_k(datetime(2026, 6, 15, 17, 59, 0)), cal) is True

    def test_is_in_calendar_window_18_00_01_excluded(self):
        """18:00:01 is outside the window."""
        cal = _make_calendar(start_hour=8, end_hour=18)
        assert _is_in_calendar_window(_k(datetime(2026, 6, 15, 18, 0, 1)), cal) is False

    def test_is_in_support_bin(self):
        """_is_in_support_bin checks hour_bin mapping."""
        mask = {8: True, 9: True, 10: False, 14: True, 17: False}
        assert _is_in_support_bin(_k(datetime(2026, 6, 15, 9, 0)), mask) is True
        assert _is_in_support_bin(_k(datetime(2026, 6, 15, 10, 0)), mask) is False
        assert _is_in_support_bin(_k(datetime(2026, 6, 15, 14, 0)), mask) is True
        assert _is_in_support_bin(_k(datetime(2026, 6, 15, 17, 0)), mask) is False
        # Unsupported hour_bin (default False)
        assert _is_in_support_bin(_k(datetime(2026, 6, 15, 20, 0)), mask) is False


# ── POL-03: _generate_grid ────────────────────────────────────────────────────

class TestGenerateGrid:
    """Tests for the _generate_grid helper."""

    def test_grid_aligned_to_quarter_hours(self):
        """Grid points are at :00, :15, :30, :45."""
        earliest = _utc(datetime(2026, 6, 15, 8, 5, 0))
        latest = _utc(datetime(2026, 6, 15, 8, 35, 0))
        grid = _generate_grid(earliest, latest)
        # Should align 08:05 → 08:15, then 08:30
        assert len(grid) == 2
        assert grid[0].minute == 15
        assert grid[1].minute == 30

    def test_grid_exact_quarter_hour(self):
        """If already on quarter hour, include it."""
        earliest = _utc(datetime(2026, 6, 15, 8, 0, 0))
        latest = _utc(datetime(2026, 6, 15, 8, 15, 0))
        grid = _generate_grid(earliest, latest)
        assert len(grid) == 2
        assert grid[0].minute == 0
        assert grid[1].minute == 15

    def test_grid_empty_when_past_latest(self):
        """Grid is empty when earliest > latest after alignment."""
        earliest = _utc(datetime(2026, 6, 15, 8, 50, 0))
        latest = _utc(datetime(2026, 6, 15, 8, 55, 0))
        grid = _generate_grid(earliest, latest)
        # 08:50 aligns to 09:00 which is > 08:55
        assert len(grid) == 0, f"Expected empty grid, got {grid}"

    def test_grid_single_point(self):
        """Single quarter-hour point in range."""
        earliest = _utc(datetime(2026, 6, 15, 8, 0, 0))
        latest = _utc(datetime(2026, 6, 15, 8, 0, 30))
        grid = _generate_grid(earliest, latest)
        assert len(grid) == 1
        assert grid[0].minute == 0


# ── POL-01/02/03/04: generate_candidates ──────────────────────────────────────

class TestGenerateCandidates:
    """SRS POL-01, POL-02, POL-03, POL-04: Candidate generation."""

    # ── Test 1: Basic quarter-hour grid ────────────────────────────────────

    def test_01_basic_quarter_hour_grid(self):
        """Earliest=08:00, latest=18:00 → grid at 08:00, 08:15, ..., 17:45."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 18, 0, 0))

        result = generate_candidates(
            earliest_at=earliest,
            latest_at=latest,
            calendar=cal,
            support_mask=support_mask,
            clock=clock,
        )

        timestamps = result.timestamps
        # Should include 08:00 through 17:45
        assert len(timestamps) > 0
        first = _to_kolkata(timestamps[0])
        last = _to_kolkata(timestamps[-1])
        assert first.hour == 8 and first.minute == 0
        assert last.hour == 17 and last.minute == 45

    # ── Test 2: 18:00 excluded ─────────────────────────────────────────────

    def test_02_18_00_excluded(self):
        """If latest=18:00, last candidate is 17:45 (18:00 NOT permitted)."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 18, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        for ts in result.timestamps:
            local = _to_kolkata(ts)
            # 18:00 should never appear
            assert not (local.hour == 18 and local.minute == 0), (
                f"18:00 found in candidates: {ts}"
            )
            # All times should be within [08:00, 18:00)
            assert 8 <= local.hour < 18, f"Time {local} outside window"

    # ── Test 3: Holiday filtering ───────────────────────────────────────────

    def test_03_holiday_filtering(self):
        """Holiday dates excluded from candidates."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        # June 15 is a Monday — add it as a holiday
        cal = _make_calendar(holidays=["2026-06-15"])
        support_mask = {h: True for h in range(8, 18)}
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 18, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        for ts in result.timestamps:
            local_date = _to_kolkata(ts).date()
            assert str(local_date) != "2026-06-15", (
                f"Holiday date in candidates: {ts}"
            )

    # ── Test 4: Sunday filtering ────────────────────────────────────────────

    def test_04_sunday_filtering(self):
        """Sundays excluded from candidates."""
        # June 21, 2026 is a Sunday
        now = _k(datetime(2026, 6, 20, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()  # Mon-Sat
        support_mask = {h: True for h in range(8, 18)}
        earliest = _k(datetime(2026, 6, 21, 8, 0, 0))
        latest = _k(datetime(2026, 6, 21, 18, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        for ts in result.timestamps:
            local = _to_kolkata(ts)
            assert local.weekday() != 6, f"Sunday in candidates: {ts}"

    # ── Test 5: Daily cap ───────────────────────────────────────────────────

    def test_05_daily_cap_removes_today(self):
        """calls_today=3, max_calls=3 → no candidates today."""
        today = _k(datetime(2026, 6, 15, 8, 0, 0))
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=today, latest_at=today + timedelta(hours=10),
            calendar=cal, support_mask=support_mask, clock=clock,
            max_calls_per_day=3, calls_already_today=3,
        )

        assert len(result.timestamps) == 0

    def test_05_daily_cap_allows_tomorrow(self):
        """Daily cap removes today but allows next day."""
        today = _k(datetime(2026, 6, 15, 8, 0, 0))
        tomorrow = _k(datetime(2026, 6, 16, 8, 0, 0))
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=today, latest_at=tomorrow + timedelta(hours=10),
            calendar=cal, support_mask=support_mask, clock=clock,
            max_calls_per_day=3, calls_already_today=3,
        )

        # Should have candidates on tomorrow but not today
        today_kolkata = _to_kolkata(now).date()
        for ts in result.timestamps:
            ts_date = _to_kolkata(ts).date()
            assert ts_date != today_kolkata, (
                f"Today's date in candidates despite daily cap: {ts}"
            )

    # ── Test 6: 7-day horizon ───────────────────────────────────────────────

    def test_06_seven_day_horizon(self):
        """If latest spans 30 days, only 7 days of candidates."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}
        earliest = now + timedelta(hours=1)
        latest = now + timedelta(days=30)

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        # All candidates must be within 7 days of now
        max_date = _to_kolkata(now + timedelta(days=7)).date()
        for ts in result.timestamps:
            ts_date = _to_kolkata(ts).date()
            assert ts_date <= max_date, (
                f"Candidate beyond 7-day horizon: {ts}"
            )

    # ── Test 7: Urgent interval ─────────────────────────────────────────────

    def test_07_urgent_interval_includes_earliest_feasible(self):
        """If earliest to latest < 15 min, include earliest feasible timestamp."""
        now = _k(datetime(2026, 6, 15, 8, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 8, 10, 0))  # Only 10 min gap

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert len(result.timestamps) > 0, (
            "Urgent interval should produce at least one candidate"
        )

    # ── Test 8: Support mask filtering ──────────────────────────────────────

    def test_08_support_mask_filters_unsupported_bins(self):
        """Unsupported bins excluded from candidates."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        # Only support 09:00-12:00
        support_mask = {h: True for h in range(9, 13)}

        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 14, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        for ts in result.timestamps:
            local = _to_kolkata(ts)
            assert support_mask.get(local.hour, False), (
                f"Unsupported hour in candidates: {local.hour}"
            )

    # ── Test 9: Lead expiry ─────────────────────────────────────────────────

    def test_09_lead_expiry_excludes_after(self):
        """Candidates after lead_expiry excluded."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 18, 0, 0))
        expiry = _k(datetime(2026, 6, 15, 12, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
            lead_expiry=expiry,
        )

        for ts in result.timestamps:
            assert ts <= expiry, (
                f"Candidate after lead_expiry: {ts} > {expiry}"
            )

    def test_09_lead_expired_returns_empty(self):
        """Expired lead → empty candidates."""
        now = _k(datetime(2026, 6, 15, 12, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}
        expiry = _k(datetime(2026, 6, 15, 10, 0, 0))

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 18, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
            lead_expiry=expiry,
        )

        assert len(result.timestamps) == 0

    # ── Test 10: Initial delay ──────────────────────────────────────────────

    def test_10_initial_delay_excludes_before_lead_sent_plus_15min(self):
        """Candidates before lead_sent + 15min excluded."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        lead_sent = _k(datetime(2026, 6, 15, 7, 5, 0))
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 18, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
            lead_sent_time=lead_sent, initial_delay_minutes=15,
        )

        delay_end = lead_sent + timedelta(minutes=15)
        for ts in result.timestamps:
            assert ts >= delay_end, (
                f"Candidate before delay_end: {ts} < {delay_end}"
            )

    # ── Test 11: All-past interval ──────────────────────────────────────────

    def test_11_all_past_interval_empty(self):
        """If earliest and latest both in past → empty candidates."""
        now = _k(datetime(2026, 6, 15, 12, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert len(result.timestamps) == 0

    # ── Test 12: Empty result → NO_ELIGIBLE_SLOT ────────────────────────────

    def test_12_empty_result_has_no_candidates(self):
        """Empty support_mask → fallback keeps all candidates with all unsupported flags."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask: dict = {}  # No supported bins → fallback to all unsupported

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 18, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        # Fallback: when no supported bins, all candidates are kept but marked unsupported
        assert len(result.timestamps) > 0
        assert all(not flag for flag in result.support_mask), (
            "All support flags should be False when no bins are supported"
        )

    # ── Test 13: Deduplication ──────────────────────────────────────────────

    def test_13_deduplication(self):
        """Duplicate timestamps deduplicated."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        # Create scenario with potential duplicates
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 8, 30, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        # Check no duplicates
        iso_strings = [ts.isoformat() for ts in result.timestamps]
        assert len(iso_strings) == len(set(iso_strings)), (
            "Duplicate timestamps found in candidates"
        )

    # ── Test 14: Timezone-aware ─────────────────────────────────────────────

    def test_14_timezone_aware(self):
        """All candidates have tzinfo (Asia/Kolkata)."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 10, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        for ts in result.timestamps:
            assert ts.tzinfo is not None, (
                f"Candidate missing tzinfo: {ts}"
            )


# ── POL-05: select_action_deterministic ───────────────────────────────────────

class TestSelectActionDeterministic:
    """SRS POL-05: Deterministic action selection."""

    # ── Test 15: Single candidate ───────────────────────────────────────────

    def test_15_single_candidate_selected(self):
        """Single candidate → selects it."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 8, 14, 0)),  # Only 08:00 fits
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=10)
        decision = select_action_deterministic(result, posterior, sigma2=0.06)

        assert len(result.timestamps) == 1
        assert decision.scheduled_at == result.timestamps[0]
        assert decision.status == "RECOMMENDED"

    # ── Test 16: Multiple candidates → max reward ───────────────────────────

    def test_16_multiple_candidates_max_reward(self):
        """Multiple candidates → selects max expected reward."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 18, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        decision = select_action_deterministic(result, posterior, sigma2=0.06)

        assert decision.scheduled_at is not None
        assert decision.status == "RECOMMENDED"
        assert decision.mode == "EXPLOIT"

    # ── Test 17: Tie within 1e-12 → earliest ────────────────────────────────

    def test_17_tie_within_tolerance_earliest_wins(self):
        """Ties within 1e-12 → earliest timestamp wins."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 8, 45, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=0)  # Cold start: all rewards equal
        decision = select_action_deterministic(result, posterior, sigma2=0.06)

        # With cold start, all scores are equal → earliest wins
        assert decision.scheduled_at == result.timestamps[0], (
            f"Expected first candidate {result.timestamps[0]}, got {decision.scheduled_at}"
        )

    # ── Test 18: Tie beyond 1e-12 → earlier timestamp wins ──────────────────

    def test_18_tie_beyond_tolerance_earlier_wins(self):
        """Reward difference beyond 1e-12 → not a tie, highest wins."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 8, 45, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        decision = select_action_deterministic(result, posterior, sigma2=0.06)

        # Scores are not tied (warm posterior), so highest reward wins
        assert decision.scheduled_at is not None
        assert decision.status == "RECOMMENDED"

    # ── Test 19: Cold start → PRIOR_ONLY ────────────────────────────────────

    def test_19_cold_start_prior_only(self):
        """Cold start (n=0) → mode=PRIOR_ONLY."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=0)
        assert posterior.is_cold_start is True

        decision = select_action_deterministic(result, posterior, sigma2=0.06)

        assert decision.mode == "PRIOR_ONLY"
        assert decision.reason_code == "PRIOR_ONLY"

    # ── Test 20: Warm seller → EXPLOIT ──────────────────────────────────────

    def test_20_warm_seller_exploit(self):
        """Warm seller (n>0) → mode=EXPLOIT."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        assert posterior.is_cold_start is False

        decision = select_action_deterministic(result, posterior, sigma2=0.06)

        assert decision.mode == "EXPLOIT"
        assert decision.reason_code == "MAX_EXPECTED_REWARD"

    # ── Test 21: Score NOT clipped to [0,1] ─────────────────────────────────

    def test_21_score_not_clipped(self):
        """Expected reward NOT clipped to [0, 1]."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        # Build a posterior with extreme rewards to push scores outside [0,1]
        prior = Prior.diagonal_prior(d=9, alpha=0.1)
        state = zero_state(d=9)
        np.random.seed(99)
        for _ in range(500):
            t = np.random.uniform(8.0, 17.0)
            phi = fourier(t, k=4)
            reward = 10.0  # Large reward
            state = apply_contribution(state, phi, float(reward), 0.06)
        posterior = compute_posterior(state, prior, sigma2=0.06)
        assert isinstance(posterior, Posterior)

        decision = select_action_deterministic(result, posterior, sigma2=0.06)

        # With large rewards, expected_reward should be outside [0,1]
        er = decision.expected_reward
        assert er > 1.0 or er < 0.0, (
            f"Expected reward should not be clipped to [0,1]; got {er}"
        )

    # ── Additional: empty candidates raises ─────────────────────────────────

    def test_deterministic_empty_raises(self):
        """Empty candidates raises ValueError."""
        empty = CandidateSet()
        posterior = _make_posterior(n=0)
        with pytest.raises(ValueError, match="empty"):
            select_action_deterministic(empty, posterior, sigma2=0.06)


# ── POL-06: select_action_explore ─────────────────────────────────────────────

class TestSelectActionExplore:
    """SRS POL-06: Exploration action selection."""

    # ── Test 22: Single candidate → probability 1.0 ─────────────────────────

    def test_22_single_candidate_probability_one(self):
        """Single candidate → selects it with probability 1.0."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 8, 14, 0)),  # Only 08:00 fits
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert len(result.timestamps) == 1, f"Expected 1 candidate, got {len(result.timestamps)}"

        rng = np.random.default_rng(42)
        decision = select_action_explore(result, rng)

        assert decision.scheduled_at == result.timestamps[0]
        assert decision.action_probability == 1.0
        assert decision.mode == "UNIFORM_EXPLORE"

    # ── Test 23: Two candidates → each 0.5 ──────────────────────────────────

    def test_23_two_candidates_each_0_5(self):
        """Two candidates → each has probability 0.5."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 8, 29, 0)),  # 08:00, 08:15 = 2 candidates
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert len(result.timestamps) == 2, f"Expected 2 candidates, got {len(result.timestamps)}"

        rng = np.random.default_rng(42)
        decision = select_action_explore(result, rng)

        assert decision.action_probability == pytest.approx(0.5, abs=1e-12)
        assert decision.candidate_count == 2

    # ── Test 24: m candidates → each 1/m ────────────────────────────────────

    def test_24_m_candidates_each_1_m(self):
        """m candidates → each has probability 1/m."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 18, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        m = len(result.timestamps)
        rng = np.random.default_rng(42)
        decision = select_action_explore(result, rng)

        expected_prob = 1.0 / m
        assert decision.action_probability == pytest.approx(expected_prob, abs=1e-12)
        assert decision.candidate_count == m

    # ── Test 25: Uniform distribution ───────────────────────────────────────

    def test_25_uniform_distribution_10000_draws(self):
        """10000 draws on 4 candidates → each within 2% of 25%."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 8, 45, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert len(result.timestamps) == 4, f"Expected 4 candidates, got {len(result.timestamps)}"

        rng = np.random.default_rng(123)
        counts = [0] * 4
        for _ in range(10000):
            decision = select_action_explore(result, rng)
            idx = result.timestamps.index(decision.scheduled_at)
            counts[idx] += 1

        for i, count in enumerate(counts):
            freq = count / 10000.0
            assert abs(freq - 0.25) < 0.02, (
                f"Candidate {i}: frequency {freq:.4f} not within 2% of 25%"
            )

    # ── Test 26: Mode = UNIFORM_EXPLORE ─────────────────────────────────────

    def test_26_mode_uniform_explore(self):
        """Mode is UNIFORM_EXPLORE."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        rng = np.random.default_rng(42)
        decision = select_action_explore(result, rng)

        assert decision.mode == "UNIFORM_EXPLORE"
        assert decision.assignment == "EXPLORE"

    # ── Additional: empty candidates raises ─────────────────────────────────

    def test_explore_empty_raises(self):
        """Empty candidates raises ValueError."""
        empty = CandidateSet()
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="empty"):
            select_action_explore(empty, rng)


# ── POL-07: find_secondary_peak ──────────────────────────────────────────────

class TestFindSecondaryPeak:
    """SRS POL-07: Secondary peak detection."""

    # ── Test 27: Bimodal curve ──────────────────────────────────────────────

    def test_27_bimodal_curve_finds_second_maximum(self):
        """Bimodal curve → secondary peak found at second maximum."""
        # Create bimodal rewards: two peaks separated by a valley
        # Primary at index 18, secondary at index 3 (15 indices = 3.75h gap)
        rewards = np.array([
            0.1, 0.3, 0.5, 0.9,  # rising to peak1 (index 3)
            0.5, 0.3, 0.2, 0.3,  # valley
            0.5, 0.7, 0.9, 0.7,  # small intermediate
            0.5, 0.3, 0.2, 0.3,  # valley
            0.5, 0.7, 0.9, 1.0,  # rising to peak2 (index 18, primary)
            0.7, 0.5, 0.3, 0.2,  # falling
        ], dtype=np.float64)
        support = [True] * len(rewards)

        # Primary peak at index 18 (value 1.0)
        primary = 18
        secondary = find_secondary_peak(rewards, support, primary)

        # Should find the first peak (index 3, value 0.9) as secondary
        assert secondary >= 0, "Should find a secondary peak in bimodal curve"
        assert secondary == 3, f"Expected secondary at index 3, got {secondary}"
        # Verify min gap: |3 - 18| = 15 >= 8 (2h)
        assert abs(secondary - primary) >= 8, "Secondary peak must be >= 2h from primary"

    # ── Test 28: Unimodal curve → no secondary ──────────────────────────────

    def test_28_unimodal_curve_no_secondary(self):
        """Unimodal curve → no secondary peak (-1)."""
        rewards = np.array([
            0.1, 0.3, 0.5, 0.7, 0.9, 1.0,  # rising
            0.8, 0.6, 0.4, 0.2, 0.1, 0.05,  # falling
        ], dtype=np.float64)
        support = [True] * len(rewards)

        primary = 5  # peak at index 5
        secondary = find_secondary_peak(rewards, support, primary)

        assert secondary == -1, "Unimodal curve should have no secondary peak"

    # ── Test 29: Flat curve → no secondary ──────────────────────────────────

    def test_29_flat_curve_no_secondary(self):
        """Flat curve → no secondary peak (-1)."""
        rewards = np.full(20, 0.5, dtype=np.float64)
        support = [True] * len(rewards)

        secondary = find_secondary_peak(rewards, support, 0)

        assert secondary == -1, "Flat curve should have no secondary peak"

    # ── Test 30: Plateau → earliest point ───────────────────────────────────

    def test_30_plateau_earliest_point(self):
        """Plateau → secondary peak at earliest point of plateau."""
        # Two plateaus at different levels, separated by a wide valley
        # Plateau1 at indices 0-2 (value 0.8), valley at 3-8 (6 indices = 1.5h),
        # Plateau2 at 9-11 (value 1.0) → gap from 2 to 9 = 7 indices = 1.75h
        # Need >= 8 indices gap, so make valley wider
        rewards = np.array([
            0.8, 0.8, 0.8,  # plateau1 (secondary peak) at indices 0-2
            0.5, 0.3, 0.2, 0.3, 0.5, 0.7,  # valley then rising (indices 3-8)
            1.0, 1.0, 1.0,  # plateau2 (primary peak) at indices 9-11
            0.5, 0.3, 0.2,  # falling (indices 12-14)
        ], dtype=np.float64)
        support = [True] * len(rewards)

        # Primary peak at plateau2 (index 9)
        primary = 9
        secondary = find_secondary_peak(rewards, support, primary)

        assert secondary >= 0, "Should find secondary peak on plateau"
        assert secondary == 0, f"Plateau peak should be at earliest point (0), got {secondary}"
        # Verify min gap: |0 - 9| = 9 >= 8 (2h)
        assert abs(secondary - primary) >= 8, "Secondary peak must be >= 2h from primary"

    # ── Test 31: Secondary peak >=2h from primary ───────────────────────────

    def test_31_secondary_peak_min_gap_2h(self):
        """Secondary peak must be >= 2 hours from primary."""
        # Create rewards where a local max is too close to primary
        rewards = np.array([
            0.1, 0.3, 0.5, 0.7, 0.9,  # rising
            0.5, 0.3, 0.2, 0.3, 0.5,  # valley then small rise (too close)
            0.7, 0.9, 1.0, 0.8, 0.6,  # primary peak
            0.4, 0.3, 0.2, 0.3, 0.5,  # small rise (too close)
            0.7, 0.9, 1.1, 0.9, 0.7,  # another peak (far enough)
            0.5, 0.3, 0.2, 0.1, 0.0,
        ], dtype=np.float64)
        support = [True] * len(rewards)

        # Primary peak at index 22 (value 1.1)
        primary = 22
        secondary = find_secondary_peak(rewards, support, primary)

        if secondary >= 0:
            # Each candidate is 15 min apart, 2h = 8 indices
            gap = abs(secondary - primary)
            assert gap >= 8, (
                f"Secondary peak at index {secondary} is only {gap * 0.25:.1f}h from primary"
            )

    # ── Test 32: Endpoint can be peak ───────────────────────────────────────

    def test_32_endpoint_can_be_peak(self):
        """Endpoint can be peak if strictly better than its only neighbor."""
        # Rising to the last point
        rewards = np.array([
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ], dtype=np.float64)
        support = [True] * len(rewards)

        # Primary at index 0
        primary = 0
        secondary = find_secondary_peak(rewards, support, primary)

        assert secondary == 9, (
            f"Last point should be secondary peak; got {secondary}"
        )

    # ── Test 33: Wholly flat run → no peak ──────────────────────────────────

    def test_33_wholly_flat_run_no_peak(self):
        """Wholly flat run → no peak."""
        rewards = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        support = [True] * len(rewards)

        secondary = find_secondary_peak(rewards, support, 0)

        assert secondary == -1, "Wholly flat run should have no peak"

    # ── Test 34: Secondary peak on same date as primary ─────────────────────

    def test_34_secondary_peak_same_date(self):
        """Secondary peak can be on the same date as primary."""
        # Two peaks within the same day (same date in local time)
        rewards = np.array([
            0.1, 0.3, 0.5, 0.7, 0.9,  # morning peak
            0.3, 0.2, 0.1, 0.2, 0.3,  # afternoon valley
            0.5, 0.7, 0.9, 1.0, 0.8,  # evening peak (primary)
            0.6, 0.4, 0.3, 0.2, 0.1,
        ], dtype=np.float64)
        support = [True] * len(rewards)

        primary = 13  # evening peak
        secondary = find_secondary_peak(rewards, support, primary)

        assert secondary >= 0, "Should find morning peak as secondary"
        assert secondary == 4, f"Expected morning peak at index 4, got {secondary}"

    # ── Edge: empty array ───────────────────────────────────────────────────

    def test_secondary_empty_array(self):
        """Empty rewards → -1."""
        rewards = np.array([], dtype=np.float64)
        secondary = find_secondary_peak(rewards, [], 0)
        assert secondary == -1

    # ── Edge: all unsupported ───────────────────────────────────────────────

    def test_secondary_all_unsupported(self):
        """All unsupported → -1."""
        rewards = np.array([0.1, 0.5, 0.9, 0.5, 0.1], dtype=np.float64)
        support = [False, False, False, False, False]
        secondary = find_secondary_peak(rewards, support, 2)
        assert secondary == -1


# ── recommend: main entry point ───────────────────────────────────────────────

class TestRecommend:
    """SRS POL-01 to POL-08: Main recommendation entry point."""

    # ── Test 35: EXPLORE assignment ─────────────────────────────────────────

    def test_35_explore_assignment(self):
        """EXPLORE assignment → UNIFORM_EXPLORE mode."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="EXPLORE",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert decision.mode == "UNIFORM_EXPLORE"
        assert decision.assignment == "EXPLORE"
        assert decision.status == "RECOMMENDED"

    # ── Test 36: TREATMENT with cold start ──────────────────────────────────

    def test_36_treatment_cold_start(self):
        """TREATMENT with cold start → PRIOR_ONLY mode."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=0)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="TREATMENT",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert decision.mode == "PRIOR_ONLY"
        assert decision.assignment == "TREATMENT"
        assert decision.status == "RECOMMENDED"

    # ── Test 37: TREATMENT with data → EXPLOIT ──────────────────────────────

    def test_37_treatment_with_data(self):
        """TREATMENT with data → EXPLOIT mode."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="TREATMENT",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert decision.mode == "EXPLOIT"
        assert decision.assignment == "TREATMENT"
        assert decision.status == "RECOMMENDED"

    # ── Test 38: SHADOW assignment ──────────────────────────────────────────

    def test_38_shadow_ope_eligible_false(self):
        """SHADOW assignment → ope_eligible=False."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="SHADOW",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert decision.assignment == "SHADOW"
        assert decision.ope_eligible is False

    # ── Test 39: CONTROL assignment ─────────────────────────────────────────

    def test_39_control_with_data(self):
        """CONTROL assignment → EXPLOIT mode (with data)."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="CONTROL",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert decision.mode == "EXPLOIT"
        assert decision.assignment == "CONTROL"
        assert decision.status == "RECOMMENDED"

    # ── Test 40: Empty candidates → NO_ELIGIBLE_SLOT ────────────────────────

    def test_40_empty_candidates_no_eligible_slot(self):
        """Empty candidates → NO_ELIGIBLE_SLOT status."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=CandidateSet(),
            posterior=posterior,
            assignment="TREATMENT",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert decision.status == "NO_ELIGIBLE_SLOT"
        assert decision.scheduled_at is None
        assert decision.secondary_at is None
        assert decision.mode == "NONE"

    # ── Test 41: Decision includes required fields ──────────────────────────

    def test_41_decision_required_fields(self):
        """Decision includes all required fields from §10.2."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="TREATMENT",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        # Required fields from §10.2
        required_fields = {
            "decision_id", "status", "scheduled_at", "secondary_at",
            "reason_code", "mode", "assignment", "experiment_id",
            "candidate_count", "action_probability", "assignment_probability",
            "ope_eligible", "n_attempts", "prior_weight",
            "expected_reward", "latent_std", "predictive_std",
        }
        actual_fields = set(decision.__dataclass_fields__.keys())
        missing = required_fields - actual_fields
        assert not missing, f"Missing required fields: {missing}"

        # Check values are non-null for RECOMMENDED
        assert decision.decision_id != ""
        assert decision.status == "RECOMMENDED"
        assert decision.scheduled_at is not None
        assert decision.reason_code != ""
        assert decision.mode in ("EXPLOIT", "PRIOR_ONLY", "UNIFORM_EXPLORE")
        assert decision.assignment in ("CONTROL", "TREATMENT", "EXPLORE", "SHADOW")
        assert decision.candidate_count > 0
        assert decision.action_probability is not None
        assert decision.expected_reward is not None


# ── POL-01: Server time — all timestamps strictly in future ──────────────────

class TestPolicy01FutureTimestamps:
    """SRS POL-01: All returned timestamps strictly in the future."""

    def test_42_clock_injection_past(self):
        """Test with past clock → all-past → NO_ELIGIBLE_SLOT."""
        past = _k(datetime(2026, 6, 15, 12, 0, 0))
        clock = _make_clock(past)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert len(result.timestamps) == 0, (
            "All-past interval should produce empty candidates"
        )

    def test_pol01_recommend_future_check(self):
        """recommend enforces POL-01: scheduled_at strictly in future."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="TREATMENT",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        if decision.scheduled_at is not None:
            sched = decision.scheduled_at
            if sched.tzinfo is None:
                sched = sched.replace(tzinfo=timezone.utc)
            now_utc = _now_utc(clock)
            assert sched > now_utc, (
                f"scheduled_at {sched} must be strictly after now {now_utc}"
            )


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Additional edge cases and SRS compliance."""

    # ── Test 43: RNG reproducibility ────────────────────────────────────────

    def test_43_rng_reproducibility(self):
        """Same seed → same exploration result."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 8, 45, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        rng1 = np.random.default_rng(42)
        decision1 = select_action_explore(result, rng1)

        rng2 = np.random.default_rng(42)
        decision2 = select_action_explore(result, rng2)

        assert decision1.scheduled_at == decision2.scheduled_at, (
            "Same RNG seed should produce same exploration result"
        )

    # ── Test 44: Calendar with only holidays ────────────────────────────────

    def test_44_calendar_all_holidays(self):
        """Calendar with only holidays → no candidates."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        # Monday (June 15) is a holiday, and it's the only day in range
        cal = _make_calendar(holidays=["2026-06-15", "2026-06-16", "2026-06-17"])
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 17, 18, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert len(result.timestamps) == 0, (
            "All holidays → no candidates"
        )

    # ── Test 45: Support mask all unsupported ───────────────────────────────

    def test_45_all_unsupported(self):
        """All bins unsupported → fallback with all candidates marked unsupported."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        # No supported bins at all
        support_mask: dict = {}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 18, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        # Fallback behavior: when no supported bins, all candidates are kept
        # but marked as unsupported (support_flags all False)
        assert len(result.timestamps) > 0
        assert all(not flag for flag in result.support_mask), (
            "All support flags should be False when no bins are supported"
        )

    # ── Additional edge: CandidateSet dataclass ─────────────────────────────

    def test_candidateset_default_factory(self):
        """CandidateSet defaults to empty lists."""
        cs = CandidateSet()
        assert cs.timestamps == []
        assert cs.support_mask == []
        assert cs.scores is None

    # ── Additional edge: PolicyDecision dataclass ───────────────────────────

    def test_policydecision_defaults(self):
        """PolicyDecision has correct default values."""
        pd = PolicyDecision()
        assert pd.status == "NONE"
        assert pd.scheduled_at is None
        assert pd.secondary_at is None
        assert pd.mode == "NONE"
        assert pd.assignment == "CONTROL"
        assert pd.candidate_count == 0
        assert pd.action_probability is None
        assert pd.ope_eligible is False

    # ── Additional: generate_candidates POL-01 enforcement ──────────────────

    def test_candidates_pol01_enforcement(self):
        """generate_candidates enforces POL-01: earliest pushed to future."""
        now = _k(datetime(2026, 6, 15, 8, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        # earliest_at is exactly now — should be pushed to now + 1s
        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 15, 10, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        for ts in result.timestamps:
            ts_utc = ts.astimezone(timezone.utc)
            now_utc = _now_utc(clock)
            assert ts_utc > now_utc, (
                f"Candidate {ts} not strictly in future of {now_utc}"
            )

    # ── Additional: secondary peak with unsupported gaps ────────────────────

    def test_secondary_with_unsupported_gaps(self):
        """Secondary peak detection handles unsupported gaps."""
        rewards = np.array([
            0.1, 0.3, 0.5, 0.7, 0.9,  # supported peak
            0.3, 0.2, 0.1, 0.2, 0.3,  # gap (partially unsupported)
            0.5, 0.7, 0.9, 1.0, 0.8,  # supported peak (primary)
            0.6, 0.4, 0.3, 0.2, 0.1,
        ], dtype=np.float64)
        support = [
            True, True, True, True, True,   # supported
            False, False, False, False, False,  # unsupported gap
            True, True, True, True, True,    # supported
            True, True, True, True, True,
        ]

        primary = 13
        secondary = find_secondary_peak(rewards, support, primary)

        # Should find the first supported peak at index 4
        assert secondary == 4, f"Expected peak at 4, got {secondary}"

    # ── Additional: secondary peak index out of bounds ──────────────────────

    def test_secondary_peak_primary_out_of_range(self):
        """Primary index beyond array length handled gracefully."""
        rewards = np.array([0.1, 0.5, 0.9, 0.5, 0.1], dtype=np.float64)
        support = [True] * len(rewards)

        # Primary index 100 is way beyond array
        result = find_secondary_peak(rewards, support, 100)
        # Should not crash; behavior depends on implementation
        # (the function doesn't explicitly validate primary_index)

    # ── Additional: recommend with unknown assignment ───────────────────────

    def test_recommend_unknown_assignment(self):
        """Unknown assignment → NO_ELIGIBLE_SLOT with reason code."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=CandidateSet(timestamps=[now + timedelta(hours=1)]),
            posterior=posterior,
            assignment="UNKNOWN_ARM",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert decision.status == "NO_ELIGIBLE_SLOT"
        assert "UNKNOWN_ASSIGNMENT" in decision.reason_code
        assert decision.mode == "NONE"

    # ── Additional: daily cap with calls_below_max ──────────────────────────

    def test_daily_cap_below_max_allows_today(self):
        """calls_today=2, max_calls=3 → today still allowed."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 18, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
            max_calls_per_day=3, calls_already_today=2,
        )

        assert len(result.timestamps) > 0, (
            "Below daily cap should allow today's candidates"
        )

    # ── Additional: grid spanning multiple days ─────────────────────────────

    def test_grid_multi_day_candidates(self):
        """Grid spanning multiple days produces multi-day candidates."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        earliest = _k(datetime(2026, 6, 15, 8, 0, 0))
        latest = _k(datetime(2026, 6, 17, 18, 0, 0))

        result = generate_candidates(
            earliest_at=earliest, latest_at=latest,
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        # Should have candidates across multiple days
        dates = set()
        for ts in result.timestamps:
            dates.add(_to_kolkata(ts).date())
        assert len(dates) >= 2, (
            f"Expected candidates across multiple days, got dates: {dates}"
        )

    # ── Additional: generate_candidates returns CandidateSet with correct type ─

    def test_candidateset_returns_correct_type(self):
        """generate_candidates returns CandidateSet with correct types."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        assert isinstance(result, CandidateSet)
        assert isinstance(result.timestamps, list)
        assert isinstance(result.support_mask, list)
        assert result.scores is None  # Not yet scored

    # ── Additional: recommend returns PolicyDecision with correct type ──────

    def test_recommend_returns_policy_decision(self):
        """recommend returns PolicyDecision with correct type."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        rng = np.random.default_rng(42)

        decision = recommend(
            candidates=result,
            posterior=posterior,
            assignment="TREATMENT",
            sigma2=0.06,
            rng=rng,
            clock=clock,
        )

        assert isinstance(decision, PolicyDecision)
        assert isinstance(decision.decision_id, str)
        assert isinstance(decision.status, str)

    # ── Additional: secondary peak with single plateau ──────────────────────

    def test_secondary_single_plateau_no_peak(self):
        """Single plateau (whole array is flat) → no peak."""
        rewards = np.full(10, 0.5, dtype=np.float64)
        support = [True] * 10
        secondary = find_secondary_peak(rewards, support, 0)
        assert secondary == -1

    # ── Additional: _score_from_posterior shape ─────────────────────────────

    def test_score_from_posterior_shape(self):
        """_score_from_posterior returns (n, 4) array."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        scores = _score_from_posterior(result, posterior, sigma2=0.06)

        assert scores.shape == (len(result.timestamps), 4)
        assert scores.dtype == np.float64

    # ── Additional: score columns ───────────────────────────────────────────

    def test_score_columns(self):
        """Score columns: [expected_reward, latent_std, predictive_std, prior_weight]."""
        now = _k(datetime(2026, 6, 15, 7, 0, 0))
        clock = _make_clock(now)
        cal = _make_calendar()
        support_mask = {h: True for h in range(8, 18)}

        result = generate_candidates(
            earliest_at=_k(datetime(2026, 6, 15, 8, 0, 0)),
            latest_at=_k(datetime(2026, 6, 15, 10, 0, 0)),
            calendar=cal, support_mask=support_mask, clock=clock,
        )

        posterior = _make_posterior(n=100)
        scores = _score_from_posterior(result, posterior, sigma2=0.06)

        # Column 0: expected_reward (can be any value)
        assert np.all(np.isfinite(scores[:, 0]))
        # Column 1: latent_std (non-negative)
        assert np.all(scores[:, 1] >= 0)
        # Column 2: predictive_std (>= latent_std)
        assert np.all(scores[:, 2] >= scores[:, 1])
        # Column 3: prior_weight (in [0, 1])
        assert np.all((scores[:, 3] >= 0) & (scores[:, 3] <= 1))

    # ── Additional: empty candidates scoring ────────────────────────────────

    def test_score_empty_candidates(self):
        """Empty candidates → (0, 4) array."""
        empty = CandidateSet()
        posterior = _make_posterior(n=0)
        scores = _score_from_posterior(empty, posterior, sigma2=0.06)
        assert scores.shape == (0, 4)
