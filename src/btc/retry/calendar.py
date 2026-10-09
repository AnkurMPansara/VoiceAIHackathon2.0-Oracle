"""Business calendar and supported-hours computation for retry scheduling.

Implements SRS POL-02, POL-03, and TRAIN-05:

- POL-02: Calendar windows are start-inclusive, end-exclusive.
  Development: 08:00-18:00 Asia/Kolkata, Mon-Sat, no holidays.
- POL-03: next_working_day walks the calendar forward (not +24h)
  to find the next eligible day.
- TRAIN-05: Supported 15-minute bins are computed per segment with
  minimum thresholds (>=50 attempts, >=30 sellers).

All calendar functions are PURE — no mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from btc.config import Calendar as AppConfigCalendar, ModelConfig


# ── Timezone constant ──────────────────────────────────────────────────────────

_KOLKATTA = timezone(timedelta(hours=5, minutes=30))
"""Asia/Kolkata UTC offset used throughout the module."""


# ── BusinessCalendar ───────────────────────────────────────────────────────────


@dataclass
class BusinessCalendar:
    """Business calendar configuration.

    Attributes
    ----------
    days_of_week : list[int]
        Allowed weekdays as ISO numbers (0=Monday … 6=Sunday).
    start_hour : int
        Inclusive start of the calling window (0–23).
    end_hour : int
        Exclusive end of the calling window (0–23).
    holidays : list[str]
        ISO date strings (YYYY-MM-DD) excluded from the calendar.
    timezone : str
        IANA timezone name (default ``"Asia/Kolkata"``).
    """

    days_of_week: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    start_hour: int = 8
    end_hour: int = 18
    holidays: list[str] = field(default_factory=list)
    timezone: str = "Asia/Kolkata"

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_calendar: AppConfigCalendar) -> BusinessCalendar:
        """Create a BusinessCalendar from a ``btc.config.Calendar``.

        Parameters
        ----------
        config_calendar : AppConfigCalendar
            Calendar configuration loaded from YAML.

        Returns
        -------
        BusinessCalendar
            Fully populated business calendar.
        """
        return cls(
            days_of_week=list(config_calendar.days_of_week),
            start_hour=config_calendar.start_hour,
            end_hour=config_calendar.end_hour,
            holidays=list(config_calendar.holidays),
            timezone="Asia/Kolkata",
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


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
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_KOLKATTA)


def _date_str(dt: datetime) -> str:
    """Return an ISO date string ``YYYY-MM-DD`` for a datetime.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.

    Returns
    -------
    str
        Formatted date string.
    """
    local = _to_kolkata(dt)
    return local.strftime("%Y-%m-%d")


# ── Public API ─────────────────────────────────────────────────────────────────


def is_working_day(dt: datetime, calendar: BusinessCalendar) -> bool:
    """Check if datetime falls on a working day.

    POL-02: A working day is one where the ISO weekday is in
    ``days_of_week`` and the date is not in ``holidays``.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.
    calendar : BusinessCalendar
        Business calendar configuration.

    Returns
    -------
    bool
        True if the datetime falls on a working day.
    """
    local = _to_kolkata(dt)
    iso_weekday = local.weekday()  # Monday=0 … Sunday=6

    if iso_weekday not in calendar.days_of_week:
        return False

    date_str = local.strftime("%Y-%m-%d")
    if date_str in calendar.holidays:
        return False

    return True


def is_within_call_window(dt: datetime, calendar: BusinessCalendar) -> bool:
    """Check if datetime is within 08:00-18:00 call window.

    POL-02: start_hour is inclusive, end_hour is exclusive.
    For the default calendar, 18:00 is NOT permitted.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.
    calendar : BusinessCalendar
        Business calendar configuration.

    Returns
    -------
    bool
        True if the time falls within ``[start_hour, end_hour)``.
    """
    local = _to_kolkata(dt)
    time_minutes = local.hour * 60 + local.minute + local.second / 60.0
    start_minutes = calendar.start_hour * 60
    end_minutes = calendar.end_hour * 60

    return start_minutes <= time_minutes < end_minutes


def next_working_day(dt: datetime, calendar: BusinessCalendar) -> datetime:
    """Calculate next working day through calendar, not by adding 24h.

    POL-03: Walks day-by-day from the day AFTER *dt*'s local date
    until a working day is found. Returns the datetime at
    ``start_hour:00:00`` of that working day, in the calendar timezone.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime to start from.
    calendar : BusinessCalendar
        Business calendar configuration.

    Returns
    -------
    datetime
        The local date of the next working day at
        ``start_hour:00:00``, timezone-aware in Asia/Kolkata.
    """
    local = _to_kolkata(dt)
    day = local.date() + timedelta(days=1)
    while True:
        if _is_working_day_date(day, calendar):
            return datetime(
                year=day.year,
                month=day.month,
                day=day.day,
                hour=calendar.start_hour,
                minute=0,
                second=0,
                tzinfo=_KOLKATTA,
            )
        day += timedelta(days=1)


def _is_working_day_date(date_obj, calendar: BusinessCalendar) -> bool:
    """Check if a ``date`` object falls on a working day.

    Parameters
    ----------
    date_obj : datetime.date
        Calendar date to check.
    calendar : BusinessCalendar
        Business calendar configuration.

    Returns
    -------
    bool
        True if the date is a working day.
    """
    if date_obj.weekday() not in calendar.days_of_week:
        return False
    date_str = date_obj.isoformat()  # YYYY-MM-DD
    if date_str in calendar.holidays:
        return False
    return True


def find_next_available_slot(
    after: datetime,
    calendar: BusinessCalendar,
    support_mask: dict,
    min_gap_minutes: int = 15,
) -> datetime:
    """Find next available slot after *after* respecting calendar and support.

    Walks forward in 15-minute increments from *after*, checking:
    1. The candidate is on a working day.
    2. The candidate is within the call window.
    3. The candidate is at least *min_gap_minutes* after *after*.
    4. The candidate's hour bin is supported (in support_mask).

    If no slot is found within 7 days, falls back to the next
    working day at start_hour.

    Parameters
    ----------
    after : datetime
        Earliest acceptable time (exclusive).
    calendar : BusinessCalendar
        Business calendar configuration.
    support_mask : dict
        Mapping of hour_bin (int) to is_supported (bool).
    min_gap_minutes : int
        Minimum gap in minutes from *after* (default 15).

    Returns
    -------
    datetime
        The earliest available slot, timezone-aware in Asia/Kolkata.
    """
    from btc.model.policy import _is_in_support_bin

    local_after = _to_kolkata(after)
    gap_end = local_after + timedelta(minutes=min_gap_minutes)

    candidate = gap_end.replace(second=0, microsecond=0)
    # Align to next 15-minute boundary
    minute = candidate.minute
    if minute % 15 != 0:
        next_minute = (minute // 15 + 1) * 15
        if next_minute >= 60:
            candidate = (candidate + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0,
            )
        else:
            candidate = candidate.replace(minute=next_minute, second=0, microsecond=0)

    max_date = _to_kolkata(after).date() + timedelta(days=7)

    while candidate.date() <= max_date:
        if not _is_working_day_date(candidate.date(), calendar):
            # Skip to next working day
            next_day = candidate.date() + timedelta(days=1)
            candidate = datetime(
                year=next_day.year,
                month=next_day.month,
                day=next_day.day,
                hour=calendar.start_hour,
                minute=0,
                second=0,
                tzinfo=_KOLKATTA,
            )
            continue

        if not is_within_call_window(candidate, calendar):
            # Skip to next working day start
            next_day = candidate.date() + timedelta(days=1)
            candidate = datetime(
                year=next_day.year,
                month=next_day.month,
                day=next_day.day,
                hour=calendar.start_hour,
                minute=0,
                second=0,
                tzinfo=_KOLKATTA,
            )
            continue

        if _is_in_support_bin(candidate, support_mask):
            return candidate

        # Move to next 15-minute slot
        candidate += timedelta(minutes=15)

    # Fallback: next working day at start_hour
    return next_working_day(after, calendar)


def compute_support_mask(
    segment_data: list[dict],
    segment_key: str,
    min_attempts: int = 50,
    min_sellers: int = 30,
) -> dict:
    """TRAIN-05: Compute supported 15-min bins per segment.

    Aggregates *segment_data* by hour_bin (0-23) and counts
    total attempts and distinct sellers per bin. A bin is
    supported if it has >= *min_attempts* and >= *min_sellers*.

    Parameters
    ----------
    segment_data : list[dict]
        List of attempt records, each containing at least:
        - ``segment_key``: the segment identifier.
        - ``call_end_time``: datetime of call end.
        - ``seller_id``: seller identifier.
    segment_key : str
        Key to filter segment data (e.g. ``"category_group"``).
    min_attempts : int
        Minimum attempts in a 15-minute bin (default 50).
    min_sellers : int
        Minimum distinct sellers in a 15-minute bin (default 30).

    Returns
    -------
    dict
        Mapping of ``hour_bin`` (int) to ``is_supported`` (bool).
        Only bins meeting both thresholds are included as ``True``.
    """
    # Aggregate per hour_bin
    bin_attempts: dict[int, int] = {}
    bin_sellers: dict[int, set[str]] = {}

    for record in segment_data:
        if record.get("segment_key") != segment_key:
            continue

        call_end = record.get("call_end_time")
        if call_end is None:
            continue

        seller_id = record.get("seller_id")
        if seller_id is None:
            continue

        # Get hour bin from call_end_time
        if isinstance(call_end, str):
            call_end = datetime.fromisoformat(call_end)
        local = _to_kolkata(call_end)
        hour_bin = local.hour

        bin_attempts[hour_bin] = bin_attempts.get(hour_bin, 0) + 1
        if hour_bin not in bin_sellers:
            bin_sellers[hour_bin] = set()
        bin_sellers[hour_bin].add(seller_id)

    # Build support mask
    support: dict[int, bool] = {}
    for hour_bin in sorted(bin_attempts.keys()):
        attempts = bin_attempts[hour_bin]
        sellers = len(bin_sellers.get(hour_bin, set()))
        support[hour_bin] = attempts >= min_attempts and sellers >= min_sellers

    return support
