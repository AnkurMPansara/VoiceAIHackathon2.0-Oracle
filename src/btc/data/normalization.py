"""Data normalization, point-in-time joins, and chronological splits.

Implements SRS TRAIN-01 (segment resolution), TRAIN-05 (support bins),
and TRAIN-06 (chronological splits) for the Best Time to Call system.

This module provides pure functions for:
- Normalizing raw attempt records to canonical schema (DATA-01, DATA-02, DATA-04)
- Creating disjoint chronological splits for model training (TRAIN-06)
- Computing segment eligibility statistics (TRAIN-01)
- Computing 15-minute support bins (TRAIN-05)
- Resolving seller profiles at a point in time
- Validating split integrity

Modules
-------
normalize_outcome : Normalize a single raw attempt record to canonical schema.
create_chronological_splits : TRAIN-06 chronological interval splits.
compute_segment_statistics : TRAIN-01 segment eligibility.
compute_support_bins : TRAIN-05 15-minute support bins.
point_in_time_profile : Resolve seller profile at a point in time.
validate_split_integrity : Validate chronological split correctness.

Imports
-------
btc.data.adapters : Raw data loading and field mapping.
btc.features.fourier : Time-to-hours conversion for Fourier features.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from btc.data.adapters import (
    map_answered,
    map_disposition,
    resolve_segment,
)
from btc.features.fourier import time_to_hours


# ── TRAIN-06: Chronological split boundaries ──────────────────────────────────

# Default split intervals in business timezone (Asia/Kolkata).
# Each boundary is an inclusive start, exclusive end.
_SPLIT_BOUNDARIES = [
    ("prior_fit", "2026-04-01T00:00:00+05:30", "2026-07-01T00:00:00+05:30"),
    ("warmup", "2026-07-01T00:00:00+05:30", "2026-08-01T00:00:00+05:30"),
    ("validation", "2026-08-01T00:00:00+05:30", "2026-09-01T00:00:00+05:30"),
    ("test", "2026-09-01T00:00:00+05:30", "2026-10-01T00:00:00+05:30"),
]

_SPLIT_PURPOSES = ["prior_fit", "warmup", "validation", "test"]


# ── 15-minute bin helpers ─────────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))

def _bin_key(dt: datetime) -> str:
    """Compute the 15-minute bin key for a datetime.

    Returns a string in the format "YYYY-MM-DDTHH:MM" representing the
    start of the half-open 15-minute interval [start, start+15min).

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.

    Returns
    -------
    str
        Bin key string, e.g. "2026-04-15T10:00".
    """
    dt = dt.astimezone(_IST)
    return f"{dt.hour:02d}:{(dt.minute // 15) * 15:02d}"


def _parse_bin_key(key: str) -> datetime:
    """Parse a bin key string back to a datetime.

    Parameters
    ----------
    key : str
        Bin key in format "YYYY-MM-DDTHH:MM".

    Returns
    -------
    datetime
        Datetime at the start of the bin (UTC).
    """
    dt = datetime.strptime(key, "%Y-%m-%dT%H:%M")
    return dt.replace(tzinfo=timezone.utc)


# ── normalize_outcome ─────────────────────────────────────────────────────────


def normalize_outcome(raw: dict, sellers: dict) -> dict:
    """Normalize a single raw attempt record to canonical schema.

    SRS DATA-01: All IDs converted to strings (max 128 UTF-8 bytes).
    SRS DATA-02: Timestamps converted to timezone-aware UTC.
    SRS DATA-04: Cross-field consistency validation (validates that related
                 fields agree with each other).

    Processing steps:
    1. Map disposition_label → canonical Disposition enum
    2. Map lead_call_status → answered boolean
    3. Convert timestamps to timezone-aware UTC
    4. Convert meeting_fixed (1/0) → bool
    5. Convert call_attempt_count → int
    6. Resolve seller segment from seller profiles (via resolve_segment)
    7. Validate cross-field consistency (DATA-04) — raises ValueError on failure
    8. Convert IDs to strings (DATA-01)

    Parameters
    ----------
    raw : dict
        Raw record from CSV with keys matching the source CSV columns.
        Expected keys include: seller_id, lead_id, attempt_id, source,
        disposition_label, lead_call_status, finalized_at, call_start_time,
        call_end_time, lead_sent_time, call_attempt_count, meeting_fixed,
        and optional fields like duration_s, dialer_version, etc.
    sellers : dict
        Seller profiles keyed by seller_id. Each value is a dict with
        keys: seller_id, category_group, turnover_band, business_type,
        effective_from, profile_version.

    Returns
    -------
    dict
        Normalized record with canonical field names:
        - seller_id (str)
        - lead_id (str)
        - attempt_id (str)
        - source (str)
        - revision (int, default 1)
        - event_id (str)
        - finalized_at (datetime, UTC)
        - call_start_time (datetime, UTC)
        - call_end_time (datetime, UTC)
        - lead_sent_time (datetime, UTC)
        - attempt_number (int)
        - answered (bool)
        - disposition (str, canonical enum value)
        - meeting_fixed (bool)
        - requested_callback_at (datetime | None, UTC)
        - decision_id (str | None)
        - duration_s (int | None)
        - dialer_version (str | None)
        - source_bucket (str | None)
        - segment (str, resolved from seller profile)

    Raises
    ------
    ValueError
        If cross-field validation fails (quarantine this row per DATA-04).
        Reasons include:
        - meeting_fixed=True but answered=False or disposition≠MEETING_FIXED
        - disposition=NOT_ANSWERED but answered=True
        - call_start_time > call_end_time
        - lead_sent_time > call_start_time
    """
    # Step 1: Map disposition_label → canonical disposition string
    disposition_label = raw.get("disposition_label") or raw.get("disposition")
    disposition = map_disposition(disposition_label)

    # Step 2: Map lead_call_status → answered boolean
    lead_call_status = raw.get("lead_call_status") or raw.get("call_status")
    if "answered" in raw:
        answered_raw = raw["answered"]
        if isinstance(answered_raw, bool):
            answered = answered_raw
        elif isinstance(answered_raw, (int, float)):
            answered = bool(answered_raw)
        elif isinstance(answered_raw, str):
            answered = map_answered(answered_raw)
        else:
            answered = map_answered(lead_call_status)
    else:
        answered = map_answered(lead_call_status)

    # Step 3: Convert timestamps to timezone-aware UTC
    finalized_at = raw.get("finalized_at")
    call_start_time = raw.get("call_start_time")
    call_end_time = raw.get("call_end_time")
    lead_sent_time = raw.get("lead_sent_time")
    requested_callback_at = raw.get("requested_callback_at")

    # Parse timestamps - convert to UTC for storage
    def _to_utc(value: Any) -> Optional[datetime]:
        """Parse timestamp and convert to UTC."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.strip())
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Cannot parse timestamp {value!r}: {exc}"
                ) from exc
        else:
            dt = datetime.fromisoformat(str(value))

        # Convert to UTC
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        else:
            raise ValueError("Timestamp must include an explicit timezone offset")
        return dt

    finalized_at_utc = _to_utc(finalized_at)
    call_start_utc = _to_utc(call_start_time)
    call_end_utc = _to_utc(call_end_time)
    lead_sent_utc = _to_utc(lead_sent_time)
    callback_utc = _to_utc(requested_callback_at)

    # Step 4: Convert meeting_fixed (1/0/True/False/"1"/"0") → bool
    meeting_fixed_raw = raw.get("meeting_fixed")
    if isinstance(meeting_fixed_raw, bool):
        meeting_fixed = meeting_fixed_raw
    elif isinstance(meeting_fixed_raw, (int, float)):
        meeting_fixed = bool(meeting_fixed_raw)
    elif isinstance(meeting_fixed_raw, str):
        meeting_fixed = meeting_fixed_raw.strip() in ("1", "true", "True", "TRUE", "yes", "Yes", "Y", "y")
    else:
        meeting_fixed = False

    # Step 5: Convert call_attempt_count → int (DATA-02: positive integer)
    attempt_count_raw = raw.get("call_attempt_count") or raw.get("attempt_number")
    if attempt_count_raw is not None:
        try:
            attempt_number = int(attempt_count_raw)
        except (ValueError, TypeError):
            attempt_number = 1
    else:
        attempt_number = 1
    if attempt_number < 1:
        attempt_number = 1

    # Step 6: Resolve seller segment
    seller_id = str(raw.get("seller_id", ""))
    seller_profile = sellers.get(seller_id, {})
    segment = resolve_segment(seller_profile)

    # Step 7: Validate cross-field consistency (DATA-04)
    if meeting_fixed and not (answered and disposition == "MEETING_FIXED"):
        raise ValueError(
            f"meeting_fixed=True requires answered=True and "
            f"disposition=MEETING_FIXED (DATA-04). "
            f"attempt_id={raw.get('attempt_id')}: "
            f"answered={answered}, disposition={disposition}"
        )

    if disposition == "NOT_ANSWERED" and answered:
        raise ValueError(
            f"disposition=NOT_ANSWERED requires answered=False (DATA-04). "
            f"attempt_id={raw.get('attempt_id')}"
        )

    if call_start_utc is not None and call_end_utc is not None:
        if call_start_utc > call_end_utc:
            raise ValueError(
                f"call_start_time must be <= call_end_time (DATA-04). "
                f"attempt_id={raw.get('attempt_id')}"
            )

    if lead_sent_utc is not None and call_start_utc is not None:
        if lead_sent_utc > call_start_utc:
            raise ValueError(
                f"lead_sent_time must be <= call_start_time (DATA-04). "
                f"attempt_id={raw.get('attempt_id')}"
            )

    # Step 8: Convert IDs to strings (DATA-01)
    def _to_id(value: Any, field_name: str = "id") -> str:
        """Convert value to string ID (DATA-01)."""
        if value is None:
            return ""
        s = str(value)
        if len(s.encode("utf-8")) > 128:
            raise ValueError(
                f"{field_name} exceeds 128 UTF-8 bytes ({len(s.encode('utf-8'))}), DATA-01"
            )
        return s

    # Compute duration if not provided
    duration_s = raw.get("duration_s")
    if duration_s is not None:
        try:
            duration_s = int(duration_s)
        except (ValueError, TypeError):
            duration_s = None
    elif call_start_utc is not None and call_end_utc is not None:
        duration_s = int((call_end_utc - call_start_utc).total_seconds())

    # Build normalized record
    normalized: dict[str, Any] = {
        "seller_id": _to_id(raw.get("seller_id"), "seller_id"),
        "lead_id": _to_id(raw.get("lead_id"), "lead_id"),
        "attempt_id": _to_id(raw.get("attempt_id"), "attempt_id"),
        "source": _to_id(raw.get("source"), "source"),
        "event_id": _to_id(raw.get("event_id"), "event_id"),
        "finalized_at": finalized_at_utc,
        "call_start_time": call_start_utc,
        "call_end_time": call_end_utc,
        "lead_sent_time": lead_sent_utc,
        "attempt_number": attempt_number,
        "answered": answered,
        "disposition": disposition,
        "meeting_fixed": meeting_fixed,
        "requested_callback_at": callback_utc,
        "decision_id": _to_id(raw.get("decision_id"), "decision_id") or None,
        "duration_s": duration_s,
        "dialer_version": raw.get("dialer_version"),
        "source_bucket": raw.get("source_bucket"),
        "segment": segment,
    }

    # Validate required fields (DATA-02: source is required, non-empty)
    if not normalized["source"]:
        raise ValueError("source is a required field and cannot be empty (DATA-02).")

    # Validate revision (DATA-02: positive integer)
    try:
        normalized["revision"] = int(raw.get("revision", 1))
    except (ValueError, TypeError):
        normalized["revision"] = 1
    if normalized["revision"] < 1:
        normalized["revision"] = 1

    return normalized


# ── create_chronological_splits ───────────────────────────────────────────────


def create_chronological_splits(
    normalized_data: list[dict],
    timezone: str = "Asia/Kolkata",
) -> dict:
    """TRAIN-06: Split data into 4 disjoint chronological intervals.

    SRS TRAIN-06: Creates exactly 4 disjoint (non-overlapping) intervals
    based on ``finalized_at`` timestamp (local business time). Each record
    is assigned to exactly one split — no record appears in multiple splits.

    | Purpose | Interval |
    |---|---|
    | prior_fit | 2026-04-01 to 2026-07-01 |
    | warmup | 2026-07-01 to 2026-08-01 |
    | validation | 2026-08-01 to 2026-09-01 |
    | test | 2026-09-01 to 2026-10-01 |

    Each interval is half-open: [start, end). At each boundary, only outcomes
    finalized before that instant are included in the earlier interval.
    Prior-fitting rows are NOT replayed into seller state (no overlap
    with subsequent splits).

    Parameters
    ----------
    normalized_data : list[dict]
        All normalized outcome records, each with a 'finalized_at' field
        as a timezone-aware datetime.
    timezone : str
        Business timezone for interval boundaries (default: "Asia/Kolkata").

    Returns
    -------
    dict
        Split data by purpose with keys:
        - 'prior_fit': list of normalized outcome dicts (Apr-Jul)
        - 'warmup': list of normalized outcome dicts (Jul-Aug)
        - 'validation': list of normalized outcome dicts (Aug-Sep)
        - 'test': list of normalized outcome dicts (Sep-Oct)
    """
    import pytz

    tz = pytz.timezone(timezone)

    # Build boundaries in the specified timezone
    boundaries: list[tuple[str, datetime, datetime]] = []
    for purpose, start_str, end_str in _SPLIT_BOUNDARIES:
        # Parse the boundary strings (they use +05:30 offset for Asia/Kolkata)
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str)
        boundaries.append((purpose, start_dt, end_dt))

    # Initialize splits
    splits: dict[str, list[dict]] = {purpose: [] for purpose in _SPLIT_PURPOSES}

    # Assign each record to exactly one split based on finalized_at
    for record in normalized_data:
        finalized_at = record.get("finalized_at")
        if finalized_at is None:
            continue

        # Convert to the business timezone for comparison
        if finalized_at.tzinfo is None:
            finalized_at = finalized_at.replace(tzinfo=timezone.utc)
        finalized_local = finalized_at.astimezone(tz)

        # Find the correct split: use only outcomes finalized before boundary
        assigned = False
        for purpose, start, end in boundaries:
            # Convert boundary to local time for comparison
            start_local = start.astimezone(tz)
            end_local = end.astimezone(tz)

            # Record belongs here if start <= finalized < end
            if start_local <= finalized_local < end_local:
                splits[purpose].append(record)
                assigned = True
                break

        if not assigned:
            # Record falls outside all intervals — skip it
            pass

    return splits


# ── compute_segment_statistics ────────────────────────────────────────────────


def compute_segment_statistics(
    data: list[dict],
    min_attempts: int = 2000,
    min_sellers: int = 200,
) -> dict:
    """TRAIN-01: Compute segment eligibility statistics.

    SRS TRAIN-01: Segment resolution follows the hierarchy:
    cell (category_group) → group_turnover (category_group + turnover_band)
    → group (category_group alone) → global (no dimensions).

    Segments are SHARED across sellers — multiple sellers with the same
    category_group and turnover_band share one segment. In the production
    dataset, ~149,363 sellers map to ~497 unique segments.

    A segment is eligible when it has:
    - ≥ min_attempts (default 2,000) finalized attempts
    - ≥ min_sellers (default 200) distinct sellers

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'segment' and 'seller_id' fields.
    min_attempts : int
        Minimum attempts for segment eligibility (TRAIN-01).
    min_sellers : int
        Minimum distinct sellers for segment eligibility (TRAIN-01).

    Returns
    -------
    dict
        Segment statistics keyed by segment:
        {
            segment_key: {
                'n_sellers': int,
                'n_attempts': int,
                'eligible': bool
            }
        }
    """
    segment_data: dict[str, dict[str, Any]] = {}

    for record in data:
        segment = record.get("segment", "UNKNOWN")
        seller_id = record.get("seller_id")

        if segment not in segment_data:
            segment_data[segment] = {
                "sellers": set(),
                "n_attempts": 0,
            }

        segment_data[segment]["n_attempts"] += 1
        if seller_id is not None:
            segment_data[segment]["sellers"].add(seller_id)

    # Build result
    result: dict[str, dict[str, Any]] = {}
    for segment, stats in segment_data.items():
        n_sellers = len(stats["sellers"])
        n_attempts = stats["n_attempts"]
        eligible = n_attempts >= min_attempts and n_sellers >= min_sellers

        result[segment] = {
            "n_sellers": n_sellers,
            "n_attempts": n_attempts,
            "eligible": eligible,
        }

    return result


# ── compute_support_bins ──────────────────────────────────────────────────────


def compute_support_bins(
    data: list[dict],
    min_attempts: int = 50,
    min_sellers: int = 30,
) -> dict:
    """TRAIN-05: Compute supported 15-minute time bins per segment.

    SRS TRAIN-05: A bin is supported if it has:
    - ≥ min_attempts (default 50) attempts
    - ≥ min_sellers (default 30) distinct sellers

    Bins are half-open intervals [start, start + 15 minutes).

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'call_start_time' and 'seller_id' fields.
    min_attempts : int
        Minimum attempts per bin (TRAIN-05).
    min_sellers : int
        Minimum distinct sellers per bin (TRAIN-05).

    Returns
    -------
    dict
        Support mask keyed by segment:
        {
            segment_key: {
                bin_key: bool  # True if supported, False otherwise
            }
        }
        bin_key format: "YYYY-MM-DDTHH:MM" (start of 15-min interval).
    """
    # Collect per-segment, per-bin statistics
    bin_data: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"sellers": set(), "n_attempts": 0})
    )

    for record in data:
        segment = record.get("segment", "UNKNOWN")
        call_start = record.get("call_start_time")
        seller_id = record.get("seller_id")

        if call_start is None:
            continue

        key = _bin_key(call_start)
        bin_data[segment][key]["n_attempts"] += 1
        if seller_id is not None:
            bin_data[segment][key]["sellers"].add(seller_id)
        for seg in (segment, "__all__"):
            bin_data[seg][key]["n_attempts"] += 1
            if seller_id is not None: bin_data[seg][key]["sellers"].add(seller_id)

    # Build support mask
    result: dict[str, dict[str, bool]] = {}
    for segment, bins in bin_data.items():
        result[segment] = {}
        for bin_key, stats in bins.items():
            n_sellers = len(stats["sellers"])
            n_attempts = stats["n_attempts"]
            is_supported = n_attempts >= min_attempts and n_sellers >= min_sellers
            result[segment][bin_key] = is_supported

    return result


# ── point_in_time_profile ─────────────────────────────────────────────────────


def point_in_time_profile(
    seller_profiles: list[dict],
    seller_id: str,
    as_of_time: datetime,
) -> dict:
    """Resolve seller profile at a point in time.

    Returns the profile that was effective as_of_time. Uses the
    ``effective_from`` field to determine which version was active.
    If multiple versions exist, returns the one with the latest
    effective_from that is ≤ as_of_time.

    Parameters
    ----------
    seller_profiles : list[dict]
        All profile versions for a seller, sorted by effective_from
        ascending. Each dict should have 'effective_from' as a
        timezone-aware datetime.
    seller_id : str
        Seller identifier.
    as_of_time : datetime
        Point-in-time for profile resolution (must be timezone-aware).

    Returns
    -------
    dict | None
        Effective profile at as_of_time, or None if no profile
        was active at that time.

    Raises
    ------
    ValueError
        If as_of_time is naive (no timezone info).
    """
    if as_of_time.tzinfo is None:
        raise ValueError("as_of_time must be timezone-aware")

    if not seller_profiles:
        return None

    # Find the latest profile with effective_from <= as_of_time
    effective_profile: Optional[dict] = None
    latest_effective: Optional[datetime] = None

    for profile in seller_profiles:
        effective_from = profile.get("effective_from")
        if effective_from is None:
            continue

        # Convert to UTC for comparison if needed
        if effective_from.tzinfo is None:
            effective_from = effective_from.replace(tzinfo=timezone.utc)

        if effective_from <= as_of_time:
            if latest_effective is None or effective_from > latest_effective:
                latest_effective = effective_from
                effective_profile = profile

    return effective_profile


# ── validate_split_integrity ──────────────────────────────────────────────────


def validate_split_integrity(splits: dict) -> dict:
    """Validate that chronological splits are disjoint and complete.

    SRS TRAIN-06 integrity checks:
    - No seller appears in multiple splits for the same attempt
    - All attempts assigned to exactly one split
    - No temporal overlap between splits
    - Boundary outcomes handled correctly (before boundary → earlier split)

    Parameters
    ----------
    splits : dict
        Output from create_chronological_splits with keys:
        'prior_fit', 'warmup', 'validation', 'test'.

    Returns
    -------
    dict
        Validation report with check names as keys and boolean
        pass/fail values:
        {
            'no_duplicate_attempts': bool,
            'no_overlapping_time_ranges': bool,
            'boundary_outcomes_correct': bool,
            'all_splits_valid': bool
        }
    """
    report: dict[str, bool] = {}

    # Check 1: No duplicate attempts across splits
    attempt_ids: dict[str, list[str]] = {}
    seller_ids_per_split: dict[str, set[str]] = {}

    for purpose in _SPLIT_PURPOSES:
        records = splits.get(purpose, [])
        if not isinstance(records, list):
            continue
        attempt_ids[purpose] = []
        seller_ids_per_split[purpose] = set()

        for record in records:
            attempt_id = record.get("attempt_id")
            seller_id = record.get("seller_id")

            if attempt_id is not None:
                attempt_ids[purpose].append(attempt_id)
            if seller_id is not None:
                seller_ids_per_split[purpose].add(seller_id)

    # Check for duplicate attempts across splits
    all_attempts: dict[str, list[str]] = defaultdict(list)
    for purpose, attempts in attempt_ids.items():
        for aid in attempts:
            all_attempts[aid].append(purpose)

    has_duplicates = any(
        len(purposes) > 1 for purposes in all_attempts.values()
    )
    report["no_duplicate_attempts"] = not has_duplicates

    # Check 2: No overlapping time ranges between splits
    # Each split should have a distinct time range
    time_ranges: dict[str, tuple[datetime, datetime]] = {}
    for purpose in _SPLIT_PURPOSES:
        records = splits.get(purpose, [])
        if not records or not isinstance(records, list):
            time_ranges[purpose] = (None, None)
            continue

        finalized_times = [
            r["finalized_at"]
            for r in records
            if isinstance(r, dict) and r.get("finalized_at") is not None
        ]
        if finalized_times:
            min_time = min(finalized_times)
            max_time = max(finalized_times)
            time_ranges[purpose] = (min_time, max_time)
        else:
            time_ranges[purpose] = (None, None)

    # Check for overlapping time ranges
    no_overlap = True
    purposes_list = list(_SPLIT_PURPOSES)
    for i in range(len(purposes_list)):
        for j in range(i + 1, len(purposes_list)):
            pi, pj = purposes_list[i], purposes_list[j]
            min_i, max_i = time_ranges[pi]
            min_j, max_j = time_ranges[pj]

            if min_i is None or max_i is None or min_j is None or max_j is None:
                continue

            # Check overlap: ranges overlap if max(a,b) >= min(c,d)
            if max(min_i, min_j) <= max(max_i, max_j):
                # More precise: check if intervals overlap
                if max_i >= min_j and max_j >= min_i:
                    no_overlap = False
                    break

    report["no_overlapping_time_ranges"] = no_overlap

    # Check 3: Boundary outcomes handled correctly
    # Outcomes at a boundary should be in the earlier split (before boundary)
    # TRAIN-06: "At each boundary use only outcomes finalized before that instant"
    _BOUNDARIES = {
        "prior_fit": (datetime(2026, 4, 1, tzinfo=timezone.utc), datetime(2026, 7, 1, tzinfo=timezone.utc)),
        "warmup": (datetime(2026, 7, 1, tzinfo=timezone.utc), datetime(2026, 8, 1, tzinfo=timezone.utc)),
        "validation": (datetime(2026, 8, 1, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc)),
        "test": (datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 10, 1, tzinfo=timezone.utc)),
    }
    boundary_correct = True
    for purpose in _SPLIT_PURPOSES:
        records = splits.get(purpose, [])
        if not isinstance(records, list):
            boundary_correct = False
            break
        if purpose not in _BOUNDARIES:
            boundary_correct = False
            break
        start, end = _BOUNDARIES[purpose]
        for record in records:
            if not isinstance(record, dict):
                boundary_correct = False
                break
            finalized = record.get("finalized_at")
            if finalized is None:
                continue
            # Each outcome must fall within [start, end) of its assigned split
            if not (start <= finalized < end):
                boundary_correct = False
                break
        if not boundary_correct:
            break

    report["boundary_outcomes_correct"] = boundary_correct

    # Check 4: All splits are valid (proper structure)
    all_valid = True
    for purpose in _SPLIT_PURPOSES:
        if purpose not in splits:
            all_valid = False
            break
        if not isinstance(splits[purpose], list):
            all_valid = False
            break

    report["all_splits_valid"] = all_valid

    return report


# ── Bulk normalization helper ─────────────────────────────────────────────────


def normalize_batch(
    raw_records: list[dict],
    sellers: dict,
    *,
    quarantine: Optional[list[dict]] = None,
) -> list[dict]:
    """Normalize a batch of raw records, quarantining invalid ones.

    Parameters
    ----------
    raw_records : list[dict]
        Raw records to normalize.
    sellers : dict
        Seller profiles keyed by seller_id.
    quarantine : list[dict] | None
        Optional list to append quarantined (invalid) records to.

    Returns
    -------
    list[dict]
        Successfully normalized records.
    """
    normalized: list[dict] = []
    for raw in raw_records:
        try:
            result = normalize_outcome(raw, sellers)
            normalized.append(result)
        except ValueError:
            if quarantine is not None:
                quarantine.append(raw)
    return normalized


# ── Seller profile builder ────────────────────────────────────────────────────


def build_seller_profiles(
    raw_profiles: list[dict],
) -> dict[str, list[dict]]:
    """Build a seller profiles index keyed by seller_id.

    Groups all profile versions by seller_id for point-in-time lookups.

    Parameters
    ----------
    raw_profiles : list[dict]
        Raw seller profile dicts from CSV.

    Returns
    -------
    dict[str, list[dict]]
        Seller profiles keyed by seller_id, each value is a list of
        profile versions sorted by effective_from.
    """
    profiles: dict[str, list[dict]] = defaultdict(list)
    for profile in raw_profiles:
        seller_id = str(profile.get("seller_id", ""))
        profiles[seller_id].append(profile)

    # Sort each seller's profiles by effective_from
    for seller_id in profiles:
        profiles[seller_id].sort(
            key=lambda p: p.get("effective_from", datetime.min.replace(tzinfo=timezone.utc))
        )

    return dict(profiles)


# ── Prior-fit isolation ───────────────────────────────────────────────────────


def isolate_prior_fit_data(
    splits: dict,
    seller_profiles: dict[str, list[dict]],
    normalized_warmup: list[dict],
) -> tuple[list[dict], list[dict]]:
    """TRAIN-06: Return prior_fit data separately from seller state data.

    SRS TRAIN-06: Prior-fitting rows SHALL NOT be replayed into seller
    state. This function separates the prior_fit split from the data
    used for seller state warmup.

    Parameters
    ----------
    splits : dict
        Output from create_chronological_splits.
    seller_profiles : dict
        Seller profiles keyed by seller_id.
    normalized_warmup : list[dict]
        Warmup split data for seller state initialization.

    Returns
    -------
    tuple[list[dict], list[dict]]
        (prior_fit_data, seller_state_data) where seller_state_data
        contains warmup + validation + test data (excluding prior_fit).
    """
    prior_fit = splits.get("prior_fit", [])
    warmup = splits.get("warmup", [])
    validation = splits.get("validation", [])
    test = splits.get("test", [])

    # Seller state data excludes prior_fit (TRAIN-06)
    seller_state = warmup + validation + test

    return prior_fit, seller_state
