"""CSV data adapters for the Best Time to Call prediction system.

Implements WP2 (Data/training agent) raw CSV ingestion:
- DATA-07: Field mappings verified before production use
- DATA-08: Import reports with counts, exclusions, unknowns
- DATA-01: IDs as strings (max 128 UTF-8 bytes)
- DATA-02: Timestamps ISO 8601 with offset
- DATA-03: Reject unknown fields, finite values only
- DATA-04: Validate outcome consistency

Reads raw CSV files from the data dictionary schema and maps to canonical
schema defined in ``src/btc/schemas.py``.

Usage
-----
>>> attempts, report = load_attempts_csv("data/Best-Time-to-Call - Call Attempts Apr-Sep 2026.csv")
>>> sellers = load_sellers_csv("data/Best-Time-to-Call - Sellers.csv")
>>> joined = join_attempts_sellers(attempts, sellers)
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disposition label mapping (DATA-07: verified field mappings)
# ---------------------------------------------------------------------------

_DISPOSITION_MAP: dict[str, str] = {
    "Not Answered": "NOT_ANSWERED",
    "NotAnswered": "NOT_ANSWERED",
    "Meeting Fixed": "MEETING_FIXED",
    "Meeting_Fixed": "MEETING_FIXED",
    "MEETING_FIXED": "MEETING_FIXED",
    "Not Interested": "NOT_INTERESTED",
    "Not_Interested": "NOT_INTERESTED",
    "NOT_INTERESTED": "NOT_INTERESTED",
    "General": "GENERAL",
    "GENERAL": "GENERAL",
    "General (talked)": "GENERAL",
    "General (talked) (call back)": "CALL_LATER_BUSY",
    "Not Interested (talked)": "NOT_INTERESTED",
    "Not Interested (talked) (call back)": "CALL_LATER_BUSY",
    "Meeting Fixed (talked)": "MEETING_FIXED",
    "Call Later": "CALL_LATER_BUSY",
    "Call_Later": "CALL_LATER_BUSY",
    "Busy": "CALL_LATER_BUSY",
    "Not Available": "CALL_LATER_BUSY",
    "CALL_LATER_BUSY": "CALL_LATER_BUSY",
    "Wrong Number": "UNKNOWN",
    "Invalid Number": "UNKNOWN",
    "Number Closed": "UNKNOWN",
    "Refused": "NOT_INTERESTED",
    "No Response": "NOT_ANSWERED",
    "Call Back": "CALL_LATER_BUSY",
    "Callback": "CALL_LATER_BUSY",
    "Later": "CALL_LATER_BUSY",
    "Later Call": "CALL_LATER_BUSY",
    "Call Tomorrow": "CALL_LATER_BUSY",
    "Call Next Week": "CALL_LATER_BUSY",
    "Call Next Month": "CALL_LATER_BUSY",
    "Call Next Year": "CALL_LATER_BUSY",
    "NOT_ANSWERED": "NOT_ANSWERED",
    "Call After 5": "CALL_LATER_BUSY",
    "Call After 6": "CALL_LATER_BUSY",
    "Call After 7": "CALL_LATER_BUSY",
    "Call After 8": "CALL_LATER_BUSY",
    "Call After 9": "CALL_LATER_BUSY",
    "Call After 10": "CALL_LATER_BUSY",
    "Call After 11": "CALL_LATER_BUSY",
    "Call After 12": "CALL_LATER_BUSY",
    "Call After 1": "CALL_LATER_BUSY",
    "Call After 2": "CALL_LATER_BUSY",
    "Call After 3": "CALL_LATER_BUSY",
    "Call After 4": "CALL_LATER_BUSY",
}

_CANONICAL_DISPOSITIONS = frozenset({
    "MEETING_FIXED", "NOT_INTERESTED", "GENERAL",
    "CALL_LATER_BUSY", "NOT_ANSWERED", "UNKNOWN",
})

# Known source disposition fields for DATA-07 verification.
_KNOWN_DISPOSITION_FIELDS = frozenset({
    "disposition_label", "lead_call_status", "meeting_fixed",
    "lead_call_duration", "lead_tbro_time",
})

# Timestamp patterns for parse_timestamp.
_SIMPLE_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")


# ---------------------------------------------------------------------------
# Disposition mapping
# ---------------------------------------------------------------------------

def map_disposition(raw_label: str) -> str:
    """Map source disposition labels to canonical values.

    Applies exact-match lookup first, then prefix-based pattern matching.
    Falls back to UNKNOWN for unrecognised labels (DATA-08 unknown_mappings).

    Parameters
    ----------
    raw_label : str
        Raw disposition from CSV.

    Returns
    -------
    str
        Canonical disposition: MEETING_FIXED, NOT_INTERESTED, GENERAL,
        CALL_LATER_BUSY, NOT_ANSWERED, UNKNOWN
    """
    if not isinstance(raw_label, str):
        return "UNKNOWN"

    stripped = raw_label.strip()
    if not stripped:
        return "UNKNOWN"

    # Exact match first.
    if stripped in _DISPOSITION_MAP:
        return _DISPOSITION_MAP[stripped]

    # Prefix-based matching: check if any known key is a prefix of the label.
    for key, canonical in _DISPOSITION_MAP.items():
        if stripped.startswith(key):
            return canonical

    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Answered mapping
# ---------------------------------------------------------------------------

def map_answered(raw_status: str) -> bool:
    """Map source call status to boolean answered.

    Parameters
    ----------
    raw_status : str
        Source status: 'Answered' or 'NotAnswered'

    Returns
    -------
    bool
        True if the call was answered, False otherwise.
    """
    if not isinstance(raw_status, str):
        return False

    stripped = raw_status.strip()
    if not stripped:
        return False

    if stripped == "Answered":
        return True
    if stripped == "NotAnswered":
        return False

    # Treat any other value as not answered (DATA-08 unknown_mappings).
    return False


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

def parse_timestamp(ts_str: str, source_tz: str = "Asia/Kolkata") -> datetime:
    """Parse a timestamp string to timezone-aware datetime.

    Handles naive timestamps by localizing with source_tz.
    Already-aware timestamps are converted to UTC.

    Parameters
    ----------
    ts_str : str
        Timestamp string (ISO format or 'YYYY-MM-DD HH:MM:SS').
    source_tz : str
        Source timezone for naive timestamps.

    Returns
    -------
    datetime
        Timezone-aware datetime in UTC.

    Raises
    ------
    ValueError
        If the timestamp is unparseable.
    """
    from zoneinfo import ZoneInfo

    if not isinstance(ts_str, str):
        raise ValueError(f"Timestamp must be a string, got {type(ts_str).__name__}")

    stripped = ts_str.strip()
    if not stripped:
        raise ValueError("Empty timestamp string")

    # Try parsing as ISO format first (handles Z suffix, +HH:MM offset).
    try:
        dt = datetime.fromisoformat(stripped)
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc)
        # Naive ISO timestamp -- localize with source_tz.
        tz = ZoneInfo(source_tz)
        dt = dt.replace(tzinfo=tz)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        pass

    # Try simple format: 'YYYY-MM-DD HH:MM:SS'.
    if _SIMPLE_TS_RE.match(stripped):
        tz = ZoneInfo(source_tz)
        dt = datetime.strptime(stripped, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=tz)
        return dt.astimezone(timezone.utc)

    raise ValueError(f"Unparseable timestamp: {ts_str!r}")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_id(value: Any) -> str:
    """Validate and stringify an ID field (DATA-01).

    Parameters
    ----------
    value : Any
        The value to validate.

    Returns
    -------
    str
        The validated string.

    Raises
    ------
    ValueError
        If the value is not a string or exceeds 128 UTF-8 bytes.
    """
    if value is None:
        raise ValueError("ID must be non-null (DATA-01)")
    s = str(value).strip()
    if not s:
        raise ValueError("ID must be nonempty (DATA-01)")
    encoded = s.encode("utf-8")
    if len(encoded) > 128:
        raise ValueError(f"ID exceeds 128 UTF-8 bytes ({len(encoded)}), DATA-01")
    return s


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float, returning default on failure.

    Parameters
    ----------
    value : Any
        The value to convert.
    default : float | None
        Default value on conversion failure.

    Returns
    -------
    float | None
        The converted float or default.
    """
    if value is None:
        return default
    try:
        f = float(value)
        if not math.isfinite(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely convert a value to int, returning default on failure.

    Parameters
    ----------
    value : Any
        The value to convert.
    default : int | None
        Default value on conversion failure.

    Returns
    -------
    int | None
        The converted int or default.
    """
    if value is None:
        return default
    try:
        f = float(value)
        if not math.isfinite(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default


def _build_import_report(
    input_rows: int,
    seen_ids: set,
    duplicates: list,
    exclusions_by_reason: dict,
    unknown_mappings: int,
    missing_profiles: set,
    retained_rows: int,
) -> dict:
    """Build the DATA-08 import report dictionary.

    Parameters
    ----------
    input_rows : int
        Total rows read from CSV.
    seen_ids : set
        Set of unique attempt IDs processed.
    duplicates : list
        List of duplicate attempt IDs removed.
    exclusions_by_reason : dict
        Count of exclusions by reason.
    unknown_mappings : int
        Count of disposition/status mapping failures.
    missing_profiles : set
        Set of seller_ids not found in sellers.
    retained_rows : int
        Rows after cleaning.

    Returns
    -------
    dict
        Import report dictionary.
    """
    return {
        "input_rows": input_rows,
        "unique_attempts": len(seen_ids),
        "duplicates": len(duplicates),
        "exclusions_by_reason": dict(exclusions_by_reason),
        "unknown_mappings": unknown_mappings,
        "missing_profiles": len(missing_profiles),
        "retained_rows": retained_rows,
    }


# ---------------------------------------------------------------------------
# load_attempts_csv
# ---------------------------------------------------------------------------

def load_attempts_csv(
    path: str,
    source_tz: str = "Asia/Kolkata",
    source: str = "indiamart",
) -> tuple[list[dict], dict]:
    """Load and normalize attempts CSV (DATA-08).

    Reads raw attempts CSV, maps fields to canonical schema, deduplicates
    by attempt_id, validates outcome consistency (DATA-04), and produces
    an import report with counts, exclusions, and unknowns.

    Parameters
    ----------
    path : str
        Path to attempts CSV.
    source_tz : str
        Source timezone for naive timestamps.
    source : str
        Source identifier for canonical payload.

    Returns
    -------
    tuple[list[dict], dict]
        (normalized_outcomes, import_report) where import_report contains:
        - input_rows: total rows read
        - unique_attempts: unique attempt_ids
        - duplicates: duplicate attempt_ids removed
        - exclusions_by_reason: dict of reason -> count
        - unknown_mappings: disposition/status mapping failures
        - missing_profiles: seller_ids not found in sellers
        - retained_rows: rows after cleaning
    """
    # Read CSV.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    input_rows = len(df)
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    exclusions_by_reason: dict[str, int] = {}
    unknown_mappings = 0
    missing_profiles: set[str] = set()
    normalized: list[dict] = []

    # DATA-07: Verify known fields exist in the CSV.
    known_source_fields = frozenset({
        "data_hotlead_disposition_dtlid", "fk_glusr_usr_id", "redis_bucket",
        "lead_bot_version", "call_attempt_count", "lead_sent_time",
        "call_start_time", "vendor_response_time", "lead_call_status",
        "lead_call_duration", "disposition_label", "meeting_fixed",
        "lead_tbro_time",
    })
    csv_columns = set(df.columns)
    unknown_csv_fields = csv_columns - known_source_fields
    if unknown_csv_fields:
        logger.warning("Unknown CSV fields detected (DATA-07): %s", unknown_csv_fields)

    for idx, row in df.iterrows():
        try:
            # -- Extract and validate attempt_id (DATA-01) --
            raw_attempt_id = row.get("data_hotlead_disposition_dtlid", "")
            attempt_id = _validate_id(raw_attempt_id)

            # -- Deduplicate by attempt_id --
            if attempt_id in seen_ids:
                duplicates.append(attempt_id)
                reason = "duplicate_attempt"
                exclusions_by_reason[reason] = exclusions_by_reason.get(reason, 0) + 1
                continue
            seen_ids.add(attempt_id)

            # -- Extract and validate seller_id (DATA-01) --
            raw_seller_id = row.get("fk_glusr_usr_id", "")
            seller_id = _validate_id(raw_seller_id)

            # -- Extract source_bucket (nullable diagnostic) --
            source_bucket = row.get("redis_bucket", "") or None
            if source_bucket:
                source_bucket = source_bucket.strip() or None

            # -- Extract attempt_number --
            attempt_number = _safe_int(row.get("call_attempt_count"), default=1)
            if attempt_number is None or attempt_number < 1:
                attempt_number = 1

            # -- Parse timestamps (DATA-02) --
            lead_sent_time = parse_timestamp(row.get("lead_sent_time", ""), source_tz)
            call_start_time = parse_timestamp(row.get("call_start_time", ""), source_tz)

            # finalized_at: when the outcome became available (vendor_response_time or call_start_time)
            finalized_at = parse_timestamp(row.get("vendor_response_time", ""), source_tz)
            if finalized_at is None:
                finalized_at = call_start_time

            # lead_tbro_time (nullable callback request).
            raw_callback = row.get("lead_tbro_time", "")
            requested_callback_at: Optional[datetime] = None
            if raw_callback and raw_callback.strip():
                try:
                    requested_callback_at = parse_timestamp(raw_callback, source_tz)
                except ValueError:
                    requested_callback_at = None

            # -- Map answered (DATA-04) --
            raw_status = row.get("lead_call_status", "")
            answered = map_answered(raw_status)

            # -- Map disposition (DATA-04) --
            raw_disposition = row.get("disposition_label", "")
            disposition = map_disposition(raw_disposition)
            if disposition == "UNKNOWN":
                unknown_mappings += 1

            # -- Parse duration (DATA-03: finite values only) --
            duration_s = _safe_int(row.get("lead_call_duration"), default=None)
            if duration_s is not None and duration_s < 0:
                duration_s = None

            # -- Parse meeting_fixed --
            raw_meeting = row.get("meeting_fixed", "0")
            if isinstance(raw_meeting, str):
                raw_meeting = raw_meeting.strip()
            meeting_fixed = bool(_safe_int(raw_meeting, default=0))

            # -- DATA-04: Validate outcome consistency --
            # meeting_fixed=True requires answered=True and disposition=MEETING_FIXED.
            if meeting_fixed and not (answered and disposition == "MEETING_FIXED"):
                reason = "contradictory_meeting_fixed"
                exclusions_by_reason[reason] = exclusions_by_reason.get(reason, 0) + 1
                continue

            # NOT_ANSWERED disposition requires answered=False.
            if disposition == "NOT_ANSWERED" and answered:
                reason = "contradictory_not_answered"
                exclusions_by_reason[reason] = exclusions_by_reason.get(reason, 0) + 1
                continue

            # -- Build canonical outcome dict --
            outcome = {
                "attempt_id": attempt_id,
                "seller_id": seller_id,
                "source": source,
                "source_bucket": source_bucket,
                "attempt_number": attempt_number,
                "lead_sent_time": lead_sent_time,
                "call_start_time": call_start_time,
                "finalized_at": finalized_at,
                "answered": answered,
                "disposition": disposition,
                "meeting_fixed": meeting_fixed,
                "requested_callback_at": requested_callback_at,
                "duration_s": duration_s,
            }

            normalized.append(outcome)

        except ValueError as exc:
            # Quarantine rows with unresolvable errors (DATA-08).
            reason = "parse_error"
            exclusions_by_reason[reason] = exclusions_by_reason.get(reason, 0) + 1
            logger.debug("Quarantined row %d: %s", idx, exc)
            continue
        except Exception as exc:
            # Catch-all quarantine.
            reason = "unexpected_error"
            exclusions_by_reason[reason] = exclusions_by_reason.get(reason, 0) + 1
            logger.debug("Quarantined row %d (unexpected): %s", idx, exc)
            continue

    # Build import report.
    import_report = _build_import_report(
        input_rows=input_rows,
        seen_ids=seen_ids,
        duplicates=duplicates,
        exclusions_by_reason=exclusions_by_reason,
        unknown_mappings=unknown_mappings,
        missing_profiles=missing_profiles,
        retained_rows=len(normalized),
    )

    return normalized, import_report


# ---------------------------------------------------------------------------
# load_sellers_csv
# ---------------------------------------------------------------------------

def load_sellers_csv(path: str) -> dict:
    """Load and normalize sellers CSV.

    Returns dict mapping seller_id (str) -> seller_profile dict.

    Parameters
    ----------
    path : str
        Path to sellers CSV.

    Returns
    -------
    dict
        seller_id -> profile mapping.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    sellers: dict[str, dict] = {}

    for idx, row in df.iterrows():
        try:
            raw_seller_id = row.get("fk_glusr_usr_id", "")
            seller_id = _validate_id(raw_seller_id)

            # Address fields (nullable).
            seller_city = row.get("seller_city", "") or None
            seller_district = row.get("seller_district", "") or None
            seller_state = row.get("seller_state", "") or None
            seller_pincode = row.get("seller_pincode", "") or None

            # Business attributes (nullable).
            business_type = row.get("business_type", "") or None
            turnover_band = row.get("annual_turnover", "") or None
            nature_of_business = row.get("nature_of_business", "") or None

            # Categories.
            top_cat_1 = row.get("top_category_1", "") or None
            top_cat_2 = row.get("top_category_2", "") or None
            top_cat_3 = row.get("top_category_3", "") or None

            top_parent_category = row.get("top_parent_category", "") or None
            top_category_group = row.get("top_category_group", "") or None
            num_categories = _safe_int(row.get("num_categories"), default=None)

            profile = {
                "seller_id": seller_id,
                "seller_city": seller_city,
                "seller_district": seller_district,
                "seller_state": seller_state,
                "seller_pincode": seller_pincode,
                "business_type": business_type,
                "turnover_band": turnover_band,
                "nature_of_business": nature_of_business,
                "top_category_1": top_cat_1,
                "top_category_2": top_cat_2,
                "top_category_3": top_cat_3,
                "top_parent_category": top_parent_category,
                "top_category_group": top_category_group,
                "num_categories": num_categories,
            }

            sellers[seller_id] = profile

        except ValueError as exc:
            logger.warning("Skipping seller row %d: %s", idx, exc)
            continue
        except Exception as exc:
            logger.warning("Skipping seller row %d (unexpected): %s", idx, exc)
            continue

    return sellers


# ---------------------------------------------------------------------------
# resolve_segment
# ---------------------------------------------------------------------------

def resolve_segment(seller_profile: dict) -> str:
    """Resolve seller to segment key for hierarchical pooling.

    Implements TRAIN-01: cell -> group_turnover -> group -> global.
    Missing dimensions stop at nearest valid parent.

    The segment key is a JSON array string representing the hierarchical path:
    - [cell_value, turnover_band] -> group_turnover segment
    - [group_value, turnover_band] -> group segment
    - [global] -> global segment

    Parameters
    ----------
    seller_profile : dict
        Seller profile from load_sellers_csv.

    Returns
    -------
    str
        Canonical segment key (JSON array string).
    """
    if not seller_profile:
        return json.dumps(["global"])

    # Extract category group for the hierarchical path.
    category_group = seller_profile.get("top_category_group")
    turnover_band = seller_profile.get("turnover_band")

    # Determine the segment hierarchy based on available dimensions.
    if category_group and turnover_band:
        # Full group_turnover segment.
        segment = [category_group, turnover_band]
    elif category_group:
        # Group-only segment.
        segment = [category_group]
    elif turnover_band:
        # Turnover-only segment.
        segment = ["global", turnover_band]
    else:
        # Global fallback.
        segment = ["global"]

    return json.dumps(segment)


# ---------------------------------------------------------------------------
# join_attempts_sellers
# ---------------------------------------------------------------------------

def join_attempts_sellers(
    attempts: list[dict],
    sellers: dict,
) -> list[dict]:
    """Join attempts with seller profiles.

    Attempts without matching seller get UNKNOWN segment.

    Parameters
    ----------
    attempts : list[dict]
        Normalized attempts.
    sellers : dict
        Seller profiles.

    Returns
    -------
    list[dict]
        Joined records with seller features.
    """
    joined: list[dict] = []

    for attempt in attempts:
        seller_id = attempt.get("seller_id", "")
        seller_profile = sellers.get(seller_id)

        record = dict(attempt)

        if seller_profile:
            # Attach seller features.
            record["seller_city"] = seller_profile.get("seller_city")
            record["seller_district"] = seller_profile.get("seller_district")
            record["seller_state"] = seller_profile.get("seller_state")
            record["seller_pincode"] = seller_profile.get("seller_pincode")
            record["business_type"] = seller_profile.get("business_type")
            record["turnover_band"] = seller_profile.get("turnover_band")
            record["nature_of_business"] = seller_profile.get("nature_of_business")
            record["top_category_1"] = seller_profile.get("top_category_1")
            record["top_category_2"] = seller_profile.get("top_category_2")
            record["top_category_3"] = seller_profile.get("top_category_3")
            record["top_parent_category"] = seller_profile.get("top_parent_category")
            record["top_category_group"] = seller_profile.get("top_category_group")
            record["num_categories"] = seller_profile.get("num_categories")

            # Resolve segment.
            record["segment"] = resolve_segment(seller_profile)
        else:
            # No matching seller -- UNKNOWN segment.
            record["segment"] = json.dumps(["UNKNOWN"])
            record["seller_city"] = None
            record["seller_district"] = None
            record["seller_state"] = None
            record["seller_pincode"] = None
            record["business_type"] = None
            record["turnover_band"] = None
            record["nature_of_business"] = None
            record["top_category_1"] = None
            record["top_category_2"] = None
            record["top_category_3"] = None
            record["top_parent_category"] = None
            record["top_category_group"] = None
            record["num_categories"] = None

        joined.append(record)

    return joined
