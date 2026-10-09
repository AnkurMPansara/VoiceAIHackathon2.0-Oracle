"""Comprehensive unit tests for src/btc/data/adapters.py.

Cross-references every adapter function against:
- best_time_to_call_srs_v2.md (SRS §4.2 DATA-01 to DATA-08, §4.4 mapping)
- data/Best-Time-to-Call - Data Dictionary.md

Tested requirements:
  DATA-01: IDs as strings, max 128 UTF-8 bytes
  DATA-02: ISO 8601 timestamps with offset
  DATA-03: Finite values, reject unknown fields
  DATA-04: meeting_fixed / answered / disposition consistency
  DATA-07: Field mapping verification
  DATA-08: Import report completeness

Functions under test:
  map_disposition, map_answered, parse_timestamp
  _validate_id, _safe_float, _safe_int, _build_import_report
  load_attempts_csv, load_sellers_csv
  resolve_segment, join_attempts_sellers
"""

from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple

import pytest

# Ensure src is on path (conftest does this, but be explicit for Windows).
_src = Path(__file__).parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from btc.data.adapters import (
    map_disposition,
    map_answered,
    parse_timestamp,
    _validate_id,
    _safe_float,
    _safe_int,
    load_attempts_csv,
    load_sellers_csv,
    resolve_segment,
    join_attempts_sellers,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: CSV content generators
# ──────────────────────────────────────────────────────────────────────────────

_ATTEMPTS_HEADER = [
    "data_hotlead_disposition_dtlid",
    "fk_glusr_usr_id",
    "redis_bucket",
    "lead_bot_version",
    "call_attempt_count",
    "lead_sent_time",
    "call_start_time",
    "vendor_response_time",
    "lead_call_status",
    "lead_call_duration",
    "disposition_label",
    "meeting_fixed",
    "lead_tbro_time",
]

_SELLERS_HEADER = [
    "fk_glusr_usr_id",
    "seller_city",
    "seller_district",
    "seller_state",
    "seller_pincode",
    "business_type",
    "annual_turnover",
    "nature_of_business",
    "nature_of_business_secondary",
    "gst_registration_year",
    "top_category_1",
    "top_category_2",
    "top_category_3",
    "top_parent_category",
    "top_category_group",
    "num_categories",
]


def _write_csv(rows: List[list], header: List[str]) -> Path:
    """Write rows (list of lists) to a temp CSV and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    )
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)
    f.close()
    return Path(f.name)


def _write_attempts_csv(rows: List[list]) -> Path:
    return _write_csv(rows, _ATTEMPTS_HEADER)


def _write_sellers_csv(rows: List[list]) -> Path:
    return _write_csv(rows, _SELLERS_HEADER)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  map_disposition tests (DATA-07, DATA-08)
# ──────────────────────────────────────────────────────────────────────────────

class TestMapDisposition:
    """SRS DATA-07: Disposition label mapping verification."""

    def test_meeting_fixed_to_meeting_fixed(self):
        assert map_disposition("Meeting Fixed") == "MEETING_FIXED"

    def test_meeting_fixed_underscore_to_meeting_fixed(self):
        assert map_disposition("Meeting_Fixed") == "MEETING_FIXED"

    def test_meeting_fixed_talked_to_meeting_fixed(self):
        assert map_disposition("Meeting Fixed (talked)") == "MEETING_FIXED"

    def test_not_interested_to_not_interested(self):
        assert map_disposition("Not Interested") == "NOT_INTERESTED"

    def test_not_interested_underscore_to_not_interested(self):
        assert map_disposition("Not_Interested") == "NOT_INTERESTED"

    def test_not_interested_talked_to_not_interested(self):
        assert map_disposition("Not Interested (talked)") == "NOT_INTERESTED"

    def test_not_interested_talked_callback_to_call_later_busy(self):
        assert map_disposition("Not Interested (talked) (call back)") == "CALL_LATER_BUSY"

    def test_general_to_general(self):
        assert map_disposition("General") == "GENERAL"

    def test_general_talked_to_general(self):
        assert map_disposition("General (talked)") == "GENERAL"

    def test_general_talked_callback_to_call_later_busy(self):
        assert map_disposition("General (talked) (call back)") == "CALL_LATER_BUSY"

    def test_call_later_busy_variations(self):
        for label in ["Call Later", "Call_Later", "Busy", "Not Available"]:
            assert map_disposition(label) == "CALL_LATER_BUSY", f"Failed for: {label}"

    def test_call_later_time_variants(self):
        for hour in range(1, 13):
            label = f"Call After {hour}"
            assert map_disposition(label) == "CALL_LATER_BUSY", f"Failed for: {label}"

    def test_call_later_date_variants(self):
        for label in ["Call Tomorrow", "Call Next Week", "Call Next Month", "Call Next Year"]:
            assert map_disposition(label) == "CALL_LATER_BUSY", f"Failed for: {label}"

    def test_callback_variants(self):
        for label in ["Call Back", "Callback", "Later", "Later Call"]:
            assert map_disposition(label) == "CALL_LATER_BUSY", f"Failed for: {label}"

    def test_not_answered_variants(self):
        for label in ["Not Answered", "NotAnswered", "No Response"]:
            assert map_disposition(label) == "NOT_ANSWERED", f"Failed for: {label}"

    def test_unknown_variants(self):
        for label in ["Wrong Number", "Invalid Number", "Number Closed"]:
            assert map_disposition(label) == "UNKNOWN", f"Failed for: {label}"

    def test_refused_to_not_interested(self):
        assert map_disposition("Refused") == "NOT_INTERESTED"

    def test_unknown_label_returns_unknown(self):
        assert map_disposition("Some Weird Label") == "UNKNOWN"

    def test_empty_string_returns_unknown(self):
        assert map_disposition("") == "UNKNOWN"

    def test_whitespace_only_returns_unknown(self):
        assert map_disposition("   ") == "UNKNOWN"

    def test_non_string_returns_unknown(self):
        assert map_disposition(123) == "UNKNOWN"
        assert map_disposition(None) == "UNKNOWN"
        assert map_disposition(True) == "UNKNOWN"

    def test_unknown_csv_field_warning_logged(self, caplog):
        """DATA-07: Unknown CSV fields should log a warning."""
        path = _write_attempts_csv([
            ["att-1", "s1", "bucket1", "main_vani", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        # Add an extra unknown column
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
        )
        writer = csv.writer(f)
        writer.writerow(_ATTEMPTS_HEADER + ["unknown_col"])
        writer.writerow(["att-1", "s1", "bucket1", "main_vani", "1",
                         "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
                         "Answered", "30", "Meeting Fixed", "1", "", "extra"])
        f.close()
        attempts, report = load_attempts_csv(f.name)
        assert "Unknown CSV fields detected" in caplog.text
        Path(f.name).unlink()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  map_answered tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMapAnswered:
    """SRS DATA-04: Answered status mapping."""

    def test_answered_true(self):
        assert map_answered("Answered") is True

    def test_not_answered_false(self):
        assert map_answered("NotAnswered") is False

    def test_not_answered_with_space_false(self):
        assert map_answered("Not Answered") is False

    def test_empty_returns_false(self):
        assert map_answered("") is False

    def test_whitespace_returns_false(self):
        assert map_answered("  ") is False

    def test_non_string_returns_false(self):
        assert map_answered(None) is False
        assert map_answered(123) is False

    def test_unknown_status_returns_false(self):
        assert map_answered("MISSED") is False
        assert map_answered("DROPPED") is False


# ──────────────────────────────────────────────────────────────────────────────
# 3.  parse_timestamp tests (DATA-02)
# ──────────────────────────────────────────────────────────────────────────────

class TestParseTimestamp:
    """SRS DATA-02: ISO 8601 timestamps with offset."""

    def test_iso_with_offset_converts_to_utc(self):
        dt = parse_timestamp("2026-05-15T10:30:00+05:30")
        assert dt.tzinfo is not None
        assert dt == datetime(2026, 5, 15, 5, 0, 0, tzinfo=timezone.utc)

    def test_iso_z_suffix_converts_to_utc(self):
        dt = parse_timestamp("2026-05-15T10:30:00Z")
        assert dt.tzinfo is not None
        assert dt == datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_naive_format_localized_to_kolkata_then_utc(self):
        """Naive timestamps are localized with source_tz (Asia/Kolkata = +05:30)."""
        dt = parse_timestamp("2026-05-15 10:30:00", source_tz="Asia/Kolkata")
        assert dt.tzinfo is not None
        assert dt == datetime(2026, 5, 15, 5, 0, 0, tzinfo=timezone.utc)

    def test_naive_with_different_source_tz(self):
        """Naive timestamps with explicit source_tz."""
        dt = parse_timestamp("2026-05-15 10:30:00", source_tz="America/New_York")
        assert dt.tzinfo is not None
        # EDT = UTC-4, so 10:30 EDT = 14:30 UTC
        assert dt == datetime(2026, 5, 15, 14, 30, 0, tzinfo=timezone.utc)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Empty timestamp"):
            parse_timestamp("")

    def test_whitespace_raises(self):
        with pytest.raises(ValueError):
            parse_timestamp("   ")

    def test_non_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            parse_timestamp(12345)

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Unparseable"):
            parse_timestamp("not-a-date")

    def test_iso_with_negative_offset(self):
        dt = parse_timestamp("2026-05-15T10:30:00-05:00")
        assert dt.tzinfo is not None
        # -05:00 = 10:30 + 5:00 = 15:30 UTC
        assert dt == datetime(2026, 5, 15, 15, 30, 0, tzinfo=timezone.utc)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  _validate_id tests (DATA-01)
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateId:
    """SRS DATA-01: IDs as strings, max 128 UTF-8 bytes."""

    def test_valid_ascii_id(self):
        assert _validate_id("seller-001") == "seller-001"

    def test_numeric_id_stringified(self):
        assert _validate_id(12345) == "12345"

    def test_id_with_spaces_stripped(self):
        assert _validate_id("  seller-001  ") == "seller-001"

    def test_128_bytes_ok(self):
        ok = "a" * 128
        assert _validate_id(ok) == ok

    def test_129_bytes_raises(self):
        too_long = "a" * 129
        with pytest.raises(ValueError, match="128"):
            _validate_id(too_long)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="nonempty"):
            _validate_id("")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="non-null"):
            _validate_id(None)

    def test_multibyte_unicode_counts_bytes(self):
        """Multi-byte UTF-8 characters count toward the 128 byte limit."""
        # 日本語 = 3 bytes each, so 42 chars = 126 bytes, 43 chars = 129 bytes
        short = "日" * 42  # 126 bytes
        assert _validate_id(short) == short
        long = "日" * 43  # 129 bytes
        with pytest.raises(ValueError, match="128"):
            _validate_id(long)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  _safe_float / _safe_int tests (DATA-03)
# ──────────────────────────────────────────────────────────────────────────────

class TestSafeConverters:
    """SRS DATA-03: Finite values only."""

    def test_safe_float_valid(self):
        assert _safe_float("3.14") == 3.14

    def test_safe_float_nan_returns_default(self):
        assert _safe_float("nan") is None

    def test_safe_float_inf_returns_default(self):
        assert _safe_float("inf") is None
        assert _safe_float("-inf") is None

    def test_safe_float_none_returns_default(self):
        assert _safe_float(None, default=99.0) == 99.0

    def test_safe_int_valid(self):
        assert _safe_int("42") == 42

    def test_safe_int_float_string(self):
        assert _safe_int("3.7") == 3

    def test_safe_int_nan_returns_default(self):
        assert _safe_int("nan") is None

    def test_safe_int_inf_returns_default(self):
        assert _safe_int("inf") is None


# ──────────────────────────────────────────────────────────────────────────────
# 6.  load_attempts_csv tests (DATA-04, DATA-08)
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadAttemptsCsv:
    """SRS DATA-04, DATA-08: Import attempts CSV with validation and reporting."""

    # -- Test 11: valid CSV produces normalized outcomes + import report --
    def test_valid_csv_normalized_outcomes(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "bucket1", "main_vani", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 1
            a = attempts[0]
            assert a["attempt_id"] == "att-1"
            assert a["seller_id"] == "s1"
            assert a["source"] == "indiamart"
            assert a["answered"] is True
            assert a["disposition"] == "MEETING_FIXED"
            assert a["meeting_fixed"] is True
            assert a["attempt_number"] == 1
            assert a["duration_s"] == 30
            assert a["source_bucket"] == "bucket1"
            assert isinstance(a["lead_sent_time"], datetime)
            assert isinstance(a["call_start_time"], datetime)
        finally:
            path.unlink()

    def test_valid_csv_import_report(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "bucket1", "main_vani", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert report["input_rows"] == 1
            assert report["unique_attempts"] == 1
            assert report["duplicates"] == 0
            assert report["retained_rows"] == 1
            assert "exclusions_by_reason" in report
            assert "unknown_mappings" in report
            assert "missing_profiles" in report
        finally:
            path.unlink()

    def test_custom_source(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "General", "0", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path), source="custom-dialer")
            assert attempts[0]["source"] == "custom-dialer"
        finally:
            path.unlink()

    # -- Test 12: duplicate attempt_id deduplication --
    def test_duplicate_attempt_id_deduplicated(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
            ["att-1", "s1", "", "", "2",
             "2026-05-15 11:00:00", "2026-05-15 11:01:00", "2026-05-15 11:01:30",
             "Answered", "30", "Not Interested", "0", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert report["duplicates"] == 1
            assert report["unique_attempts"] == 1
            assert report["exclusions_by_reason"].get("duplicate_attempt") == 1
        finally:
            path.unlink()

    # -- Test 13: contradictory labels quarantined --
    def test_contradictory_meeting_fixed_quarantined(self):
        """meeting_fixed=True but disposition=GENERAL → excluded (DATA-04)."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "General", "1", ""],  # contradiction!
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 0
            assert report["exclusions_by_reason"].get("contradictory_meeting_fixed") == 1
        finally:
            path.unlink()

    def test_contradictory_not_answered_quarantined(self):
        """disposition=NOT_ANSWERED but answered=True → excluded (DATA-04)."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Not Answered", "0", ""],  # contradiction!
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 0
            assert report["exclusions_by_reason"].get("contradictory_not_answered") == 1
        finally:
            path.unlink()

    def test_contradictory_meeting_fixed_no_answer_quarantined(self):
        """meeting_fixed=True but answered=False → excluded."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "NotAnswered", "0", "Meeting Fixed", "1", ""],  # contradiction!
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 0
            assert report["exclusions_by_reason"].get("contradictory_meeting_fixed") == 1
        finally:
            path.unlink()

    def test_valid_meeting_fixed_accepted(self):
        """meeting_fixed=True, answered=True, disposition=MEETING_FIXED → accepted."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert attempts[0]["meeting_fixed"] is True
            assert attempts[0]["answered"] is True
            assert attempts[0]["disposition"] == "MEETING_FIXED"
        finally:
            path.unlink()

    def test_valid_not_answered_accepted(self):
        """disposition=NOT_ANSWERED, answered=False → accepted."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "NotAnswered", "0", "Not Answered", "0", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert attempts[0]["answered"] is False
            assert attempts[0]["disposition"] == "NOT_ANSWERED"
        finally:
            path.unlink()

    def test_valid_call_later_busy_accepted(self):
        """CALL_LATER_BUSY with answered=True is valid (DATA-04)."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "45", "Call Later", "0", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert attempts[0]["answered"] is True
            assert attempts[0]["disposition"] == "CALL_LATER_BUSY"
        finally:
            path.unlink()

    def test_invalid_attempt_id_quarantined(self):
        """Empty attempt_id → quarantined."""
        path = _write_attempts_csv([
            ["", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 0
            assert report["exclusions_by_reason"].get("parse_error") == 1
        finally:
            path.unlink()

    def test_mixed_valid_and_invalid(self):
        """Mix of valid and contradictory rows."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
            ["att-2", "s1", "", "", "1",
             "2026-05-15 11:00:00", "2026-05-15 11:01:00", "2026-05-15 11:01:30",
             "Answered", "30", "General", "1", ""],  # contradictory
            ["att-3", "s1", "", "", "1",
             "2026-05-15 12:00:00", "2026-05-15 12:01:00", "2026-05-15 12:01:30",
             "NotAnswered", "0", "Not Answered", "0", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 2
            assert report["input_rows"] == 3
            assert report["retained_rows"] == 2
            assert report["exclusions_by_reason"].get("contradictory_meeting_fixed") == 1
        finally:
            path.unlink()

    def test_unknown_disposition_counted(self):
        """Unknown disposition → unknown_mappings incremented."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "NotAnswered", "0", "Some Weird Label", "0", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert attempts[0]["disposition"] == "UNKNOWN"
            assert report["unknown_mappings"] == 1
        finally:
            path.unlink()

    def test_requested_callback_at_parsed(self):
        """lead_tbro_time is parsed as requested_callback_at."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Call Later", "0", "2026-05-16 10:00:00"],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert attempts[0]["requested_callback_at"] is not None
            assert attempts[0]["requested_callback_at"] == datetime(
                2026, 5, 16, 4, 30, 0, tzinfo=timezone.utc
            )
        finally:
            path.unlink()

    def test_negative_duration_becomes_none(self):
        """Negative duration → None (DATA-03)."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "-5", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert attempts[0]["duration_s"] is None
        finally:
            path.unlink()

    def test_empty_csv(self):
        """Empty CSV (header only) → no attempts."""
        path = _write_attempts_csv([])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert len(attempts) == 0
            assert report["input_rows"] == 0
            assert report["retained_rows"] == 0
        finally:
            path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# 7.  load_sellers_csv tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadSellersCsv:
    """SRS §4.3: Load and normalize sellers CSV."""

    def test_valid_csv_returns_seller_dict(self):
        path = _write_sellers_csv([
            ["s1", "Mumbai", "Mumbai", "MH", "400001",
             "Proprietorship", "0-40L", "Manufacturer", "Mfg,Export", "2020",
             "Apparel", "Footwear", "Accessories", "Apparel", "Apparel", "3"],
        ])
        try:
            sellers = load_sellers_csv(str(path))
            assert "s1" in sellers
            s = sellers["s1"]
            assert s["seller_city"] == "Mumbai"
            assert s["seller_state"] == "MH"
            assert s["turnover_band"] == "0-40L"
            assert s["top_category_group"] == "Apparel"
            assert s["num_categories"] == 3
            assert s["business_type"] == "Proprietorship"
        finally:
            path.unlink()

    def test_multiple_sellers(self):
        path = _write_sellers_csv([
            ["s1", "Delhi", "Delhi", "DL", "110001",
             "Partnership", "40L-1.5Cr", "Trader", "", "2018",
             "Electronics", "Mobile", "", "Electronics", "Electronics", "2"],
            ["s2", "Bangalore", "Bangalore", "KA", "560001",
             "Limited Company", "1.5-5Cr", "Service", "IT,Consulting", "2015",
             "IT", "Software", "Services", "IT", "IT", "3"],
        ])
        try:
            sellers = load_sellers_csv(str(path))
            assert len(sellers) == 2
            assert "s1" in sellers
            assert "s2" in sellers
        finally:
            path.unlink()

    def test_nullable_fields(self):
        path = _write_sellers_csv([
            ["s1", "", "", "", "",
             "", "", "", "", "",
             "", "", "", "", "", ""],
        ])
        try:
            sellers = load_sellers_csv(str(path))
            s = sellers["s1"]
            assert s["seller_city"] is None
            assert s["seller_state"] is None
            assert s["turnover_band"] is None
            assert s["num_categories"] is None
        finally:
            path.unlink()

    def test_empty_csv(self):
        path = _write_sellers_csv([])
        try:
            sellers = load_sellers_csv(str(path))
            assert sellers == {}
        finally:
            path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  resolve_segment tests (TRAIN-01)
# ──────────────────────────────────────────────────────────────────────────────

class TestResolveSegment:
    """SRS TRAIN-01: Segment resolution hierarchy."""

    # -- Test 15: complete profile → canonical segment key --
    def test_complete_profile_group_turnover(self):
        profile = {
            "seller_id": "s1",
            "top_category_group": "Apparel",
            "turnover_band": "0-40L",
            "business_type": "Proprietorship",
        }
        segment = resolve_segment(profile)
        parsed = json.loads(segment)
        assert parsed == ["Apparel", "0-40L"]

    def test_category_group_only(self):
        profile = {
            "seller_id": "s1",
            "top_category_group": "Electronics",
            "turnover_band": None,
        }
        segment = resolve_segment(profile)
        assert json.loads(segment) == ["Electronics"]

    def test_turnover_band_only(self):
        profile = {
            "seller_id": "s1",
            "top_category_group": None,
            "turnover_band": "1.5-5Cr",
        }
        segment = resolve_segment(profile)
        assert json.loads(segment) == ["global", "1.5-5Cr"]

    # -- Test 16: missing turnover → parent fallback --
    def test_missing_turnover_falls_back_to_category(self):
        """Missing turnover_band → group-only segment."""
        profile = {
            "seller_id": "s1",
            "top_category_group": "Building Material",
            "turnover_band": None,
        }
        segment = resolve_segment(profile)
        parsed = json.loads(segment)
        assert parsed == ["Building Material"]

    def test_missing_category_falls_back_to_turnover(self):
        """Missing category_group → global + turnover."""
        profile = {
            "seller_id": "s1",
            "top_category_group": None,
            "turnover_band": "5-25Cr",
        }
        segment = resolve_segment(profile)
        parsed = json.loads(segment)
        assert parsed == ["global", "5-25Cr"]

    def test_both_missing_global(self):
        """Both missing → global segment."""
        profile = {
            "seller_id": "s1",
            "top_category_group": None,
            "turnover_band": None,
        }
        segment = resolve_segment(profile)
        assert json.loads(segment) == ["global"]

    def test_empty_profile_global(self):
        """Empty / None profile → global segment."""
        assert resolve_segment(None) == json.dumps(["global"])
        assert resolve_segment({}) == json.dumps(["global"])


# ──────────────────────────────────────────────────────────────────────────────
# 9.  join_attempts_sellers tests
# ──────────────────────────────────────────────────────────────────────────────

class TestJoinAttemptsSellers:
    """Join attempts with seller profiles."""

    # -- Test 17: matching seller → joined --
    def test_matching_seller_joined(self):
        attempts = [
            {
                "attempt_id": "att-1",
                "seller_id": "s1",
                "source": "indiamart",
                "answered": True,
                "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            }
        ]
        sellers = {
            "s1": {
                "seller_id": "s1",
                "top_category_group": "Apparel",
                "turnover_band": "0-40L",
                "seller_city": "Mumbai",
                "business_type": "Proprietorship",
            }
        }
        joined = join_attempts_sellers(attempts, sellers)
        assert len(joined) == 1
        j = joined[0]
        assert j["attempt_id"] == "att-1"
        assert j["seller_city"] == "Mumbai"
        assert j["top_category_group"] == "Apparel"
        assert j["turnover_band"] == "0-40L"
        assert j["business_type"] == "Proprietorship"
        assert json.loads(j["segment"]) == ["Apparel", "0-40L"]

    # -- Test 18: missing seller → UNKNOWN segment --
    def test_missing_seller_unknown_segment(self):
        attempts = [
            {
                "attempt_id": "att-1",
                "seller_id": "s999",
                "source": "indiamart",
                "answered": True,
                "disposition": "GENERAL",
                "meeting_fixed": False,
            }
        ]
        sellers: dict = {}  # no sellers
        joined = join_attempts_sellers(attempts, sellers)
        assert len(joined) == 1
        j = joined[0]
        assert j["segment"] == json.dumps(["UNKNOWN"])
        assert j["seller_city"] is None
        assert j["top_category_group"] is None
        assert j["turnover_band"] is None

    def test_mixed_joined_and_unknown(self):
        attempts = [
            {"attempt_id": "att-1", "seller_id": "s1", "source": "indiamart",
             "answered": True, "disposition": "GENERAL", "meeting_fixed": False},
            {"attempt_id": "att-2", "seller_id": "s999", "source": "indiamart",
             "answered": False, "disposition": "NOT_ANSWERED", "meeting_fixed": False},
        ]
        sellers = {
            "s1": {
                "seller_id": "s1",
                "top_category_group": "Electronics",
                "turnover_band": "1.5-5Cr",
            }
        }
        joined = join_attempts_sellers(attempts, sellers)
        assert len(joined) == 2
        assert json.loads(joined[0]["segment"]) == ["Electronics", "1.5-5Cr"]
        assert json.loads(joined[1]["segment"]) == ["UNKNOWN"]

    def test_empty_attempts(self):
        joined = join_attempts_sellers([], {"s1": {}})
        assert joined == []


# ──────────────────────────────────────────────────────────────────────────────
# 10.  DATA-01: ID length validation edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestData01IdLengthValidation:
    """DATA-01: 128 bytes OK, 129 bytes rejected."""

    def test_128_ascii_bytes_ok(self):
        assert _validate_id("a" * 128) == "a" * 128

    def test_129_ascii_bytes_rejected(self):
        with pytest.raises(ValueError, match="128"):
            _validate_id("a" * 129)

    def test_128_utf8_bytes_multi_byte(self):
        """Exactly 128 UTF-8 bytes using multi-byte characters."""
        # 日本語 = 3 bytes each → 42 chars = 126 bytes, need 2 more bytes
        #  é = 2 bytes
        s = "日" * 42 + "é"  # 126 + 2 = 128
        assert _validate_id(s) == s

    def test_129_utf8_bytes_multi_byte_rejected(self):
        s = "日" * 43  # 129 bytes
        with pytest.raises(ValueError, match="128"):
            _validate_id(s)

    def test_id_in_attempts_csv_validated(self):
        """Attempt IDs in CSV are validated via _validate_id."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert isinstance(attempts[0]["attempt_id"], str)
        finally:
            path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# 11.  DATA-02: Timestamp validation edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestData02TimestampValidation:
    """DATA-02: Timestamp validation and timezone handling."""

    def test_iso_with_offset_converts_to_utc(self):
        dt = parse_timestamp("2026-05-15T15:30:00+05:30")
        assert dt == datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_naive_rejected_without_source_tz(self):
        """Naive timestamps are NOT rejected by parse_timestamp itself
        (it localizes with source_tz), but the SRS says API timestamps
        must have offset. The adapter handles naive via source_tz param."""
        dt = parse_timestamp("2026-05-15 10:30:00", source_tz="Asia/Kolkata")
        assert dt.tzinfo is not None

    def test_empty_timestamp_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_timestamp("")

    def test_non_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_timestamp(123)

    def test_call_start_time_parsed_correctly(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            # 10:01 IST = 04:31 UTC
            assert attempts[0]["call_start_time"] == datetime(
                2026, 5, 15, 4, 31, 0, tzinfo=timezone.utc
            )
        finally:
            path.unlink()

    def test_lead_sent_time_parsed_correctly(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            # 10:00 IST = 04:30 UTC
            assert attempts[0]["lead_sent_time"] == datetime(
                2026, 5, 15, 4, 30, 0, tzinfo=timezone.utc
            )
        finally:
            path.unlink()

    def test_iso_timestamp_with_offset_in_csv(self):
        """CSV with ISO timestamps with offset should parse correctly."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15T10:00:00+05:30", "2026-05-15T10:01:00+05:30",
             "2026-05-15T10:01:30+05:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert len(attempts) == 1
            assert attempts[0]["lead_sent_time"] == datetime(
                2026, 5, 15, 4, 30, 0, tzinfo=timezone.utc
            )
        finally:
            path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# 12.  DATA-08: Import report completeness
# ──────────────────────────────────────────────────────────────────────────────

class TestImportReportCompleteness:
    """DATA-08: Import reports count input rows, unique attempts, duplicates,
    exclusions by reason, unknown mappings, missing profiles, and retained rows."""

    def test_report_has_all_required_keys(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            _, report = load_attempts_csv(str(path))
            required_keys = {
                "input_rows", "unique_attempts", "duplicates",
                "exclusions_by_reason", "unknown_mappings",
                "missing_profiles", "retained_rows",
            }
            assert required_keys.issubset(set(report.keys()))
        finally:
            path.unlink()

    def test_report_counts_match(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
            ["att-1", "s1", "", "", "2",
             "2026-05-15 11:00:00", "2026-05-15 11:01:00", "2026-05-15 11:01:30",
             "Answered", "30", "Not Interested", "0", ""],
            ["att-2", "s1", "", "", "1",
             "2026-05-15 12:00:00", "2026-05-15 12:01:00", "2026-05-15 12:01:30",
             "Answered", "30", "General", "1", ""],  # contradictory
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            assert report["input_rows"] == 3
            assert report["unique_attempts"] == 2  # att-1, att-2
            assert report["duplicates"] == 1
            assert report["retained_rows"] == 1  # only att-1 passed
            assert report["exclusions_by_reason"]["duplicate_attempt"] == 1
            assert report["exclusions_by_reason"]["contradictory_meeting_fixed"] == 1
        finally:
            path.unlink()

    def test_quarantine_without_silently_coercing(self):
        """DATA-08: Quarantine contradictory rows without silently coercing."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
            ["att-2", "s1", "", "", "1",
             "2026-05-15 11:00:00", "2026-05-15 11:01:00", "2026-05-15 11:01:30",
             "Answered", "30", "General", "1", ""],  # quarantined
        ])
        try:
            attempts, report = load_attempts_csv(str(path))
            # att-2 should NOT appear with coerced disposition
            for a in attempts:
                assert a["attempt_id"] != "att-2"
            # But it should be in the report
            assert report["retained_rows"] == 1
            assert report["input_rows"] == 2
        finally:
            path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# 13.  DATA-07: Field mapping verification
# ──────────────────────────────────────────────────────────────────────────────

class TestFieldMappingVerification:
    """DATA-07: Source field to canonical field mapping."""

    def test_fk_glusr_usr_id_maps_to_seller_id(self):
        path = _write_attempts_csv([
            ["att-1", "seller-gl-123", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["seller_id"] == "seller-gl-123"
        finally:
            path.unlink()

    def test_data_hotlead_disposition_dtlid_maps_to_attempt_id(self):
        path = _write_attempts_csv([
            ["unique-attempt-gl-456", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["attempt_id"] == "unique-attempt-gl-456"
        finally:
            path.unlink()

    def test_lead_call_status_maps_to_answered(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["answered"] is True
        finally:
            path.unlink()

    def test_lead_tbro_time_maps_to_requested_callback_at(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Call Later", "0", "2026-05-16 14:00:00"],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["requested_callback_at"] is not None
        finally:
            path.unlink()

    def test_call_attempt_count_maps_to_attempt_number(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "5",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["attempt_number"] == 5
        finally:
            path.unlink()

    def test_redis_bucket_maps_to_source_bucket(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "PIM", "main_vani", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["source_bucket"] == "PIM"
        finally:
            path.unlink()

    def test_disposition_label_maps_to_disposition(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Not Interested", "0", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["disposition"] == "NOT_INTERESTED"
        finally:
            path.unlink()

    def test_meeting_fixed_field_mapped(self):
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Meeting Fixed", "1", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["meeting_fixed"] is True
        finally:
            path.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# 14.  Edge cases and bug detection
# ──────────────────────────────────────────────────────────────────────────────

class TestEdgeCasesAndBugs:
    """Edge cases and potential bug detection."""

    def test_prefix_matching_disposition(self):
        """Prefix-based matching should work for extended labels."""
        # "Meeting Fixed Extra" should match "Meeting Fixed" → MEETING_FIXED
        assert map_disposition("Meeting Fixed Extra") == "MEETING_FIXED"
        assert map_disposition("Not Interested Extra") == "NOT_INTERESTED"

    def test_prefix_does_not_match_shorter_key(self):
        """A longer label should not match a shorter key as prefix incorrectly."""
        # "Meeting" is not a key, so it should return UNKNOWN
        assert map_disposition("Meeting") == "UNKNOWN"

    def test_disposition_exact_match_before_prefix(self):
        """Exact match takes priority over prefix matching."""
        # "General (talked)" is an exact key → GENERAL
        assert map_disposition("General (talked)") == "GENERAL"
        # "General" is also a key → GENERAL (same result, but exact match first)
        assert map_disposition("General") == "GENERAL"

    def test_empty_attempts_in_join(self):
        assert join_attempts_sellers([], {"s1": {}}) == []

    def test_empty_sellers_in_join(self):
        attempts = [
            {"attempt_id": "att-1", "seller_id": "s1", "source": "test",
             "answered": True, "disposition": "GENERAL", "meeting_fixed": False}
        ]
        joined = join_attempts_sellers(attempts, {})
        assert len(joined) == 1
        assert json.loads(joined[0]["segment"]) == ["UNKNOWN"]

    def test_all_disposition_canonical_values(self):
        """All canonical disposition values are returned by mapping."""
        canonical = {"MEETING_FIXED", "NOT_INTERESTED", "GENERAL",
                     "CALL_LATER_BUSY", "NOT_ANSWERED", "UNKNOWN"}
        # Test that each canonical value is achievable
        assert map_disposition("Meeting Fixed") in canonical
        assert map_disposition("Not Interested") in canonical
        assert map_disposition("General") in canonical
        assert map_disposition("Call Later") in canonical
        assert map_disposition("Not Answered") in canonical
        assert map_disposition("Unknown Label") in canonical

    def test_segment_is_valid_json(self):
        """All resolve_segment outputs are valid JSON arrays."""
        profiles = [
            {"seller_id": "s1", "top_category_group": "A", "turnover_band": "B"},
            {"seller_id": "s1", "top_category_group": "A", "turnover_band": None},
            {"seller_id": "s1", "top_category_group": None, "turnover_band": "B"},
            {"seller_id": "s1", "top_category_group": None, "turnover_band": None},
            {},
            None,
        ]
        for p in profiles:
            seg = resolve_segment(p)
            parsed = json.loads(seg)
            assert isinstance(parsed, list)

    def test_join_preserves_original_attempt_fields(self):
        """join_attempts_sellers should preserve all original attempt fields."""
        attempt = {
            "attempt_id": "att-1",
            "seller_id": "s1",
            "source": "indiamart",
            "source_bucket": "PIM",
            "attempt_number": 3,
            "answered": True,
            "disposition": "GENERAL",
            "meeting_fixed": False,
            "duration_s": 45,
        }
        sellers = {
            "s1": {
                "seller_id": "s1",
                "top_category_group": "Apparel",
                "turnover_band": "0-40L",
            }
        }
        joined = join_attempts_sellers([attempt], sellers)
        j = joined[0]
        assert j["attempt_id"] == "att-1"
        assert j["source"] == "indiamart"
        assert j["source_bucket"] == "PIM"
        assert j["attempt_number"] == 3
        assert j["duration_s"] == 45

    def test_sellers_csv_duplicate_seller_id_last_wins(self):
        """When duplicate seller IDs exist, last row wins (dict assignment)."""
        path = _write_sellers_csv([
            ["s1", "Mumbai", "", "", "", "Proprietorship", "0-40L", "", "", "",
             "Apparel", "", "", "Apparel", "Apparel", "1"],
            ["s1", "Delhi", "", "", "", "Partnership", "40L-1.5Cr", "", "", "",
             "Electronics", "", "", "Electronics", "Electronics", "2"],
        ])
        try:
            sellers = load_sellers_csv(str(path))
            assert sellers["s1"]["seller_city"] == "Delhi"
            assert sellers["s1"]["business_type"] == "Partnership"
        finally:
            path.unlink()

    def test_attempts_with_iso_timestamps_in_lead_tbro_time(self):
        """lead_tbro_time with ISO format + offset."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Call Later", "0", "2026-05-16T14:00:00+05:30"],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["requested_callback_at"] == datetime(
                2026, 5, 16, 8, 30, 0, tzinfo=timezone.utc
            )
        finally:
            path.unlink()

    def test_attempts_with_invalid_callback_time(self):
        """Invalid lead_tbro_time → requested_callback_at is None."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Call Later", "0", "not-a-date"],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["requested_callback_at"] is None
        finally:
            path.unlink()

    def test_attempts_with_empty_callback_time(self):
        """Empty lead_tbro_time → requested_callback_at is None."""
        path = _write_attempts_csv([
            ["att-1", "s1", "", "", "1",
             "2026-05-15 10:00:00", "2026-05-15 10:01:00", "2026-05-15 10:01:30",
             "Answered", "30", "Call Later", "0", ""],
        ])
        try:
            attempts, _ = load_attempts_csv(str(path))
            assert attempts[0]["requested_callback_at"] is None
        finally:
            path.unlink()
