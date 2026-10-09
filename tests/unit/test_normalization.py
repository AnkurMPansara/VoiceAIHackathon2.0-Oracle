"""Unit tests for btc.data.normalization module.

Tests data normalization, chronological splits, segment statistics,
support bins, point-in-time profiles, and split integrity validation.

SRS References:
- TRAIN-01: Segment resolution eligibility (>=2000 attempts, >=200 sellers)
- TRAIN-05: Support bin thresholds (>=50 attempts, >=30 sellers per 15-min bin)
- TRAIN-06: Chronological splits (disjoint, prior-fit isolation)
- DATA-01: ID string conversion (max 128 UTF-8 bytes)
- DATA-02: Timestamp handling (UTC timezone-aware)
- DATA-04: Cross-field validation (contradictory labels)
- MOD-01: Reward model fields
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from btc.data.normalization import (
    _bin_key,
    _parse_bin_key,
    build_seller_profiles,
    compute_segment_statistics,
    compute_support_bins,
    create_chronological_splits,
    isolate_prior_fit_data,
    normalize_batch,
    normalize_outcome,
    point_in_time_profile,
    validate_split_integrity,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_sellers():
    """Sample seller profiles for testing."""
    return {
        "S001": {
            "seller_id": "S001",
            "top_category_group": "Real Estate",
            "turnover_band": "High",
            "business_type": "Agency",
            "effective_from": "2026-01-01T00:00:00+05:30",
            "profile_version": "v1",
        },
        "S002": {
            "seller_id": "S002",
            "top_category_group": "Tech",
            "turnover_band": "Low",
            "business_type": "Startup",
            "effective_from": "2026-01-01T00:00:00+05:30",
            "profile_version": "v1",
        },
    }


@pytest.fixture
def valid_raw_record():
    """A valid raw attempt record."""
    return {
        "seller_id": "S001",
        "lead_id": "L001",
        "attempt_id": "A001",
        "source": "test",
        "disposition_label": "Meeting Fixed",
        "lead_call_status": "Answered",
        "finalized_at": "2026-05-15T10:30:00+05:30",
        "call_start_time": "2026-05-15T10:00:00+05:30",
        "call_end_time": "2026-05-15T10:10:00+05:30",
        "lead_sent_time": "2026-05-15T09:55:00+05:30",
        "call_attempt_count": 3,
        "meeting_fixed": 1,
    }


@pytest.fixture
def chronological_records():
    """Records spanning multiple split intervals."""
    records = []
    for i in range(10):
        records.append({
            "seller_id": f"S{i}",
            "lead_id": f"L{i}",
            "attempt_id": f"A{i}",
            "source": "test",
            "finalized_at": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
            "call_start_time": datetime(2026, 5, 15, 9, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2026, 5, 15, 9, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2026, 5, 15, 8, 55, 0, tzinfo=timezone.utc),
            "segment": "Real Estate",
            "answered": True,
            "disposition": "MEETING_FIXED",
            "meeting_fixed": True,
        })
    for i in range(10, 20):
        records.append({
            "seller_id": f"S{i}",
            "lead_id": f"L{i}",
            "attempt_id": f"A{i}",
            "source": "test",
            "finalized_at": datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc),
            "call_start_time": datetime(2026, 7, 15, 9, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2026, 7, 15, 9, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2026, 7, 15, 8, 55, 0, tzinfo=timezone.utc),
            "segment": "Real Estate",
            "answered": True,
            "disposition": "MEETING_FIXED",
            "meeting_fixed": True,
        })
    return records


# ── normalize_outcome tests ───────────────────────────────────────────────────


class TestNormalizeOutcome:
    """SRS DATA-01, DATA-02, DATA-04: Record normalization."""

    # Test 1: Valid outcome -> normalized with canonical fields
    def test_normalizes_valid_record(self, valid_raw_record, sample_sellers):
        """normalize_outcome produces canonical schema from valid input."""
        result = normalize_outcome(valid_raw_record, sample_sellers)
        assert result["seller_id"] == "S001"
        assert result["lead_id"] == "L001"
        assert result["attempt_id"] == "A001"
        assert result["source"] == "test"
        assert result["disposition"] == "MEETING_FIXED"
        assert result["answered"] is True
        assert result["meeting_fixed"] is True
        assert result["attempt_number"] == 3
        assert result["segment"] == '["Real Estate", "High"]'
        assert result["revision"] == 1
        assert isinstance(result["finalized_at"], datetime)
        assert isinstance(result["call_start_time"], datetime)
        assert isinstance(result["call_end_time"], datetime)
        assert isinstance(result["lead_sent_time"], datetime)

    # Test 2: Disposition mapping -> canonical value
    def test_maps_disposition_labels(self, sample_sellers):
        """Various disposition label formats map to canonical values."""
        for label, expected in [
            ("Meeting Fixed", "MEETING_FIXED"),
            ("Meeting_Fixed", "MEETING_FIXED"),
            ("Not Answered", "NOT_ANSWERED"),
            ("NotAnswered", "NOT_ANSWERED"),
            ("Not Interested", "NOT_INTERESTED"),
            ("General", "GENERAL"),
            ("General (talked)", "GENERAL"),
            ("Call Later", "CALL_LATER_BUSY"),
            ("Busy", "CALL_LATER_BUSY"),
            ("Unknown", "UNKNOWN"),
            ("Wrong Number", "UNKNOWN"),
            ("", "UNKNOWN"),
            (None, "UNKNOWN"),
        ]:
            raw = {
                "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
                "source": "test", "lead_call_status": "NotAnswered",
                "finalized_at": "2026-05-15T10:30:00+05:30",
                "call_start_time": "2026-05-15T10:00:00+05:30",
                "call_end_time": "2026-05-15T10:10:00+05:30",
                "lead_sent_time": "2026-05-15T09:55:00+05:30",
                "meeting_fixed": 0,
            }
            if label is not None:
                raw["disposition_label"] = label
            result = normalize_outcome(raw, sample_sellers)
            assert result["disposition"] == expected, f"Failed for label: {label!r}"

    # Test 3: Timestamp -> UTC timezone-aware
    def test_timestamps_converted_to_utc(self, sample_sellers):
        """All timestamps are converted to UTC and are timezone-aware."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        for field in ["finalized_at", "call_start_time", "call_end_time", "lead_sent_time"]:
            ts = result[field]
            assert ts is not None, f"{field} should not be None"
            assert ts.tzinfo is not None, f"{field} should be timezone-aware"
            assert ts.utcoffset() == timedelta(seconds=0), f"{field} should be UTC"

    def test_timestamps_in_utc(self, sample_sellers):
        """Timestamps from +05:30 are correctly converted to UTC."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["finalized_at"] == datetime(2026, 5, 15, 5, 0, 0, tzinfo=timezone.utc)
        assert result["call_start_time"] == datetime(2026, 5, 15, 4, 30, 0, tzinfo=timezone.utc)
        assert result["call_end_time"] == datetime(2026, 5, 15, 4, 40, 0, tzinfo=timezone.utc)
        assert result["lead_sent_time"] == datetime(2026, 5, 15, 4, 25, 0, tzinfo=timezone.utc)

    def test_naive_timestamp_raises(self, sample_sellers):
        """Naive timestamps raise ValueError (DATA-02)."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        with pytest.raises(ValueError, match="Timestamp must include an explicit timezone offset"):
            normalize_outcome(raw, sample_sellers)

    def test_null_timestamps_allowed(self, sample_sellers):
        """Null timestamps are allowed and produce None."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["finalized_at"] is None
        assert result["requested_callback_at"] is None

    # Test 4: meeting_fixed 1 -> True, 0 -> False
    def test_converts_meeting_fixed(self, sample_sellers):
        """meeting_fixed converts from various formats to bool."""
        for value, expected in [
            (1, True), (0, False), ("1", True), ("0", False),
            (True, True), (False, False),
        ]:
            raw = {
                "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
                "source": "test", "disposition_label": "Meeting Fixed",
                "lead_call_status": "Answered",
                "finalized_at": "2026-05-15T10:30:00+05:30",
                "call_start_time": "2026-05-15T10:00:00+05:30",
                "call_end_time": "2026-05-15T10:10:00+05:30",
                "lead_sent_time": "2026-05-15T09:55:00+05:30",
                "meeting_fixed": value,
            }
            result = normalize_outcome(raw, sample_sellers)
            assert result["meeting_fixed"] == expected, f"Failed for value: {value!r}"

    # Test 5: call_attempt_count -> int
    def test_converts_attempt_count(self, sample_sellers):
        """call_attempt_count converts to int."""
        for value, expected in [
            ("3", 3), (3, 3), (3.9, 3),
        ]:
            raw = {
                "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
                "source": "test", "disposition_label": "General",
                "lead_call_status": "NotAnswered",
                "finalized_at": "2026-05-15T10:30:00+05:30",
                "call_start_time": "2026-05-15T10:00:00+05:30",
                "call_end_time": "2026-05-15T10:10:00+05:30",
                "lead_sent_time": "2026-05-15T09:55:00+05:30",
                "meeting_fixed": 0,
                "call_attempt_count": value,
            }
            result = normalize_outcome(raw, sample_sellers)
            assert result["attempt_number"] == expected, f"Failed for value: {value!r}"

    def test_attempt_count_defaults_to_1(self, sample_sellers):
        """Missing call_attempt_count defaults to 1."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["attempt_number"] == 1

    def test_attempt_count_invalid_defaults(self, sample_sellers):
        """Non-numeric call_attempt_count defaults to 1."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
            "call_attempt_count": "invalid",
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["attempt_number"] == 1

    # Test 6: ID -> string
    def test_ids_converted_to_strings(self, sample_sellers):
        """All IDs are converted to strings (DATA-01)."""
        raw = {
            "seller_id": 12345,
            "lead_id": 67890,
            "attempt_id": 11111,
            "source": "test_source",
            "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["seller_id"] == "12345"
        assert result["lead_id"] == "67890"
        assert result["attempt_id"] == "11111"
        assert isinstance(result["seller_id"], str)
        assert isinstance(result["lead_id"], str)
        assert isinstance(result["attempt_id"], str)

    def test_id_max_128_bytes(self, sample_sellers):
        """IDs exceeding 128 UTF-8 bytes raise ValueError (DATA-01)."""
        long_id = "A" * 130
        raw = {
            "seller_id": long_id,
            "lead_id": "L001",
            "attempt_id": "A001",
            "source": "test",
            "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        with pytest.raises(ValueError, match="exceeds 128 UTF-8 bytes"):
            normalize_outcome(raw, sample_sellers)

    def test_null_id_becomes_empty_string(self, sample_sellers):
        """Null IDs become empty strings."""
        raw = {
            "seller_id": None,
            "lead_id": None,
            "attempt_id": None,
            "source": "test",
            "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["seller_id"] == ""
        assert result["lead_id"] == ""
        assert result["attempt_id"] == ""

    # Test 7: Contradictory labels -> raises ValueError
    def test_rejects_contradictory_meeting_fixed(self, sample_sellers):
        """meeting_fixed=True with answered=False raises ValueError (DATA-04)."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "Meeting Fixed",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 1,
        }
        with pytest.raises(ValueError, match="meeting_fixed=True requires answered=True"):
            normalize_outcome(raw, sample_sellers)

    def test_rejects_contradictory_meeting_fixed_wrong_disposition(self, sample_sellers):
        """meeting_fixed=True with disposition != MEETING_FIXED raises ValueError."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "Answered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 1,
        }
        with pytest.raises(ValueError, match="meeting_fixed=True requires answered=True"):
            normalize_outcome(raw, sample_sellers)

    def test_rejects_not_answered_with_answered(self, sample_sellers):
        """NOT_ANSWERED disposition with answered=True raises ValueError (DATA-04)."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "Not Answered",
            "lead_call_status": "Answered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        with pytest.raises(ValueError, match="disposition=NOT_ANSWERED requires answered=False"):
            normalize_outcome(raw, sample_sellers)

    def test_rejects_call_start_after_end(self, sample_sellers):
        """call_start_time > call_end_time raises ValueError (DATA-04)."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:20:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        with pytest.raises(ValueError, match="call_start_time must be <= call_end_time"):
            normalize_outcome(raw, sample_sellers)

    def test_rejects_lead_sent_after_call_start(self, sample_sellers):
        """lead_sent_time > call_start_time raises ValueError (DATA-04)."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T10:15:00+05:30",
            "meeting_fixed": 0,
        }
        with pytest.raises(ValueError, match="lead_sent_time must be <= call_start_time"):
            normalize_outcome(raw, sample_sellers)

    def test_valid_equal_boundaries_allowed(self, sample_sellers):
        """call_start_time == call_end_time is allowed."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:00:00+05:30",
            "lead_sent_time": "2026-05-15T10:00:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["duration_s"] == 0

    def test_missing_seller_fallback(self):
        """Missing seller profile falls back to global segment."""
        raw = {
            "seller_id": "UNKNOWN_SELLER", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, {})
        assert result["segment"] == '["global"]'

    def test_duration_computed_from_timestamps(self, sample_sellers):
        """duration_s computed from call_end - call_start when not provided."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["duration_s"] == 600

    def test_duration_from_raw(self, sample_sellers):
        """duration_s from raw record is used when provided."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
            "duration_s": 500,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["duration_s"] == 500

    def test_decision_id_nullable(self, sample_sellers):
        """decision_id is None when not provided or empty."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["decision_id"] is None

    def test_decision_id_string(self, sample_sellers):
        """decision_id is string when provided."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "General",
            "lead_call_status": "NotAnswered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
            "decision_id": "DEC-123",
        }
        result = normalize_outcome(raw, sample_sellers)
        assert result["decision_id"] == "DEC-123"

    def test_canonical_fields_in_result(self, sample_sellers):
        """Normalized result contains all canonical fields."""
        raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "Meeting Fixed",
            "lead_call_status": "Answered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "call_attempt_count": 3,
            "meeting_fixed": 1,
        }
        result = normalize_outcome(raw, sample_sellers)
        expected_fields = {
            "seller_id", "lead_id", "attempt_id", "source", "revision",
            "event_id", "finalized_at", "call_start_time", "call_end_time",
            "lead_sent_time", "attempt_number", "answered", "disposition",
            "meeting_fixed", "requested_callback_at", "decision_id",
            "duration_s", "dialer_version", "source_bucket", "segment",
        }
        assert set(result.keys()) == expected_fields


# ── create_chronological_splits tests ─────────────────────────────────────────


class TestChronologicalSplits:
    """SRS TRAIN-06: Chronological interval splits."""

    # Test 8: Splits are disjoint (no duplicate attempts)
    def test_disjoint_splits(self, chronological_records):
        """No attempt appears in multiple splits."""
        splits = create_chronological_splits(chronological_records)
        all_attempts = []
        for purpose, records in splits.items():
            for r in records:
                all_attempts.append(r["attempt_id"])
        assert len(all_attempts) == len(set(all_attempts)), "Duplicate attempts across splits"

    # Test 9: prior_fit: Apr-Jul, warmup: Jul-Aug, validation: Aug-Sep, test: Sep-Oct
    def test_split_boundaries(self, chronological_records):
        """Records are assigned to correct splits by boundary."""
        splits = create_chronological_splits(chronological_records)
        assert len(splits["prior_fit"]) == 10
        assert len(splits["warmup"]) == 10
        assert len(splits["validation"]) == 0
        assert len(splits["test"]) == 0

    def test_all_four_splits_present(self):
        """All four split keys exist."""
        splits = create_chronological_splits([])
        assert "prior_fit" in splits
        assert "warmup" in splits
        assert "validation" in splits
        assert "test" in splits

    def test_split_intervals_are_half_open(self):
        """Intervals are [start, end) - boundary goes to next split."""
        # Boundary at 2026-07-01T00:00:00+05:30 = 2026-06-30T18:30:00 UTC
        # Record at exact boundary goes to warmup (next interval), not prior_fit
        # because [start, end) is half-open: boundary belongs to next interval
        boundary_utc = datetime(2026, 6, 30, 18, 30, 0, tzinfo=timezone.utc)
        record = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A_BOUNDARY",
            "source": "test",
            "finalized_at": boundary_utc,
            "call_start_time": datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2026, 6, 30, 10, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2026, 6, 30, 9, 55, 0, tzinfo=timezone.utc),
            "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
            "meeting_fixed": True,
        }
        splits = create_chronological_splits([record])
        assert len(splits["prior_fit"]) == 0
        assert len(splits["warmup"]) == 1

    def test_boundary_after_split(self):
        """Record just after boundary goes to next split."""
        # 1 second after boundary
        boundary_utc = datetime(2026, 6, 30, 18, 30, 0, tzinfo=timezone.utc)
        just_after = boundary_utc + timedelta(seconds=1)
        record = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A_JUST_AFTER",
            "source": "test",
            "finalized_at": just_after,
            "call_start_time": datetime(2026, 6, 30, 10, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2026, 6, 30, 10, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2026, 6, 30, 9, 55, 0, tzinfo=timezone.utc),
            "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
            "meeting_fixed": True,
        }
        splits = create_chronological_splits([record])
        assert len(splits["prior_fit"]) == 0
        assert len(splits["warmup"]) == 1

    def test_all_attempts_assigned_exactly_once(self):
        """Every attempt is assigned to exactly one split (or skipped if outside ranges)."""
        records = []
        for i in range(5):
            records.append({
                "seller_id": f"S{i}", "lead_id": f"L{i}", "attempt_id": f"A{i}",
                "source": "test",
                "finalized_at": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
                "call_start_time": datetime(2026, 5, 15, 9, 0, 0, tzinfo=timezone.utc),
                "call_end_time": datetime(2026, 5, 15, 9, 10, 0, tzinfo=timezone.utc),
                "lead_sent_time": datetime(2026, 5, 15, 8, 55, 0, tzinfo=timezone.utc),
                "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            })
        splits = create_chronological_splits(records)
        total = sum(len(v) for v in splits.values())
        assert total == 5

    def test_outside_intervals_skipped(self):
        """Records outside all intervals are not assigned."""
        record = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test",
            "finalized_at": datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "call_start_time": datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2025, 1, 1, 0, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
            "meeting_fixed": True,
        }
        splits = create_chronological_splits([record])
        total = sum(len(v) for v in splits.values())
        assert total == 0

    def test_empty_data(self):
        """Empty input produces empty splits."""
        splits = create_chronological_splits([])
        for purpose in ["prior_fit", "warmup", "validation", "test"]:
            assert splits[purpose] == []

    def test_null_finalized_at_skipped(self):
        """Records with null finalized_at are skipped."""
        record = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test",
            "finalized_at": None,
            "call_start_time": datetime(2026, 5, 15, 9, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2026, 5, 15, 9, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2026, 5, 15, 8, 55, 0, tzinfo=timezone.utc),
            "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
            "meeting_fixed": True,
        }
        splits = create_chronological_splits([record])
        total = sum(len(v) for v in splits.values())
        assert total == 0

    def test_split_contains_all_expected_periods(self):
        """Records span all four split periods correctly."""
        records = [
            # prior_fit: Apr-Jul
            {
                "seller_id": "S0", "lead_id": "L0", "attempt_id": "A0",
                "source": "test",
                "finalized_at": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
                "call_start_time": datetime(2026, 5, 15, 9, 0, 0, tzinfo=timezone.utc),
                "call_end_time": datetime(2026, 5, 15, 9, 10, 0, tzinfo=timezone.utc),
                "lead_sent_time": datetime(2026, 5, 15, 8, 55, 0, tzinfo=timezone.utc),
                "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            },
            # warmup: Jul-Aug
            {
                "seller_id": "S1", "lead_id": "L1", "attempt_id": "A1",
                "source": "test",
                "finalized_at": datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc),
                "call_start_time": datetime(2026, 7, 15, 9, 0, 0, tzinfo=timezone.utc),
                "call_end_time": datetime(2026, 7, 15, 9, 10, 0, tzinfo=timezone.utc),
                "lead_sent_time": datetime(2026, 7, 15, 8, 55, 0, tzinfo=timezone.utc),
                "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            },
            # validation: Aug-Sep
            {
                "seller_id": "S2", "lead_id": "L2", "attempt_id": "A2",
                "source": "test",
                "finalized_at": datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc),
                "call_start_time": datetime(2026, 8, 15, 9, 0, 0, tzinfo=timezone.utc),
                "call_end_time": datetime(2026, 8, 15, 9, 10, 0, tzinfo=timezone.utc),
                "lead_sent_time": datetime(2026, 8, 15, 8, 55, 0, tzinfo=timezone.utc),
                "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            },
            # test: Sep-Oct
            {
                "seller_id": "S3", "lead_id": "L3", "attempt_id": "A3",
                "source": "test",
                "finalized_at": datetime(2026, 9, 15, 10, 0, 0, tzinfo=timezone.utc),
                "call_start_time": datetime(2026, 9, 15, 9, 0, 0, tzinfo=timezone.utc),
                "call_end_time": datetime(2026, 9, 15, 9, 10, 0, tzinfo=timezone.utc),
                "lead_sent_time": datetime(2026, 9, 15, 8, 55, 0, tzinfo=timezone.utc),
                "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            },
        ]
        splits = create_chronological_splits(records)
        assert len(splits["prior_fit"]) == 1
        assert len(splits["warmup"]) == 1
        assert len(splits["validation"]) == 1
        assert len(splits["test"]) == 1

    def test_prior_fit_not_replayed_to_warmup(self):
        """TRAIN-06: Prior-fit rows are NOT replayed into seller state."""
        prior_records = [
            {
                "seller_id": "S001", "lead_id": "L001", "attempt_id": f"PF{i}",
                "source": "test",
                "finalized_at": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
                "call_start_time": datetime(2026, 5, 15, 9, 0, 0, tzinfo=timezone.utc),
                "call_end_time": datetime(2026, 5, 15, 9, 10, 0, tzinfo=timezone.utc),
                "lead_sent_time": datetime(2026, 5, 15, 8, 55, 0, tzinfo=timezone.utc),
                "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            }
            for i in range(5)
        ]
        warmup_records = [
            {
                "seller_id": "S002", "lead_id": "L002", "attempt_id": f"W{i}",
                "source": "test",
                "finalized_at": datetime(2026, 7, 15, 10, 0, 0, tzinfo=timezone.utc),
                "call_start_time": datetime(2026, 7, 15, 9, 0, 0, tzinfo=timezone.utc),
                "call_end_time": datetime(2026, 7, 15, 9, 10, 0, tzinfo=timezone.utc),
                "lead_sent_time": datetime(2026, 7, 15, 8, 55, 0, tzinfo=timezone.utc),
                "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
                "meeting_fixed": True,
            }
            for i in range(3)
        ]
        all_records = prior_records + warmup_records
        splits = create_chronological_splits(all_records)
        prior_ids = {r["attempt_id"] for r in splits["prior_fit"]}
        warmup_ids = {r["attempt_id"] for r in splits["warmup"]}
        assert prior_ids.isdisjoint(warmup_ids)

    def test_custom_timezone(self):
        """Custom timezone parameter is used for boundaries."""
        record = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test",
            "finalized_at": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
            "call_start_time": datetime(2026, 5, 15, 9, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2026, 5, 15, 9, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2026, 5, 15, 8, 55, 0, tzinfo=timezone.utc),
            "segment": "Test", "answered": True, "disposition": "MEETING_FIXED",
            "meeting_fixed": True,
        }
        splits = create_chronological_splits([record], timezone="Asia/Kolkata")
        assert len(splits["prior_fit"]) == 1


# ── compute_segment_statistics tests ──────────────────────────────────────────


class TestSegmentStatistics:
    """SRS TRAIN-01: Segment eligibility computation."""

    # Test 13: Segment with >=2000 attempts and >=200 sellers -> eligible
    def test_eligible_segment(self):
        """Segment with >=2000 attempts and >=200 sellers is eligible."""
        data = []
        for i in range(200):
            for j in range(10):
                data.append({"seller_id": f"S{i}", "segment": "SegA"})
        stats = compute_segment_statistics(data)
        assert stats["SegA"]["n_sellers"] == 200
        assert stats["SegA"]["n_attempts"] == 2000
        assert stats["SegA"]["eligible"] is True

    # Test 14: Segment with <2000 attempts -> not eligible
    def test_ineligible_low_attempts(self):
        """Segment with <2000 attempts is not eligible."""
        data = [{"seller_id": f"S{i}", "segment": "SegA"} for i in range(10)]
        stats = compute_segment_statistics(data)
        assert stats["SegA"]["eligible"] is False
        assert stats["SegA"]["n_attempts"] == 10

    # Test 15: Segment with >=2000 attempts but <200 sellers -> not eligible
    def test_ineligible_low_sellers(self):
        """Segment with >=2000 attempts but <200 sellers is not eligible."""
        data = []
        for i in range(50):
            for j in range(40):
                data.append({"seller_id": f"S{i}", "segment": "SegA"})
        stats = compute_segment_statistics(data)
        assert stats["SegA"]["n_sellers"] == 50
        assert stats["SegA"]["n_attempts"] == 2000
        assert stats["SegA"]["eligible"] is False

    def test_computes_segment_counts(self):
        """Counts sellers and attempts per segment."""
        data = [
            {"seller_id": "S1", "segment": "SegA"},
            {"seller_id": "S1", "segment": "SegA"},
            {"seller_id": "S2", "segment": "SegA"},
            {"seller_id": "S3", "segment": "SegB"},
        ]
        stats = compute_segment_statistics(data)
        assert stats["SegA"]["n_attempts"] == 3
        assert stats["SegA"]["n_sellers"] == 2
        assert stats["SegB"]["n_attempts"] == 1
        assert stats["SegB"]["n_sellers"] == 1

    def test_eligibility_thresholds(self):
        """Eligibility check uses configured thresholds."""
        data = [{"seller_id": f"S{i}", "segment": "SegA"} for i in range(10)]
        stats = compute_segment_statistics(data, min_attempts=5, min_sellers=5)
        assert stats["SegA"]["eligible"] is True
        stats = compute_segment_statistics(data, min_attempts=15, min_sellers=5)
        assert stats["SegA"]["eligible"] is False
        stats = compute_segment_statistics(data, min_attempts=5, min_sellers=15)
        assert stats["SegA"]["eligible"] is False

    def test_default_thresholds(self):
        """Default thresholds are 2000 attempts and 200 sellers."""
        stats = compute_segment_statistics([{"seller_id": "S1", "segment": "X"}])
        assert stats["X"]["eligible"] is False

    def test_multiple_segments(self):
        """Multiple segments computed independently."""
        data = []
        for i in range(200):
            for j in range(10):
                data.append({"seller_id": f"S{i}", "segment": "SegA"})
        for i in range(50):
            for j in range(10):
                data.append({"seller_id": f"S{i}", "segment": "SegB"})
        stats = compute_segment_statistics(data)
        assert stats["SegA"]["eligible"] is True
        assert stats["SegB"]["eligible"] is False

    def test_null_seller_id_not_counted(self):
        """Null seller_id is not counted toward distinct sellers."""
        data = [
            {"seller_id": None, "segment": "SegA"},
            {"seller_id": None, "segment": "SegA"},
            {"seller_id": "S1", "segment": "SegA"},
        ]
        stats = compute_segment_statistics(data)
        assert stats["SegA"]["n_sellers"] == 1
        assert stats["SegA"]["n_attempts"] == 3

    def test_global_segment(self):
        """Global segment eligible when meeting thresholds."""
        data = []
        for i in range(250):
            for j in range(8):
                data.append({"seller_id": f"S{i}", "segment": '["global"]'})
        stats = compute_segment_statistics(data)
        global_key = '["global"]'
        assert stats[global_key]["n_sellers"] == 250
        assert stats[global_key]["n_attempts"] == 2000
        assert stats[global_key]["eligible"] is True


# ── compute_support_bins tests ────────────────────────────────────────────────


class TestSupportBins:
    """SRS TRAIN-05: 15-minute support bin computation."""

    # Test 17: Bin with >=50 attempts and >=30 sellers -> supported
    def test_supported_bin(self):
        """Bin with >=50 attempts and >=30 sellers is supported."""
        data = []
        for i in range(30):
            for j in range(2):
                data.append({
                    "seller_id": f"S{i}",
                    "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
                    "segment": "SegA",
                })
        bins = compute_support_bins(data)
        assert bins["SegA"]["2026-05-15T10:00"] is True

    # Test 18: Bin with <50 attempts -> unsupported
    def test_unsupported_low_attempts(self):
        """Bin with <50 attempts is unsupported."""
        data = [
            {"seller_id": f"S{i}", "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc), "segment": "SegA"}
            for i in range(49)
        ]
        bins = compute_support_bins(data, min_attempts=50, min_sellers=1)
        assert bins["SegA"]["2026-05-15T10:00"] is False

    # Test 19: Bin with >=50 attempts but <30 sellers -> unsupported
    def test_unsupported_low_sellers(self):
        """Bin with >=50 attempts but <30 sellers is unsupported."""
        data = []
        for i in range(5):
            for j in range(10):
                data.append({
                    "seller_id": f"S{i}",
                    "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
                    "segment": "SegA",
                })
        bins = compute_support_bins(data, min_attempts=50, min_sellers=30)
        assert bins["SegA"]["2026-05-15T10:00"] is False
        assert bins["SegA"]["2026-05-15T10:00"] is False  # 50 attempts, 5 sellers

    # Test 20: Half-open intervals: [start, end)
    def test_bins_half_open_intervals(self):
        """Records fall into correct 15-minute bins (half-open [start, end))."""
        data = [
            {"seller_id": "S1", "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc), "segment": "SegA"},
            {"seller_id": "S2", "call_start_time": datetime(2026, 5, 15, 10, 7, 0, tzinfo=timezone.utc), "segment": "SegA"},
            {"seller_id": "S3", "call_start_time": datetime(2026, 5, 15, 10, 15, 0, tzinfo=timezone.utc), "segment": "SegA"},
        ]
        bins = compute_support_bins(data)
        assert "SegA" in bins
        assert "2026-05-15T10:00" in bins["SegA"]
        assert "2026-05-15T10:15" in bins["SegA"]
        assert bins["SegA"]["2026-05-15T10:00"] is False  # 2 attempts, 2 sellers
        assert bins["SegA"]["2026-05-15T10:15"] is False  # 1 attempt, 1 seller

    def test_multiple_bins(self):
        """Multiple 15-minute bins are computed independently."""
        data = [
            {"seller_id": f"S{i}", "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc), "segment": "SegA"}
            for i in range(3)
        ]
        data += [
            {"seller_id": f"S{i}", "call_start_time": datetime(2026, 5, 15, 10, 15, 0, tzinfo=timezone.utc), "segment": "SegA"}
            for i in range(3)
        ]
        bins = compute_support_bins(data, min_attempts=2, min_sellers=2)
        assert bins["SegA"]["2026-05-15T10:00"] is True
        assert bins["SegA"]["2026-05-15T10:15"] is True

    def test_missing_call_start_skipped(self):
        """Records with missing call_start_time are skipped."""
        data = [
            {"seller_id": "S1", "call_start_time": None, "segment": "SegA"},
            {"seller_id": "S2", "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc), "segment": "SegA"},
        ]
        bins = compute_support_bins(data)
        assert "SegA" in bins
        assert "2026-05-15T10:00" in bins["SegA"]

    def test_bin_key_roundtrip(self):
        """_bin_key and _parse_bin_key are inverse operations."""
        dt = datetime(2026, 5, 15, 10, 7, 30, tzinfo=timezone.utc)
        key = _bin_key(dt)
        parsed = _parse_bin_key(key)
        assert parsed == datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)

    def test_bin_key_midnight(self):
        """Bin key at midnight is correct."""
        dt = datetime(2026, 5, 15, 0, 7, 30, tzinfo=timezone.utc)
        key = _bin_key(dt)
        assert key == "2026-05-15T00:00"

    def test_bin_key_end_of_hour(self):
        """Bin key at :45 falls into the :45 bin."""
        dt = datetime(2026, 5, 15, 10, 47, 30, tzinfo=timezone.utc)
        key = _bin_key(dt)
        assert key == "2026-05-15T10:45"

    def test_support_thresholds(self):
        """Bin support requires min_attempts AND min_sellers."""
        data = [{"seller_id": "S1", "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc), "segment": "SegA"}]
        bins = compute_support_bins(data, min_attempts=2, min_sellers=1)
        assert bins["SegA"]["2026-05-15T10:00"] is False
        data = [
            {"seller_id": f"S{i}", "call_start_time": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc), "segment": "SegA"}
            for i in range(3)
        ]
        bins = compute_support_bins(data, min_attempts=2, min_sellers=2)
        assert bins["SegA"]["2026-05-15T10:00"] is True


# ── point_in_time_profile tests ───────────────────────────────────────────────


class TestPointInTimeProfile:
    """Point-in-time seller profile resolution."""

    # Test 21: Profile effective at as_of_time -> correct version
    def test_returns_earliest_effective_profile(self):
        """Returns profile whose effective_from <= as_of_time."""
        profiles = [
            {"seller_id": "S1", "effective_from": datetime(2026, 1, 1, tzinfo=timezone.utc), "category": "A"},
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc), "category": "B"},
        ]
        result = point_in_time_profile(profiles, "S1", datetime(2026, 3, 1, tzinfo=timezone.utc))
        assert result is not None
        assert result["category"] == "A"

    def test_returns_latest_effective_profile(self):
        """Returns the latest profile active at as_of_time."""
        profiles = [
            {"seller_id": "S1", "effective_from": datetime(2026, 1, 1, tzinfo=timezone.utc), "category": "A"},
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc), "category": "B"},
        ]
        result = point_in_time_profile(profiles, "S1", datetime(2026, 9, 1, tzinfo=timezone.utc))
        assert result is not None
        assert result["category"] == "B"

    # Test 22: No profile before effective_from -> None
    def test_returns_none_before_any_profile(self):
        """Returns None when as_of_time is before all profiles."""
        profiles = [
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc), "category": "A"},
        ]
        result = point_in_time_profile(profiles, "S1", datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert result is None

    def test_returns_none_for_empty_profiles(self):
        """Returns None for empty profile list."""
        assert point_in_time_profile([], "S1", datetime(2026, 1, 1, tzinfo=timezone.utc)) is None

    def test_raises_on_naive_datetime(self):
        """Raises ValueError if as_of_time is naive."""
        with pytest.raises(ValueError, match="timezone-aware"):
            point_in_time_profile([], "S1", datetime(2026, 1, 1))

    def test_exact_boundary_profile(self):
        """Profile with effective_from == as_of_time is returned."""
        profiles = [
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc), "category": "B"},
        ]
        result = point_in_time_profile(profiles, "S1", datetime(2026, 6, 1, tzinfo=timezone.utc))
        assert result is not None
        assert result["category"] == "B"

    def test_skips_missing_effective_from(self):
        """Profiles without effective_from are skipped."""
        profiles = [
            {"seller_id": "S1", "category": "A"},
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc), "category": "B"},
        ]
        result = point_in_time_profile(profiles, "S1", datetime(2026, 9, 1, tzinfo=timezone.utc))
        assert result is not None
        assert result["category"] == "B"

    def test_tz_aware_effective_from(self):
        """effective_from with timezone offset is handled correctly."""
        profiles = [
            {"seller_id": "S1", "effective_from": datetime(2026, 1, 1, tzinfo=timezone.utc), "category": "A"},
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc), "category": "B"},
        ]
        result = point_in_time_profile(profiles, "S1", datetime(2026, 3, 1, tzinfo=timezone.utc))
        assert result["category"] == "A"

    def test_naive_effective_from_treated_as_utc(self):
        """Naive effective_from is treated as UTC."""
        profiles = [
            {"seller_id": "S1", "effective_from": datetime(2026, 1, 1), "category": "A"},
        ]
        result = point_in_time_profile(profiles, "S1", datetime(2026, 3, 1, tzinfo=timezone.utc))
        assert result is not None
        assert result["category"] == "A"

    def test_three_version_profiles(self):
        """Correct version selected from three profile versions."""
        profiles = [
            {"seller_id": "S1", "effective_from": datetime(2026, 1, 1, tzinfo=timezone.utc), "version": 1},
            {"seller_id": "S1", "effective_from": datetime(2026, 4, 1, tzinfo=timezone.utc), "version": 2},
            {"seller_id": "S1", "effective_from": datetime(2026, 7, 1, tzinfo=timezone.utc), "version": 3},
        ]
        r1 = point_in_time_profile(profiles, "S1", datetime(2026, 2, 1, tzinfo=timezone.utc))
        assert r1["version"] == 1
        r2 = point_in_time_profile(profiles, "S1", datetime(2026, 5, 1, tzinfo=timezone.utc))
        assert r2["version"] == 2
        r3 = point_in_time_profile(profiles, "S1", datetime(2026, 8, 1, tzinfo=timezone.utc))
        assert r3["version"] == 3


# ── validate_split_integrity tests ────────────────────────────────────────────


class TestValidateSplitIntegrity:
    """SRS TRAIN-06: Split integrity validation."""

    def test_no_duplicate_attempts(self, chronological_records):
        """Valid splits have no duplicate attempts."""
        splits = create_chronological_splits(chronological_records)
        report = validate_split_integrity(splits)
        assert report["no_duplicate_attempts"] is True

    def test_all_splits_valid(self, chronological_records):
        """All split keys present and are lists."""
        splits = create_chronological_splits(chronological_records)
        report = validate_split_integrity(splits)
        assert report["all_splits_valid"] is True

    def test_invalid_structure(self):
        """Invalid split structure is detected."""
        invalid_splits = {"prior_fit": [], "warmup": "not_a_list", "validation": [], "test": []}
        report = validate_split_integrity(invalid_splits)
        assert report["all_splits_valid"] is False

    def test_missing_split_key(self):
        """Missing split key is detected."""
        invalid_splits = {"prior_fit": [], "warmup": []}
        report = validate_split_integrity(invalid_splits)
        assert report["all_splits_valid"] is False

    def test_no_overlapping_time_ranges(self, chronological_records):
        """Valid splits have no overlapping time ranges."""
        splits = create_chronological_splits(chronological_records)
        report = validate_split_integrity(splits)
        assert report["no_overlapping_time_ranges"] is True

    def test_duplicate_attempts_detected(self):
        """Duplicate attempts across splits are detected."""
        record = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A_DUP",
            "source": "test", "finalized_at": datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc),
            "call_start_time": datetime(2026, 5, 15, 9, 0, 0, tzinfo=timezone.utc),
            "call_end_time": datetime(2026, 5, 15, 9, 10, 0, tzinfo=timezone.utc),
            "lead_sent_time": datetime(2026, 5, 15, 8, 55, 0, tzinfo=timezone.utc),
        }
        invalid_splits = {
            "prior_fit": [record],
            "warmup": [record],
            "validation": [],
            "test": [],
        }
        report = validate_split_integrity(invalid_splits)
        assert report["no_duplicate_attempts"] is False

    def test_empty_splits_valid(self):
        """Empty splits pass validation."""
        empty_splits = {
            "prior_fit": [],
            "warmup": [],
            "validation": [],
            "test": [],
        }
        report = validate_split_integrity(empty_splits)
        assert report["all_splits_valid"] is True
        assert report["no_duplicate_attempts"] is True


# ── normalize_batch tests ─────────────────────────────────────────────────────


class TestNormalizeBatch:
    """Batch normalization with quarantine."""

    def test_normalizes_valid_records(self, valid_raw_record, sample_sellers):
        """Valid records are normalized."""
        result = normalize_batch([valid_raw_record], sample_sellers)
        assert len(result) == 1
        assert result[0]["disposition"] == "MEETING_FIXED"

    def test_quarantines_invalid_records(self, sample_sellers, valid_raw_record):
        """Invalid records are quarantined."""
        bad_raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "Not Answered",
            "lead_call_status": "Answered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        quarantine = []
        result = normalize_batch([valid_raw_record, bad_raw], sample_sellers, quarantine=quarantine)
        assert len(result) == 1
        assert len(quarantine) == 1

    def test_no_quarantine_list(self, sample_sellers, valid_raw_record):
        """Invalid records are silently skipped when no quarantine list provided."""
        bad_raw = {
            "seller_id": "S001", "lead_id": "L001", "attempt_id": "A001",
            "source": "test", "disposition_label": "Not Answered",
            "lead_call_status": "Answered",
            "finalized_at": "2026-05-15T10:30:00+05:30",
            "call_start_time": "2026-05-15T10:00:00+05:30",
            "call_end_time": "2026-05-15T10:10:00+05:30",
            "lead_sent_time": "2026-05-15T09:55:00+05:30",
            "meeting_fixed": 0,
        }
        result = normalize_batch([valid_raw_record, bad_raw], sample_sellers)
        assert len(result) == 1

    def test_all_quarantined(self, sample_sellers):
        """All records quarantined when all are invalid."""
        bad_raws = [
            {
                "seller_id": "S001", "lead_id": "L001", "attempt_id": f"A{i}",
                "source": "test", "disposition_label": "Not Answered",
                "lead_call_status": "Answered",
                "finalized_at": "2026-05-15T10:30:00+05:30",
                "call_start_time": "2026-05-15T10:00:00+05:30",
                "call_end_time": "2026-05-15T10:10:00+05:30",
                "lead_sent_time": "2026-05-15T09:55:00+05:30",
                "meeting_fixed": 0,
            }
            for i in range(3)
        ]
        quarantine = []
        result = normalize_batch(bad_raws, sample_sellers, quarantine=quarantine)
        assert len(result) == 0
        assert len(quarantine) == 3


# ── build_seller_profiles tests ───────────────────────────────────────────────


class TestBuildSellerProfiles:
    """Seller profile index building."""

    def test_groups_by_seller_id(self):
        """Profiles are grouped by seller_id."""
        raw = [
            {"seller_id": "S1", "effective_from": datetime(2026, 1, 1, tzinfo=timezone.utc)},
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc)},
            {"seller_id": "S2", "effective_from": datetime(2026, 3, 1, tzinfo=timezone.utc)},
        ]
        profiles = build_seller_profiles(raw)
        assert len(profiles["S1"]) == 2
        assert len(profiles["S2"]) == 1

    def test_sorts_by_effective_from(self):
        """Profiles are sorted by effective_from ascending."""
        raw = [
            {"seller_id": "S1", "effective_from": datetime(2026, 6, 1, tzinfo=timezone.utc)},
            {"seller_id": "S1", "effective_from": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        ]
        profiles = build_seller_profiles(raw)
        assert profiles["S1"][0]["effective_from"].month == 1
        assert profiles["S1"][1]["effective_from"].month == 6


# ── isolate_prior_fit_data tests ──────────────────────────────────────────────


class TestIsolatePriorFitData:
    """TRAIN-06: Prior-fit data isolation."""

    def test_excludes_prior_fit_from_seller_state(self, chronological_records):
        """Prior-fit data is excluded from seller state data."""
        splits = create_chronological_splits(chronological_records)
        prior_fit, seller_state = isolate_prior_fit_data(
            splits, {}, splits["warmup"] + splits["validation"] + splits["test"]
        )
        assert len(prior_fit) == 10
        assert len(seller_state) == 10

    def test_prior_fit_separate(self, chronological_records):
        """Prior-fit data is returned separately."""
        splits = create_chronological_splits(chronological_records)
        prior_fit, seller_state = isolate_prior_fit_data(
            splits, {}, splits["warmup"] + splits["validation"] + splits["test"]
        )
        prior_ids = {r["attempt_id"] for r in prior_fit}
        state_ids = {r["attempt_id"] for r in seller_state}
        assert prior_ids.isdisjoint(state_ids)

    def test_seller_state_includes_all_non_prior(self):
        """Seller state includes warmup + validation + test."""
        splits = {
            "prior_fit": [{"attempt_id": "PF1"}, {"attempt_id": "PF2"}],
            "warmup": [{"attempt_id": "W1"}],
            "validation": [{"attempt_id": "V1"}],
            "test": [{"attempt_id": "T1"}],
        }
        prior_fit, seller_state = isolate_prior_fit_data(
            splits, {}, splits["warmup"] + splits["validation"] + splits["test"]
        )
        assert len(prior_fit) == 2
        assert len(seller_state) == 3
        state_ids = {r["attempt_id"] for r in seller_state}
        assert state_ids == {"W1", "V1", "T1"}


# ── _bin_key edge case tests ──────────────────────────────────────────────────


class TestBinKeyHelpers:
    """Edge cases for _bin_key and _parse_bin_key."""

    def test_bin_key_minute_0(self):
        """Minute 0 -> 00 bin."""
        dt = datetime(2026, 5, 15, 10, 0, 30, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:00"

    def test_bin_key_minute_14(self):
        """Minute 14 -> 00 bin."""
        dt = datetime(2026, 5, 15, 10, 14, 59, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:00"

    def test_bin_key_minute_15(self):
        """Minute 15 -> 15 bin."""
        dt = datetime(2026, 5, 15, 10, 15, 0, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:15"

    def test_bin_key_minute_29(self):
        """Minute 29 -> 15 bin."""
        dt = datetime(2026, 5, 15, 10, 29, 59, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:15"

    def test_bin_key_minute_30(self):
        """Minute 30 -> 30 bin."""
        dt = datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:30"

    def test_bin_key_minute_44(self):
        """Minute 44 -> 30 bin."""
        dt = datetime(2026, 5, 15, 10, 44, 59, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:30"

    def test_bin_key_minute_45(self):
        """Minute 45 -> 45 bin."""
        dt = datetime(2026, 5, 15, 10, 45, 0, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:45"

    def test_bin_key_minute_59(self):
        """Minute 59 -> 45 bin."""
        dt = datetime(2026, 5, 15, 10, 59, 59, tzinfo=timezone.utc)
        assert _bin_key(dt) == "2026-05-15T10:45"

    def test_parse_bin_key_format(self):
        """_parse_bin_key parses correct format."""
        dt = _parse_bin_key("2026-05-15T10:30")
        assert dt == datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc)

    def test_bin_key_different_days(self):
        """Bin keys differ by day."""
        dt1 = datetime(2026, 5, 15, 10, 7, 0, tzinfo=timezone.utc)
        dt2 = datetime(2026, 5, 16, 10, 7, 0, tzinfo=timezone.utc)
        assert _bin_key(dt1) != _bin_key(dt2)
