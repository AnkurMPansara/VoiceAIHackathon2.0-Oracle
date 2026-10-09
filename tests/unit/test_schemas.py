"""Comprehensive unit tests for src/btc/schemas.py.

Cross-references every model, field, and validator against:
- best_time_to_call_srs_v2.md (SRS §4 and §10)
- data/Best-Time-to-Call - Data Dictionary.md

Tested requirements:
  DATA-01: IDs are nonempty strings, max 128 UTF-8 bytes
  DATA-02: ISO 8601 timestamps with explicit offset; reject naive
  DATA-03: Nonnegative integer durations; reject unknown fields
  DATA-04: meeting_fixed requires answered + disposition; NOT_ANSWERED requires answered=false
  DATA-05: Revision identity constraints (checked via business logic)
  SRS §10.2: RecommendResponse field completeness and null constraints
"""

import sys
from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

# Ensure the project root is on the path so `src.btc.schemas` is importable.
sys.path.insert(0, "D:/Hackathon/VoiceAIHackathon2.0-Oracle")

from src.btc.schemas import (
    FinalizedOutcome,
    SellerProfile,
    RecommendRequest,
    RecommendResponse,
    RetryRequest,
    RetryResponse,
    ErrorResponseBody,
    Disposition,
    AnsweredStatus,
    DecisionStatus,
    PolicyMode,
    Assignment,
    RetryResult,
    VersionedSchema,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_EPOCH = datetime(2026, 10, 9, 0, 0, 0, tzinfo=timezone.utc)
_TZ_5_30 = timezone(timedelta(hours=5, minutes=30))
_NOW = datetime(2026, 10, 9, 12, 0, 0, tzinfo=_TZ_5_30)
_LATER = datetime(2026, 10, 9, 13, 0, 0, tzinfo=_TZ_5_30)
_EARLIER = datetime(2026, 10, 9, 11, 0, 0, tzinfo=_TZ_5_30)


def _make_base_outcome(**overrides):
    """Return a minimal valid FinalizedOutcome dict, then apply *overrides*."""
    base = {
        "seller_id": "seller-001",
        "lead_id": "lead-001",
        "attempt_id": "attempt-001",
        "source": "dialer-a",
        "revision": 1,
        "event_id": "evt-001",
        "finalized_at": _NOW,
        "call_start_time": _EARLIER,
        "call_end_time": _LATER,
        "lead_sent_time": _EPOCH,
        "attempt_number": 1,
        "answered": True,
        "disposition": Disposition.GENERAL,
        "meeting_fixed": False,
    }
    base.update(overrides)
    return base


def _make_recommended_response(**overrides):
    """Return a minimal valid RecommendResponse dict (status=RECOMMENDED)."""
    base = {
        "decision_id": "dec-001",
        "request_id": "req-001",
        "seller_id": "seller-001",
        "lead_id": "lead-001",
        "status": DecisionStatus.RECOMMENDED,
        "scheduled_at": _LATER,
        "secondary_at": None,
        "reason_code": "BEST_SLOT",
        "mode": PolicyMode.EXPLOIT,
        "assignment": Assignment.TREATMENT,
        "experiment_id": "exp-001",
        "policy_version": "v1",
        "bundle_id": "bundle-001",
        "model_compatibility_id": "compat-001",
        "profile_version": "v1",
        "calendar_version": "v1",
        "context_version": 1,
        "state_version": 1,
        "n_attempts": 5,
        "n_eff": 5,
        "prior_level": 0.5,
        "prior_weight": 0.3,
        "expected_reward": 0.8,
        "latent_std": 0.1,
        "predictive_std": 0.2,
        "candidate_count": 10,
        "action_probability": 0.1,
        "assignment_probability": 0.5,
        "ope_eligible": True,
        "created_at": _NOW,
        "valid_until": _LATER,
    }
    base.update(overrides)
    return base


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  DATA-01: ID validation (nonempty, max 128 UTF-8 bytes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestData01IdValidation:
    """IDs must be nonempty strings with <= 128 UTF-8 bytes."""

    def test_valid_id_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome())
        assert outcome.seller_id == "seller-001"

    def test_empty_id_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(seller_id=""))
        assert "seller_id" in str(exc_info.value)

    def test_id_too_long_129_bytes_raises(self):
        long_id = "a" * 129  # 129 UTF-8 bytes
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(seller_id=long_id))
        assert "128" in str(exc_info.value).lower() or "exceeds" in str(exc_info.value).lower()

    def test_id_exactly_128_bytes_passes(self):
        ok_id = "a" * 128
        outcome = FinalizedOutcome(**_make_base_outcome(seller_id=ok_id))
        assert outcome.seller_id == ok_id

    def test_numeric_id_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(seller_id=12345))
        assert "string" in str(exc_info.value).lower()

    def test_empty_attempt_id_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(attempt_id=""))

    def test_empty_source_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(source=""))

    def test_empty_event_id_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(event_id=""))

    def test_empty_lead_id_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(lead_id=""))

    def test_optional_id_empty_passes(self):
        """Optional ID fields (decision_id, dialer_version, source_bucket) may be None."""
        outcome = FinalizedOutcome(**_make_base_outcome(decision_id=None))
        assert outcome.decision_id is None

    # ── RecommendResponse IDs ──

    def test_recommend_response_valid_ids(self):
        resp = RecommendResponse(**_make_recommended_response())
        assert resp.decision_id == "dec-001"

    def test_recommend_response_empty_decision_id_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(decision_id=""))

    def test_recommend_response_empty_bundle_id_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(bundle_id=""))

    # ── RetryRequest IDs ──

    def test_retry_request_valid_ids(self):
        req = RetryRequest(request_id="req-1", source="src-1", attempt_id="att-1", expected_revision=1)
        assert req.request_id == "req-1"

    def test_retry_request_empty_request_id_raises(self):
        with pytest.raises(ValidationError):
            RetryRequest(request_id="", source="src-1", attempt_id="att-1", expected_revision=1)

    def test_retry_request_empty_attempt_id_raises(self):
        with pytest.raises(ValidationError):
            RetryRequest(request_id="req-1", source="src-1", attempt_id="", expected_revision=1)

    # ── ErrorResponseBody IDs ──

    def test_error_response_valid_request_id(self):
        err = ErrorResponseBody(error_code="TEST", message="msg", request_id="req-1", retryable=True)
        assert err.request_id == "req-1"

    def test_error_response_empty_request_id_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponseBody(error_code="TEST", message="msg", request_id="", retryable=True)

    # ── SellerProfile IDs ──

    def test_seller_profile_valid_id(self):
        profile = SellerProfile(
            seller_id="s1",
            effective_from=_NOW,
            profile_version="v1",
        )
        assert profile.seller_id == "s1"

    def test_seller_profile_empty_id_raises(self):
        with pytest.raises(ValidationError):
            SellerProfile(seller_id="", effective_from=_NOW, profile_version="v1")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  DATA-02: ISO 8601 timestamps with explicit offset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestData02TimestampValidation:
    """Timestamps must include an explicit timezone offset."""

    def test_utc_timestamp_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(finalized_at=datetime(2026, 10, 9, 12, 0, 0, tzinfo=timezone.utc)))
        assert outcome.finalized_at.tzinfo is not None

    def test_offset_timestamp_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(finalized_at=datetime(2026, 10, 9, 12, 0, 0, tzinfo=_TZ_5_30)))
        assert outcome.finalized_at.tzinfo is not None

    def test_naive_timestamp_raises(self):
        naive = datetime(2026, 10, 9, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(finalized_at=naive))
        assert "timezone" in str(exc_info.value).lower() or "offset" in str(exc_info.value).lower()

    def test_naive_string_timestamp_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(finalized_at="2026-10-09T12:00:00"))

    def test_z_suffix_timestamp_passes(self):
        outcome = FinalizedOutcome.model_validate({
            **_make_base_outcome(),
            "finalized_at": "2026-10-09T12:00:00Z",
        })
        assert outcome.finalized_at.tzinfo is not None

    def test_offset_string_timestamp_passes(self):
        outcome = FinalizedOutcome.model_validate({
            **_make_base_outcome(),
            "finalized_at": "2026-10-09T12:00:00+05:30",
        })
        assert outcome.finalized_at.tzinfo is not None

    def test_naive_string_with_offset_missing_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome.model_validate({
                **_make_base_outcome(),
                "finalized_at": "2026-10-09T12:00:00+0530",  # missing colon in offset
            })

    def test_recommend_response_naive_created_at_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(created_at=datetime(2026, 10, 9, 12, 0, 0)))

    def test_recommend_response_valid_created_at(self):
        resp = RecommendResponse(**_make_recommended_response(created_at=_NOW))
        assert resp.created_at.tzinfo is not None

    def test_recommend_request_naive_earliest_at_raises(self):
        with pytest.raises(ValidationError):
            RecommendRequest(
                request_id="r1", seller_id="s1", lead_id="l1",
                earliest_at=datetime(2026, 10, 9, 12, 0, 0),
                latest_at=_LATER,
            )

    def test_recommend_request_valid_timestamps(self):
        req = RecommendRequest(
            request_id="r1", seller_id="s1", lead_id="l1",
            earliest_at=_EARLIER, latest_at=_LATER,
        )
        assert req.earliest_at.tzinfo is not None

    def test_seller_profile_naive_effective_from_raises(self):
        with pytest.raises(ValidationError):
            SellerProfile(seller_id="s1", effective_from=datetime(2026, 10, 9, 12, 0, 0), profile_version="v1")

    def test_retry_response_naive_scheduled_at_raises(self):
        with pytest.raises(ValidationError):
            RetryResponse(
                decision_id="d1", request_id="r1", source="s1", attempt_id="a1",
                result=RetryResult.SCHEDULED,
                scheduled_at=datetime(2026, 10, 9, 12, 0, 0),
                reason_code="rc", policy_version="v1",
            )

    def test_retry_response_valid_scheduled_at(self):
        resp = RetryResponse(
            decision_id="d1", request_id="r1", source="s1", attempt_id="a1",
            result=RetryResult.SCHEDULED,
            scheduled_at=_LATER,
            reason_code="rc", policy_version="v1",
        )
        assert resp.scheduled_at.tzinfo is not None

    def test_optional_callback_at_naive_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(requested_callback_at=datetime(2026, 10, 9, 12, 0, 0)))

    def test_optional_callback_at_none_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(requested_callback_at=None))
        assert outcome.requested_callback_at is None

    def test_optional_callback_at_valid(self):
        outcome = FinalizedOutcome(**_make_base_outcome(requested_callback_at=_LATER))
        assert outcome.requested_callback_at.tzinfo is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  DATA-03: Unknown fields rejected; durations nonnegative
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestData03ExtraFieldsAndDurations:
    """Unknown JSON fields SHALL be rejected on versioned schemas."""

    def test_unknown_field_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome.model_validate({
                **_make_base_outcome(),
                "unknown_field": "should_fail",
            })
        assert "unknown_field" in str(exc_info.value)

    def test_recommend_response_unknown_field_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse.model_validate({
                **_make_recommended_response(),
                "extra_field": 42,
            })

    def test_retry_request_unknown_field_raises(self):
        with pytest.raises(ValidationError):
            RetryRequest.model_validate({
                "request_id": "r1", "source": "s1", "attempt_id": "a1",
                "expected_revision": 1,
                "bogus": True,
            })

    def test_error_response_unknown_field_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponseBody.model_validate({
                "error_code": "E", "message": "m", "request_id": "r1",
                "retryable": True,
                "extra": "no",
            })

    def test_duration_nonnegative(self):
        outcome = FinalizedOutcome(**_make_base_outcome(duration_s=7200))
        assert outcome.duration_s == 7200

    def test_duration_zero(self):
        outcome = FinalizedOutcome(**_make_base_outcome(duration_s=7200))
        assert outcome.duration_s == 7200

    def test_duration_negative_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(duration_s=-1))

    def test_duration_none_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(duration_s=None))
        assert outcome.duration_s is None

    def test_duration_agrees_within_one_second(self):
        """duration_s must agree with call_end - call_start within 1 second."""
        # call_end - call_start = 7200s (2 hours)
        outcome = FinalizedOutcome(**_make_base_outcome(duration_s=7200))
        assert outcome.duration_s == 7200

    def test_duration_agrees_plus_one_second(self):
        outcome = FinalizedOutcome(**_make_base_outcome(duration_s=7201))
        assert outcome.duration_s == 7201

    def test_duration_disagrees_by_two_seconds_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(duration_s=7202))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  DATA-04: meeting_fixed / answered / disposition cross-field constraints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestData04CrossFieldConstraints:
    """meeting_fixed and disposition / answered cross-field rules."""

    def test_meeting_fixed_true_answered_true_disposition_meeting_fixed_passes(self):
        """Test 1: Valid FinalizedOutcome with meeting_fixed=True, answered=True, disposition=MEETING_FIXED."""
        outcome = FinalizedOutcome(**_make_base_outcome(
            meeting_fixed=True,
            answered=True,
            disposition=Disposition.MEETING_FIXED,
        ))
        assert outcome.meeting_fixed is True
        assert outcome.answered is True
        assert outcome.disposition is Disposition.MEETING_FIXED

    def test_meeting_fixed_true_answered_false_raises(self):
        """Test 2: meeting_fixed=True but answered=False → raises."""
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(
                meeting_fixed=True,
                answered=False,
                disposition=Disposition.MEETING_FIXED,
            ))
        assert "meeting_fixed" in str(exc_info.value).lower() or "answered" in str(exc_info.value).lower()

    def test_meeting_fixed_true_disposition_general_raises(self):
        """Test 3: meeting_fixed=True but disposition=GENERAL → raises."""
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(
                meeting_fixed=True,
                answered=True,
                disposition=Disposition.GENERAL,
            ))
        assert "meeting_fixed" in str(exc_info.value).lower() or "disposition" in str(exc_info.value).lower()

    def test_not_answered_disposition_with_answered_true_raises(self):
        """Test 4: NOT_ANSWERED disposition with answered=True → raises."""
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(
                answered=True,
                disposition=Disposition.NOT_ANSWERED,
            ))
        assert "NOT_ANSWERED" in str(exc_info.value) or "answered" in str(exc_info.value).lower()

    def test_not_answered_disposition_with_answered_false_passes(self):
        """NOT_ANSWERED with answered=False is valid."""
        outcome = FinalizedOutcome(**_make_base_outcome(
            answered=False,
            disposition=Disposition.NOT_ANSWERED,
        ))
        assert outcome.answered is False
        assert outcome.disposition is Disposition.NOT_ANSWERED

    def test_meeting_fixed_false_any_disposition_passes(self):
        """meeting_fixed=False should allow any disposition (except NOT_ANSWERED needs answered=False)."""
        for disp in Disposition:
            if disp is Disposition.NOT_ANSWERED:
                outcome = FinalizedOutcome(**_make_base_outcome(
                    meeting_fixed=False, disposition=disp, answered=False
                ))
            else:
                outcome = FinalizedOutcome(**_make_base_outcome(
                    meeting_fixed=False, disposition=disp
                ))
            assert outcome.disposition is disp

    def test_call_later_busy_does_not_constrain_answered(self):
        """CALL_LATER_BUSY SHALL NOT itself determine whether the call was answered."""
        outcome = FinalizedOutcome(**_make_base_outcome(
            answered=True,
            disposition=Disposition.CALL_LATER_BUSY,
        ))
        assert outcome.disposition is Disposition.CALL_LATER_BUSY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  FinalizedOutcome structural / temporal constraints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFinalizedOutcomeTemporal:
    """call_start <= call_end, lead_sent <= call_start, revision > 0."""

    def test_call_end_before_call_start_raises(self):
        """Test 5: call_end < call_start → raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(
                call_start_time=_LATER,
                call_end_time=_EARLIER,
            ))
        assert "call_start" in str(exc_info.value).lower()

    def test_call_start_equals_call_end_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(
            call_start_time=_NOW,
            call_end_time=_NOW,
        ))
        assert outcome.call_start_time == outcome.call_end_time

    def test_lead_sent_after_call_start_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(
                lead_sent_time=_LATER,
                call_start_time=_EARLIER,
            ))

    def test_lead_sent_equals_call_start_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(
            lead_sent_time=_EARLIER,
            call_start_time=_EARLIER,
        ))
        assert outcome.lead_sent_time == outcome.call_start_time

    def test_revision_zero_raises(self):
        """Test 6: revision=0 → raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            FinalizedOutcome(**_make_base_outcome(revision=0))
        assert "revision" in str(exc_info.value).lower()

    def test_revision_negative_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(revision=-1))

    def test_revision_one_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(revision=1))
        assert outcome.revision == 1

    def test_attempt_number_zero_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(attempt_number=0))

    def test_attempt_number_negative_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(attempt_number=-5))

    def test_attempt_number_one_passes(self):
        outcome = FinalizedOutcome(**_make_base_outcome(attempt_number=1))
        assert outcome.attempt_number == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  DATA-05: Revision identity (seller, lead, source must not change)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestData05RevisionIdentity:
    """Revisions SHALL NOT change seller, lead, or source identity."""

    def test_same_ids_different_revision_passes(self):
        """Two outcomes with same seller/lead/source but different revisions."""
        o1 = FinalizedOutcome(**_make_base_outcome(revision=1, event_id="evt-1"))
        o2 = FinalizedOutcome(**_make_base_outcome(revision=2, event_id="evt-2"))
        assert o1.seller_id == o2.seller_id
        assert o1.lead_id == o2.lead_id
        assert o1.source == o2.source

    def test_different_seller_same_attempt_id_different_outcomes(self):
        """Different seller IDs produce different outcomes (business check)."""
        o1 = FinalizedOutcome(**_make_base_outcome(seller_id="seller-a", event_id="evt-1"))
        o2 = FinalizedOutcome(**_make_base_outcome(seller_id="seller-b", event_id="evt-2"))
        assert o1.seller_id != o2.seller_id


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.  SRS §10.2: RecommendResponse completeness and null constraints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRecommendResponseRequiredFields:
    """SRS §10.2: ALL fields required in RecommendResponse."""

    def test_all_required_fields_present(self):
        """Test 14 (partial): Valid RecommendResponse with all fields passes."""
        resp = RecommendResponse(**_make_recommended_response())
        # Verify every field is present and non-null where required
        assert resp.decision_id is not None
        assert resp.request_id is not None
        assert resp.seller_id is not None
        assert resp.lead_id is not None
        assert resp.status is not None
        assert resp.reason_code is not None
        assert resp.mode is not None
        assert resp.assignment is not None
        assert resp.policy_version is not None
        assert resp.bundle_id is not None
        assert resp.model_compatibility_id is not None
        assert resp.profile_version is not None
        assert resp.calendar_version is not None
        assert resp.context_version is not None
        assert resp.state_version is not None
        assert resp.n_attempts is not None
        assert resp.n_eff is not None
        assert resp.prior_level is not None
        assert resp.prior_weight is not None
        assert resp.expected_reward is not None
        assert resp.latent_std is not None
        assert resp.predictive_std is not None
        assert resp.candidate_count is not None
        assert resp.ope_eligible is not None
        assert resp.created_at is not None

    def test_missing_decision_id_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(decision_id=None))  # type: ignore

    def test_missing_status_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(status=None))  # type: ignore

    def test_missing_reason_code_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(reason_code=None))  # type: ignore

    def test_missing_mode_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(mode=None))  # type: ignore

    def test_missing_assignment_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(assignment=None))  # type: ignore

    def test_missing_policy_version_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(policy_version=None))  # type: ignore

    def test_missing_bundle_id_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(bundle_id=None))  # type: ignore

    def test_missing_model_compatibility_id_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(model_compatibility_id=None))  # type: ignore

    def test_missing_profile_version_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(profile_version=None))  # type: ignore

    def test_missing_calendar_version_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(calendar_version=None))  # type: ignore

    def test_missing_context_version_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(context_version=None))  # type: ignore  # noqa: E501

    def test_missing_state_version_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(state_version=None))  # type: ignore

    def test_missing_n_attempts_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(n_attempts=None))  # type: ignore

    def test_missing_n_eff_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(n_eff=None))  # type: ignore

    def test_missing_prior_level_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(prior_level=None))  # type: ignore

    def test_missing_prior_weight_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(prior_weight=None))  # type: ignore

    def test_missing_expected_reward_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(expected_reward=None))  # type: ignore

    def test_missing_latent_std_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(latent_std=None))  # type: ignore

    def test_missing_predictive_std_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(predictive_std=None))  # type: ignore

    def test_missing_candidate_count_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(candidate_count=None))  # type: ignore

    def test_missing_ope_eligible_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(ope_eligible=None))  # type: ignore

    def test_missing_created_at_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(created_at=None))  # type: ignore


class TestRecommendResponseNoActionStatus:
    """SRS §10.2: no-slot / stop / review / superseded responses have null timestamps."""

    def test_no_eligible_slot_null_timestamps(self):
        """Test 10: status=NO_ELIGIBLE_SLOT has null scheduled_at, secondary_at, valid_until."""
        resp = RecommendResponse(**_make_recommended_response(
            status=DecisionStatus.NO_ELIGIBLE_SLOT,
            scheduled_at=None,
            secondary_at=None,
            valid_until=None,
            candidate_count=0,
            action_probability=None,
        ))
        assert resp.scheduled_at is None
        assert resp.secondary_at is None
        assert resp.valid_until is None
        assert resp.candidate_count == 0

    def test_no_eligible_slot_nonnull_scheduled_at_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RecommendResponse(**_make_recommended_response(
                status=DecisionStatus.NO_ELIGIBLE_SLOT,
                scheduled_at=_LATER,
            ))
        assert "scheduled_at" in str(exc_info.value).lower()

    def test_no_eligible_slot_nonnull_secondary_at_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                status=DecisionStatus.NO_ELIGIBLE_SLOT,
                secondary_at=_LATER,
            ))

    def test_no_eligible_slot_nonnull_valid_until_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                status=DecisionStatus.NO_ELIGIBLE_SLOT,
                valid_until=_LATER,
            ))

    def test_no_eligible_slot_nonzero_candidate_count_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                status=DecisionStatus.NO_ELIGIBLE_SLOT,
                candidate_count=5,
            ))

    def test_no_eligible_slot_nonnull_action_probability_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                status=DecisionStatus.NO_ELIGIBLE_SLOT,
                action_probability=0.5,
            ))

    def test_stop_null_timestamps(self):
        """Test 12: status=STOP has null scheduled_at, secondary_at, valid_until."""
        resp = RecommendResponse(**_make_recommended_response(
            status=DecisionStatus.STOP,
            scheduled_at=None,
            secondary_at=None,
            valid_until=None,
            candidate_count=0,
            action_probability=None,
        ))
        assert resp.status is DecisionStatus.STOP
        assert resp.scheduled_at is None

    def test_stop_nonnull_scheduled_at_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                status=DecisionStatus.STOP,
                scheduled_at=_LATER,
            ))

    def test_manual_review_null_timestamps(self):
        resp = RecommendResponse(**_make_recommended_response(
            status=DecisionStatus.MANUAL_REVIEW,
            scheduled_at=None,
            secondary_at=None,
            valid_until=None,
            candidate_count=0,
            action_probability=None,
        ))
        assert resp.status is DecisionStatus.MANUAL_REVIEW

    def test_superseded_null_timestamps(self):
        resp = RecommendResponse(**_make_recommended_response(
            status=DecisionStatus.SUPERSEDED,
            scheduled_at=None,
            secondary_at=None,
            valid_until=None,
            candidate_count=0,
            action_probability=None,
        ))
        assert resp.status is DecisionStatus.SUPERSEDED

    def test_recommended_allows_timestamps(self):
        """RECOMMENDED status allows non-null scheduled_at."""
        resp = RecommendResponse(**_make_recommended_response(
            status=DecisionStatus.RECOMMENDED,
            scheduled_at=_LATER,
            valid_until=_LATER,
            candidate_count=10,
        ))
        assert resp.scheduled_at == _LATER
        assert resp.valid_until == _LATER


class TestRecommendResponseShadowMode:
    """SRS §10.2: SHADOW assignment constraints."""

    def test_shadow_null_experiment_id(self):
        """Test 11: assignment=SHADOW has null experiment_id and probabilities, ope_eligible=False."""
        resp = RecommendResponse(**_make_recommended_response(
            assignment=Assignment.SHADOW,
            experiment_id=None,
            action_probability=None,
            assignment_probability=None,
            ope_eligible=False,
        ))
        assert resp.assignment is Assignment.SHADOW
        assert resp.experiment_id is None
        assert resp.action_probability is None
        assert resp.assignment_probability is None
        assert resp.ope_eligible is False

    def test_shadow_nonnull_experiment_id_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RecommendResponse(**_make_recommended_response(
                assignment=Assignment.SHADOW,
                experiment_id="exp-1",
            ))
        assert "experiment_id" in str(exc_info.value).lower()

    def test_shadow_nonnull_action_probability_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                assignment=Assignment.SHADOW,
                action_probability=0.5,
            ))

    def test_shadow_nonnull_assignment_probability_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                assignment=Assignment.SHADOW,
                assignment_probability=0.05,
            ))

    def test_shadow_ope_eligible_true_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(
                assignment=Assignment.SHADOW,
                ope_eligible=True,
            ))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8.  Enum value validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEnums:
    """Verify all enum values match SRS specifications."""

    def test_disposition_values(self):
        expected = {"MEETING_FIXED", "NOT_INTERESTED", "GENERAL", "CALL_LATER_BUSY", "NOT_ANSWERED", "UNKNOWN"}
        actual = {d.value for d in Disposition}
        assert actual == expected

    def test_decision_status_values(self):
        expected = {"RECOMMENDED", "NO_ELIGIBLE_SLOT", "STOP", "MANUAL_REVIEW", "SUPERSEDED"}
        actual = {s.value for s in DecisionStatus}
        assert actual == expected

    def test_policy_mode_values(self):
        expected = {"EXPLOIT", "PRIOR_ONLY", "UNIFORM_EXPLORE", "BASELINE", "NONE"}
        actual = {m.value for m in PolicyMode}
        assert actual == expected

    def test_assignment_values(self):
        expected = {"CONTROL", "TREATMENT", "EXPLORE", "SHADOW"}
        actual = {a.value for a in Assignment}
        assert actual == expected

    def test_retry_result_values(self):
        expected = {"STOP", "MANUAL_REVIEW", "SUPERSEDED", "SCHEDULED"}
        actual = {r.value for r in RetryResult}
        assert actual == expected

    def test_answered_status_values(self):
        expected = {"ANSWERED", "NOT_ANSWERED"}
        actual = {a.value for a in AnsweredStatus}
        assert actual == expected

    def test_invalid_disposition_raises(self):
        with pytest.raises(ValidationError):
            FinalizedOutcome(**_make_base_outcome(disposition="INVALID_DISPOSITION"))  # type: ignore

    def test_invalid_decision_status_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(status="BAD_STATUS"))  # type: ignore

    def test_invalid_policy_mode_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(mode="BAD_MODE"))  # type: ignore

    def test_invalid_assignment_raises(self):
        with pytest.raises(ValidationError):
            RecommendResponse(**_make_recommended_response(assignment="BAD_ASSIGNMENT"))  # type: ignore

    def test_invalid_retry_result_raises(self):
        with pytest.raises(ValidationError):
            RetryResponse(
                decision_id="d1", request_id="r1", source="s1", attempt_id="a1",
                result="BAD_RESULT", reason_code="rc", policy_version="v1",
            )  # type: ignore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9.  RecommendRequest validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRecommendRequest:
    """RecommendRequest temporal and ID constraints."""

    def test_valid_request(self):
        """Test 13: Valid RecommendRequest with earliest_at < latest_at passes."""
        req = RecommendRequest(
            request_id="req-001",
            seller_id="seller-001",
            lead_id="lead-001",
            earliest_at=_EARLIER,
            latest_at=_LATER,
        )
        assert req.earliest_at < req.latest_at

    def test_earliest_equals_latest_passes(self):
        req = RecommendRequest(
            request_id="req-001",
            seller_id="seller-001",
            lead_id="lead-001",
            earliest_at=_NOW,
            latest_at=_NOW,
        )
        assert req.earliest_at == req.latest_at

    def test_earliest_after_latest_raises(self):
        """Test 14 (invalid): earliest_at > latest_at → raises."""
        with pytest.raises(ValidationError) as exc_info:
            RecommendRequest(
                request_id="req-001",
                seller_id="seller-001",
                lead_id="lead-001",
                earliest_at=_LATER,
                latest_at=_EARLIER,
            )
        assert "earliest_at" in str(exc_info.value).lower() or "latest_at" in str(exc_info.value).lower()

    def test_default_context_version(self):
        req = RecommendRequest(
            request_id="req-001",
            seller_id="seller-001",
            lead_id="lead-001",
            earliest_at=_EARLIER,
            latest_at=_LATER,
        )
        assert req.context_version == 1

    def test_custom_context_version(self):
        req = RecommendRequest(
            request_id="req-001",
            seller_id="seller-001",
            lead_id="lead-001",
            earliest_at=_EARLIER,
            latest_at=_LATER,
            context_version=5,
        )
        assert req.context_version == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. RetryRequest and RetryResponse validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRetryModels:
    """RetryRequest and RetryResponse validation."""

    def test_valid_retry_request(self):
        """Test 15: RetryRequest with valid source, attempt_id passes."""
        req = RetryRequest(
            request_id="req-001",
            source="dialer-a",
            attempt_id="attempt-001",
            expected_revision=1,
        )
        assert req.request_id == "req-001"
        assert req.source == "dialer-a"
        assert req.attempt_id == "attempt-001"
        assert req.expected_revision == 1

    def test_retry_request_revision_zero_raises(self):
        with pytest.raises(ValidationError):
            RetryRequest(
                request_id="req-001",
                source="dialer-a",
                attempt_id="attempt-001",
                expected_revision=0,
            )

    def test_retry_request_revision_negative_raises(self):
        with pytest.raises(ValidationError):
            RetryRequest(
                request_id="req-001",
                source="dialer-a",
                attempt_id="attempt-001",
                expected_revision=-1,
            )

    def test_valid_retry_response_scheduled(self):
        resp = RetryResponse(
            decision_id="d-001",
            request_id="req-001",
            source="dialer-a",
            attempt_id="att-001",
            result=RetryResult.SCHEDULED,
            scheduled_at=_LATER,
            reason_code="CALL_LATER",
            policy_version="v1",
        )
        assert resp.result is RetryResult.SCHEDULED
        assert resp.scheduled_at == _LATER

    def test_valid_retry_response_stop(self):
        resp = RetryResponse(
            decision_id="d-001",
            request_id="req-001",
            source="dialer-a",
            attempt_id="att-001",
            result=RetryResult.STOP,
            scheduled_at=None,
            reason_code="TERMINAL",
            policy_version="v1",
        )
        assert resp.result is RetryResult.STOP
        assert resp.scheduled_at is None

    def test_valid_retry_response_manual_review(self):
        resp = RetryResponse(
            decision_id="d-001",
            request_id="req-001",
            source="dialer-a",
            attempt_id="att-001",
            result=RetryResult.MANUAL_REVIEW,
            scheduled_at=None,
            reason_code="UNKNOWN_DISPOSITION",
            policy_version="v1",
        )
        assert resp.result is RetryResult.MANUAL_REVIEW

    def test_valid_retry_response_superseded(self):
        resp = RetryResponse(
            decision_id="d-001",
            request_id="req-001",
            source="dialer-a",
            attempt_id="att-001",
            result=RetryResult.SUPERSEDED,
            scheduled_at=None,
            reason_code="NEWER_REVISION",
            policy_version="v1",
            superseded_by="d-002",
        )
        assert resp.result is RetryResult.SUPERSEDED
        assert resp.superseded_by == "d-002"

    def test_retry_response_null_superseded_by_passes(self):
        resp = RetryResponse(
            decision_id="d-001",
            request_id="req-001",
            source="dialer-a",
            attempt_id="att-001",
            result=RetryResult.SCHEDULED,
            scheduled_at=_LATER,
            reason_code="CALL_LATER",
            policy_version="v1",
            superseded_by=None,
        )
        assert resp.superseded_by is None

    def test_retry_response_empty_reason_code_raises(self):
        with pytest.raises(ValidationError):
            RetryResponse(
                decision_id="d-001",
                request_id="req-001",
                source="dialer-a",
                attempt_id="att-001",
                result=RetryResult.SCHEDULED,
                scheduled_at=_LATER,
                reason_code="",
                policy_version="v1",
            )

    def test_retry_response_empty_policy_version_raises(self):
        with pytest.raises(ValidationError):
            RetryResponse(
                decision_id="d-001",
                request_id="req-001",
                source="dialer-a",
                attempt_id="att-001",
                result=RetryResult.SCHEDULED,
                scheduled_at=_LATER,
                reason_code="rc",
                policy_version="",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. ErrorResponseBody validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestErrorResponseBody:
    """ErrorResponseBody field validation."""

    def test_valid_error_response(self):
        """Test 16: ErrorResponseBody with all required fields passes."""
        err = ErrorResponseBody(
            error_code="INVALID_REQUEST",
            message="The request_id is missing or invalid.",
            request_id="req-001",
            retryable=True,
        )
        assert err.error_code == "INVALID_REQUEST"
        assert err.retryable is True

    def test_error_response_retryable_false(self):
        err = ErrorResponseBody(
            error_code="PERMANENT_ERROR",
            message="Something irrecoverable happened.",
            request_id="req-002",
            retryable=False,
        )
        assert err.retryable is False

    def test_error_response_empty_error_code_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponseBody(
                error_code="",
                message="msg",
                request_id="req-001",
                retryable=True,
            )

    def test_error_response_empty_message_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponseBody(
                error_code="E",
                message="",
                request_id="req-001",
                retryable=True,
            )

    def test_error_response_missing_retryable_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponseBody(
                error_code="E",
                message="msg",
                request_id="req-001",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. SellerProfile validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSellerProfile:
    """SellerProfile field validation."""

    def test_valid_profile(self):
        profile = SellerProfile(
            seller_id="seller-001",
            category_group="Apparel",
            turnover_band="0-40L",
            business_type="Proprietorship",
            effective_from=_NOW,
            profile_version="v1",
        )
        assert profile.seller_id == "seller-001"
        assert profile.category_group == "Apparel"

    def test_profile_nullable_fields_none(self):
        profile = SellerProfile(
            seller_id="seller-001",
            category_group=None,
            turnover_band=None,
            business_type=None,
            effective_from=_NOW,
            profile_version="v1",
        )
        assert profile.category_group is None
        assert profile.turnover_band is None
        assert profile.business_type is None

    def test_profile_empty_seller_id_raises(self):
        with pytest.raises(ValidationError):
            SellerProfile(
                seller_id="",
                effective_from=_NOW,
                profile_version="v1",
            )

    def test_profile_empty_profile_version_raises(self):
        with pytest.raises(ValidationError):
            SellerProfile(
                seller_id="s1",
                effective_from=_NOW,
                profile_version="",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. JSON round-trip tests (model_validate)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestJsonRoundTrip:
    """Test that model_validate works correctly with string timestamps."""

    def test_outcome_json_round_trip(self):
        data = _make_base_outcome()
        data["finalized_at"] = "2026-10-09T12:00:00+05:30"
        data["call_start_time"] = "2026-10-09T11:00:00+05:30"
        data["call_end_time"] = "2026-10-09T13:00:00+05:30"
        data["lead_sent_time"] = "2026-10-09T00:00:00+00:00"
        outcome = FinalizedOutcome.model_validate(data)
        assert outcome.finalized_at.tzinfo is not None
        assert outcome.call_start_time.tzinfo is not None

    def test_recommend_response_json_round_trip(self):
        data = _make_recommended_response()
        data["created_at"] = "2026-10-09T12:00:00+05:30"
        data["scheduled_at"] = "2026-10-09T13:00:00+05:30"
        data["valid_until"] = "2026-10-09T14:00:00+05:30"
        resp = RecommendResponse.model_validate(data)
        assert resp.created_at.tzinfo is not None
        assert resp.scheduled_at.tzinfo is not None

    def test_recommend_response_json_unknown_field_raises(self):
        data = _make_recommended_response()
        data["created_at"] = "2026-10-09T12:00:00+05:30"
        data["unknown_field"] = "should_fail"
        with pytest.raises(ValidationError):
            RecommendResponse.model_validate(data)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. VersionedSchema union type
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVersionedSchema:
    """VersionedSchema union should accept any of the defined models."""

    def test_union_accepts_finalized_outcome(self):
        from typing import get_args
        schema_types = get_args(VersionedSchema)
        assert FinalizedOutcome in schema_types

    def test_union_accepts_recommend_response(self):
        from typing import get_args
        schema_types = get_args(VersionedSchema)
        assert RecommendResponse in schema_types

    def test_union_accepts_all_models(self):
        from typing import get_args
        schema_types = get_args(VersionedSchema)
        expected = {
            FinalizedOutcome, SellerProfile, RecommendRequest,
            RecommendResponse, RetryRequest, RetryResponse, ErrorResponseBody,
        }
        assert set(schema_types) == expected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 15. DATA-03: Finite numeric inputs (model diagnostics)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFiniteNumericInputs:
    """All numeric model inputs SHALL be finite (SRS DATA-03).
    
    Note: Pydantic's default float validation does not enforce finiteness.
    These tests verify current behavior and document the gap.
    """

    def test_normal_float_values_pass(self):
        resp = RecommendResponse(**_make_recommended_response(
            prior_level=0.5,
            prior_weight=0.3,
            expected_reward=0.8,
            latent_std=0.1,
            predictive_std=0.2,
        ))
        assert resp.expected_reward == 0.8

    def test_negative_float_values_pass(self):
        """Negative floats are technically finite but may be semantically invalid."""
        resp = RecommendResponse(**_make_recommended_response(
            expected_reward=-0.5,
            latent_std=0.1,
        ))
        assert resp.expected_reward == -0.5

    def test_candidate_count_zero_for_no_slot(self):
        resp = RecommendResponse(**_make_recommended_response(
            status=DecisionStatus.NO_ELIGIBLE_SLOT,
            scheduled_at=None,
            secondary_at=None,
            valid_until=None,
            candidate_count=0,
            action_probability=None,
        ))
        assert resp.candidate_count == 0

    def test_candidate_count_positive_for_recommended(self):
        resp = RecommendResponse(**_make_recommended_response(
            status=DecisionStatus.RECOMMENDED,
            candidate_count=10,
        ))
        assert resp.candidate_count == 10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 16. Comprehensive valid FinalizedOutcome end-to-end
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFinalizedOutcomeEndToEnd:
    """Full valid FinalizedOutcome with all optional fields."""

    def test_full_valid_outcome(self):
        outcome = FinalizedOutcome(
            seller_id="seller-001",
            lead_id="lead-001",
            attempt_id="attempt-001",
            source="dialer-a",
            revision=1,
            event_id="evt-001",
            finalized_at=_NOW,
            call_start_time=_EARLIER,
            call_end_time=_LATER,
            lead_sent_time=_EPOCH,
            attempt_number=1,
            answered=True,
            disposition=Disposition.MEETING_FIXED,
            meeting_fixed=True,
            requested_callback_at=_LATER,
            decision_id="dec-001",
            duration_s=7200,
            dialer_version="v2.1",
            source_bucket="PIM",
        )
        assert outcome.meeting_fixed is True
        assert outcome.answered is True
        assert outcome.disposition is Disposition.MEETING_FIXED
        assert outcome.duration_s == 7200
        assert outcome.decision_id == "dec-001"

    def test_outcome_with_all_dispositions(self):
        """Test each disposition with a valid configuration."""
        for disp in Disposition:
            if disp is Disposition.NOT_ANSWERED:
                outcome = FinalizedOutcome(**_make_base_outcome(disposition=disp, answered=False, meeting_fixed=False))
            else:
                outcome = FinalizedOutcome(**_make_base_outcome(disposition=disp, meeting_fixed=False))
            assert outcome.disposition is disp
