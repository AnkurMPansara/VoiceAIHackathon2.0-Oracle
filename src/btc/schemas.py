"""Pydantic data contracts for the Best Time to Call system.

Implements SRS §4 (Canonical data contracts) and §10 (Service contracts):
- Enumerations for dispositions, statuses, modes, assignments, and retry results
- Core models: FinalizedOutcome, SellerProfile, RecommendRequest,
  RecommendResponse, RetryRequest, RetryResponse, ErrorResponseBody
- ID length enforcement (DATA-01), ISO 8601 timestamp validation (DATA-02),
  unknown-field rejection on versioned schemas, and cross-field validators
  (DATA-04)

All models use ``model_config = ConfigDict(str_strip_whitespace=True,
extra='forbid')`` where appropriate to enforce strict JSON parsing.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo


# ── Enumerations ──────────────────────────────────────────────────────────────


class Disposition(str, Enum):
    """Call disposition codes (SRS §4.2, DATA-04).

    Attributes
    ----------
    MEETING_FIXED
        The seller agreed to a specific call time.
    NOT_INTERESTED
        The seller declined further contact.
    GENERAL
        Generic disposition not matching other categories.
    CALL_LATER_BUSY
        The seller requested a callback at a later time.
    NOT_ANSWERED
        The call was not answered by the seller.
    UNKNOWN
        Disposition could not be determined.
    """

    MEETING_FIXED = "MEETING_FIXED"
    NOT_INTERESTED = "NOT_INTERESTED"
    GENERAL = "GENERAL"
    CALL_LATER_BUSY = "CALL_LATER_BUSY"
    NOT_ANSWERED = "NOT_ANSWERED"
    UNKNOWN = "UNKNOWN"


class AnsweredStatus(str, Enum):
    """Whether the call was answered (SRS §4.2).

    Attributes
    ----------
    ANSWERED
        The seller picked up the call.
    NOT_ANSWERED
        The call was not answered.
    """

    ANSWERED = "ANSWERED"
    NOT_ANSWERED = "NOT_ANSWERED"


class DecisionStatus(str, Enum):
    """Decision status returned in RecommendResponse (SRS §10.2).

    Attributes
    ----------
    RECOMMENDED
        A concrete call timestamp was recommended.
    NO_ELIGIBLE_SLOT
        No eligible time slot exists within constraints.
    STOP
        Automatic retry is stopped (terminal disposition, suppression, etc.).
    MANUAL_REVIEW
        A human must review the decision (unknown disposition, stale callback).
    SUPERSEDED
        A newer revision of the same attempt has superseded this decision.
    """

    RECOMMENDED = "RECOMMENDED"
    NO_ELIGIBLE_SLOT = "NO_ELIGIBLE_SLOT"
    STOP = "STOP"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    SUPERSEDED = "SUPERSEDED"


class PolicyMode(str, Enum):
    """Policy mode used for a recommendation (SRS §10.2, POL-05/06).

    Attributes
    ----------
    EXPLOIT
        Deterministic treatment: selects the maximum expected reward.
    PRIOR_ONLY
        Cold-start sellers with no history use the segment prior.
    UNIFORM_EXPLORE
        Exploration arm: uniform sampling over eligible candidates.
    BASELINE
        Approved baseline policy (fallback).
    NONE
        No policy active (no action).
    """

    EXPLOIT = "EXPLOIT"
    PRIOR_ONLY = "PRIOR_ONLY"
    UNIFORM_EXPLORE = "UNIFORM_EXPLORE"
    BASELINE = "BASELINE"
    NONE = "NONE"


class Assignment(str, Enum):
    """Experiment assignment arm (SRS §10.2, EXP-01).

    Attributes
    ----------
    CONTROL
        Control arm (45 %).
    TREATMENT
        Treatment arm (50 %), deterministic policy.
    EXPLORE
        Exploration arm (5 %), uniform sampling.
    SHADOW
        Shadow mode: proposals logged but never dispatched.
    """

    CONTROL = "CONTROL"
    TREATMENT = "TREATMENT"
    EXPLORE = "EXPLORE"
    SHADOW = "SHADOW"


class RetryResult(str, Enum):
    """Retry decision result (SRS §8.2, RET-01).

    Attributes
    ----------
    STOP
        No automatic retry (terminal disposition, suppression, etc.).
    MANUAL_REVIEW
        Requires human review (unknown disposition, stale callback).
    SUPERSEDED
        A newer revision of the attempt has superseded this retry.
    SCHEDULED
        A retry timestamp was proposed.
    """

    STOP = "STOP"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    SUPERSEDED = "SUPERSEDED"
    SCHEDULED = "SCHEDULED"


# ── Helper validators ─────────────────────────────────────────────────────────

# Regex for ISO 8601 timestamps with an explicit offset (DATA-02).
# Accepts patterns like 2026-10-09T11:15:00+05:30, 2026-10-09T05:45:00Z,
# 2026-10-09T11:15:00.123456+05:30, etc.
_ISO_8601_OFFSET_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)


def _validate_id(value: Any, *, field_name: str = "id") -> str:
    """Validate that *value* is a string with at most 128 UTF-8 bytes (DATA-01).

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field (used in error messages).

    Returns
    -------
    str
        The validated string.

    Raises
    ------
    ValueError
        If the value is not a string or exceeds 128 UTF-8 bytes.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    encoded = value.encode("utf-8")
    if len(encoded) == 0:
        raise ValueError(f"{field_name} must be nonempty (max 128 UTF-8 bytes, DATA-01)")
    if len(encoded) > 128:
        raise ValueError(
            f"{field_name} exceeds 128 UTF-8 bytes ({len(encoded)}), DATA-01"
        )
    return value


def _validate_iso_offset(value: Any, *, field_name: str = "timestamp") -> datetime:
    """Validate and parse an ISO 8601 timestamp with explicit offset (DATA-02).

    Accepts both ISO 8601 strings (for JSON parsing) and already-parsed
    ``datetime`` objects (for programmatic construction). In both cases
    the result must be timezone-aware.

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field (used in error messages).

    Returns
    -------
    datetime
        The parsed datetime object, timezone-aware.

    Raises
    ------
    ValueError
        If the timestamp is naive (no offset) or unparseable.
    """
    # Already a datetime (programmatic construction)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{field_name} must include an explicit timezone offset (DATA-02)")
        return value

    # String input (JSON parsing)
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO 8601 string with explicit offset (DATA-02)")
    if not _ISO_8601_OFFSET_RE.match(value):
        raise ValueError(
            f"{field_name} must include an explicit timezone offset "
            f"(e.g. +05:30 or Z), got {value!r} (DATA-02)"
        )
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name} is not a valid ISO 8601 timestamp: {exc}") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{field_name} must include an explicit timezone offset (DATA-02)")
    return dt


# ── Base model mixin for versioned schemas ─────────────────────────────────────


class _StrictBase(BaseModel):
    """Base model that rejects unknown JSON fields (DATA-03).

    All versioned public schemas inherit from this class to enforce
    ``extra='forbid'``.
    """

    model_config = ConfigDict(extra="forbid")


# ── Core Models ───────────────────────────────────────────────────────────────


class FinalizedOutcome(_StrictBase):
    """Finalised attempt outcome (SRS §4.2, DATA-04, DATA-05).

    Represents a single call attempt outcome that contributes to seller
    sufficient statistics. Revisions SHALL NOT change seller, lead, or
    source identity (DATA-05).

    Attributes
    ----------
    seller_id : str
        Seller identifier (DATA-01, max 128 UTF-8 bytes).
    lead_id : str
        Lead identifier (DATA-01, max 128 UTF-8 bytes).
    attempt_id : str
        Globally unique attempt ID within the source namespace (DATA-01).
    source : str
        Producer identifier; ``(source, attempt_id)`` is the durable key (DATA-01).
    revision : int
        Positive integer, increasing for corrections to the same attempt.
    event_id : str
        Unique delivery identifier, retained for audit (DATA-01).
    finalized_at : datetime
        Timestamp when this outcome became available to consumers (DATA-02).
    call_start_time : datetime
        Actual dial start time (DATA-02, model time).
    call_end_time : datetime
        Call end time; must be >= call_start_time (DATA-02).
    lead_sent_time : datetime
        When the lead was sent; must be <= call_start_time (DATA-02).
    attempt_number : int
        Positive integer within ``(seller_id, lead_id)``; includes all dials.
    answered : bool
        Whether the call was answered (SRS §4.2).
    disposition : Disposition
        Call disposition code (DATA-04).
    meeting_fixed : bool
        Whether a meeting was fixed; requires answered=True and
        disposition=MEETING_FIXED (DATA-04).
    requested_callback_at : datetime | None
        Seller-requested callback timestamp, nullable (DATA-02).
    decision_id : str | None
        Links to the scheduling decision, nullable (DATA-01).
    duration_s : int | None
        Duration in seconds; if present, must agree with call_end - call_start
        within one second (DATA-03).
    dialer_version : str | None
        Nullable string for diagnostics (DATA-01).
    source_bucket : str | None
        Nullable string for diagnostics (DATA-01).
    """

    seller_id: str = Field(..., description="Seller identifier (DATA-01)")
    lead_id: str = Field(..., description="Lead identifier (DATA-01)")
    attempt_id: str = Field(..., description="Globally unique attempt ID within source namespace (DATA-01)")
    source: str = Field(..., description="Producer identifier (DATA-01)")
    revision: int = Field(..., gt=0, description="Positive revision number for corrections")
    event_id: str = Field(..., description="Unique delivery identifier for audit (DATA-01)")
    finalized_at: datetime = Field(..., description="When outcome became available (DATA-02)")
    call_start_time: datetime = Field(..., description="Actual dial start time (DATA-02)")
    call_end_time: datetime = Field(..., description="Call end time, >= call_start_time (DATA-02)")
    lead_sent_time: datetime = Field(..., description="Lead sent time, <= call_start_time (DATA-02)")
    attempt_number: int = Field(..., gt=0, description="Attempt number within (seller_id, lead_id)")
    answered: bool = Field(..., description="Whether the call was answered")
    disposition: Disposition = Field(..., description="Call disposition code")
    meeting_fixed: bool = Field(..., description="Whether a meeting was fixed (DATA-04)")
    requested_callback_at: Optional[datetime] = Field(None, description="Seller-requested callback time (DATA-02)")
    decision_id: Optional[str] = Field(None, description="Links to scheduling decision (DATA-01)")
    duration_s: Optional[int] = Field(None, ge=0, description="Duration in seconds, nonnegative (DATA-03)")
    dialer_version: Optional[str] = Field(None, description="Dialer version for diagnostics (DATA-01)")
    source_bucket: Optional[str] = Field(None, description="Source bucket for diagnostics (DATA-01)")

    # ── Field validators ──────────────────────────────────────────────────

    @field_validator("seller_id", "lead_id", "attempt_id", "source", "event_id")
    @classmethod
    def _validate_id_field(cls, v: str) -> str:
        """Validate ID fields are nonempty strings <= 128 UTF-8 bytes (DATA-01)."""
        return _validate_id(v, field_name=cls.model_fields.get(v, FieldInfo()).alias or v)

    @field_validator("decision_id", "dialer_version", "source_bucket")
    @classmethod
    def _validate_optional_id_field(cls, v: str | None) -> str | None:
        """Validate optional ID fields when present (DATA-01)."""
        if v is not None:
            return _validate_id(v, field_name=cls.model_fields.get(v, FieldInfo()).alias or v)
        return v

    @field_validator("finalized_at", "call_start_time", "call_end_time", "lead_sent_time", mode="before")
    @classmethod
    def _validate_timestamp_field(cls, v: Any) -> datetime:
        """Validate ISO 8601 timestamps with explicit offset (DATA-02).

        Runs in 'before' mode so the raw string is validated before
        Pydantic coerces it to a ``datetime``.
        """
        return _validate_iso_offset(v, field_name="timestamp")

    @field_validator("requested_callback_at", mode="before")
    @classmethod
    def _validate_callback_at(cls, v: Any) -> datetime | None:
        """Validate requested_callback_at when present (DATA-02).

        Runs in 'before' mode so the raw string is validated before
        Pydantic coerces it to a ``datetime``.
        """
        if v is None:
            return None
        return _validate_iso_offset(v, field_name="requested_callback_at")

    # ── Cross-field validators ────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_constraints(self) -> FinalizedOutcome:
        """Enforce DATA-04 and SRS §4.2 cross-field constraints."""
        # DATA-04: meeting_fixed=True requires answered=True and disposition=MEETING_FIXED
        if self.meeting_fixed and not (self.answered and self.disposition is Disposition.MEETING_FIXED):
            raise ValueError(
                "meeting_fixed=True requires answered=True and disposition=MEETING_FIXED (DATA-04)"
            )

        # DATA-04: NOT_ANSWERED disposition requires answered=False
        if self.disposition is Disposition.NOT_ANSWERED and self.answered:
            raise ValueError(
                "disposition=NOT_ANSWERED requires answered=False (DATA-04)"
            )

        # call_start_time <= call_end_time
        if self.call_start_time > self.call_end_time:
            raise ValueError("call_start_time must be <= call_end_time")

        # lead_sent_time <= call_start_time
        if self.lead_sent_time > self.call_start_time:
            raise ValueError("lead_sent_time must be <= call_start_time")

        # duration_s must agree with call_end - call_start within one second
        if self.duration_s is not None:
            actual_duration = int((self.call_end_time - self.call_start_time).total_seconds())
            if abs(self.duration_s - actual_duration) > 1:
                raise ValueError(
                    f"duration_s ({self.duration_s}) must agree with "
                    f"call_end - call_start ({actual_duration}) within 1 second"
                )

        return self


class SellerProfile(_StrictBase):
    """Seller profile for segment resolution (SRS §4.3).

    Contains the seller's category, turnover band, and business type
    used for hierarchical segment resolution. Nullable fields indicate
    missing or unknown values; the adapter maps them to an explicit
    ``UNKNOWN`` value.

    Attributes
    ----------
    seller_id : str
        Seller identifier (DATA-01).
    category_group : str | None
        Category group, nullable (DATA-01).
    turnover_band : str | None
        Turnover band, nullable (DATA-01).
    business_type : str | None
        Business type, nullable (DATA-01).
    effective_from : datetime
        Timestamp when this profile version became effective (DATA-02).
    profile_version : str
        Version string for the profile.
    """

    seller_id: str = Field(..., description="Seller identifier (DATA-01)")
    category_group: Optional[str] = Field(None, description="Category group, nullable (DATA-01)")
    turnover_band: Optional[str] = Field(None, description="Turnover band, nullable (DATA-01)")
    business_type: Optional[str] = Field(None, description="Business type, nullable (DATA-01)")
    effective_from: datetime = Field(..., description="Profile effective timestamp (DATA-02)")
    profile_version: str = Field(..., min_length=1, description="Version identifier for the profile")

    @field_validator("seller_id")
    @classmethod
    def _validate_seller_id(cls, v: str) -> str:
        """Validate seller_id is nonempty string <= 128 UTF-8 bytes (DATA-01)."""
        return _validate_id(v, field_name="seller_id")

    @field_validator("category_group", "turnover_band", "business_type")
    @classmethod
    def _validate_optional_str_field(cls, v: str | None) -> str | None:
        """Validate optional string fields when present (DATA-01)."""
        if v is not None:
            return _validate_id(v, field_name="optional field")
        return v

    @field_validator("effective_from", mode="before")
    @classmethod
    def _validate_effective_from(cls, v: Any) -> datetime:
        """Validate effective_from is ISO 8601 with offset (DATA-02).

        Runs in 'before' mode so the raw string is validated before
        Pydantic coerces it to a ``datetime``.
        """
        return _validate_iso_offset(v, field_name="effective_from")


class RecommendRequest(_StrictBase):
    """Recommendation request (SRS §10.1, API-01).

    Submitted by a trusted caller to obtain a call-time recommendation
    for a seller-lead pair. Each request carries a UUID for idempotency.

    Attributes
    ----------
    request_id : str
        UUID request identifier supplied by the caller (DATA-01).
    seller_id : str
        Seller identifier (DATA-01).
    lead_id : str
        Lead identifier (DATA-01).
    earliest_at : datetime
        Earliest feasible call timestamp (DATA-02).
    latest_at : datetime
        Latest feasible call timestamp (DATA-02).
    context_version : int
        Context version, defaults to 1 (optional).
    """

    request_id: str = Field(..., description="UUID request identifier (DATA-01)")
    seller_id: str = Field(..., description="Seller identifier (DATA-01)")
    lead_id: str = Field(..., description="Lead identifier (DATA-01)")
    earliest_at: datetime = Field(..., description="Earliest feasible call time (DATA-02)")
    latest_at: datetime = Field(..., description="Latest feasible call time (DATA-02)")
    context_version: int = Field(default=1, description="Context version, default 1")

    @field_validator("request_id", "seller_id", "lead_id")
    @classmethod
    def _validate_id_field(cls, v: str) -> str:
        """Validate ID fields are nonempty strings <= 128 UTF-8 bytes (DATA-01)."""
        return _validate_id(v, field_name=cls.model_fields.get(v, FieldInfo()).alias or v)

    @field_validator("earliest_at", "latest_at", mode="before")
    @classmethod
    def _validate_timestamp_field(cls, v: Any) -> datetime:
        """Validate ISO 8601 timestamps with explicit offset (DATA-02).

        Runs in 'before' mode so the raw string is validated before
        Pydantic coerces it to a ``datetime``.
        """
        return _validate_iso_offset(v, field_name="timestamp")

    @model_validator(mode="after")
    def _validate_time_range(self) -> RecommendRequest:
        """Ensure earliest_at <= latest_at."""
        if self.earliest_at > self.latest_at:
            raise ValueError("earliest_at must be <= latest_at")
        return self


class RecommendResponse(_StrictBase):
    """Recommendation response (SRS §10.2).

    Contains the full decision payload including model diagnostics,
    experiment metadata, and scheduling information. All fields are
    required per SRS §10.2; nullable fields use ``Optional``.

    Attributes
    ----------
    decision_id : str
        Unique decision identifier (DATA-01).
    request_id : str
        Echoed request identifier (DATA-01).
    seller_id : str
        Seller identifier (DATA-01).
    lead_id : str
        Lead identifier (DATA-01).
    status : DecisionStatus
        Decision status (SRS §10.2).
    scheduled_at : datetime | None
        Primary recommended call timestamp, nullable (DATA-02).
    secondary_at : datetime | None
        Secondary recommended call timestamp, nullable (DATA-02).
    reason_code : str
        Human-readable reason code for the decision.
    mode : PolicyMode
        Policy mode used for this recommendation (SRS §10.2).
    assignment : Assignment
        Experiment assignment arm (SRS §10.2, EXP-01).
    experiment_id : str | None
        Experiment identifier, nullable (DATA-01).
    policy_version : str
        Version of the policy used.
    bundle_id : str
        Model bundle identifier (DATA-01).
    model_compatibility_id : str
        Model compatibility identifier (DATA-01).
    profile_version : str
        Seller profile version used.
    calendar_version : str
        Business calendar version used.
    context_version : int
        Context version used for the decision.
    state_version : int
        Seller model state version.
    n_attempts : int
        Number of finalized attempts for the seller.
    n_eff : int
        Effective number of observations.
    prior_level : float
        Prior level (shrinkage target).
    prior_weight : float
        Weight of the prior in the posterior.
    expected_reward : float
        Expected reward for the chosen action under the posterior mean.
    latent_std : float
        Latent standard deviation (posterior uncertainty).
    predictive_std : float
        Predictive standard deviation (including noise).
    candidate_count : int
        Number of eligible candidates evaluated.
    action_probability : float | None
        Probability of selecting the chosen action, nullable.
    assignment_probability : float | None
        Probability of the assignment arm, nullable.
    ope_eligible : bool
        Whether this decision is eligible for off-policy evaluation.
    created_at : datetime
        Timestamp when the decision was created (DATA-02).
    valid_until : datetime | None
        Earliest of lead expiry, context expiry, and scheduled_at +
        tolerance; nullable when no action exists (DATA-02).
    """

    decision_id: str = Field(..., description="Unique decision identifier (DATA-01)")
    request_id: str = Field(..., description="Echoed request identifier (DATA-01)")
    seller_id: str = Field(..., description="Seller identifier (DATA-01)")
    lead_id: str = Field(..., description="Lead identifier (DATA-01)")
    status: DecisionStatus = Field(..., description="Decision status (SRS §10.2)")
    scheduled_at: Optional[datetime] = Field(None, description="Primary call timestamp (DATA-02)")
    secondary_at: Optional[datetime] = Field(None, description="Secondary call timestamp (DATA-02)")
    reason_code: str = Field(..., min_length=1, description="Human-readable reason code")
    mode: PolicyMode = Field(..., description="Policy mode (SRS §10.2)")
    assignment: Assignment = Field(..., description="Experiment assignment arm (EXP-01)")
    experiment_id: Optional[str] = Field(None, description="Experiment identifier (DATA-01)")
    policy_version: str = Field(..., min_length=1, description="Policy version used")
    bundle_id: str = Field(..., description="Model bundle identifier (DATA-01)")
    model_compatibility_id: str = Field(..., description="Model compatibility ID (DATA-01)")
    profile_version: str = Field(..., description="Seller profile version")
    calendar_version: str = Field(..., description="Business calendar version")
    context_version: int = Field(..., description="Context version")
    state_version: int = Field(..., description="Seller model state version")
    n_attempts: int = Field(..., description="Number of finalized attempts")
    n_eff: int = Field(..., description="Effective number of observations")
    prior_level: float = Field(..., description="Prior level (shrinkage target)")
    prior_weight: float = Field(..., description="Prior weight in posterior")
    expected_reward: float = Field(..., description="Expected reward for chosen action")
    latent_std: float = Field(..., description="Latent standard deviation")
    predictive_std: float = Field(..., description="Predictive standard deviation")
    candidate_count: int = Field(..., description="Number of eligible candidates")
    action_probability: Optional[float] = Field(None, description="Action selection probability")
    assignment_probability: Optional[float] = Field(None, description="Assignment arm probability")
    ope_eligible: bool = Field(..., description="Eligible for off-policy evaluation")
    created_at: datetime = Field(..., description="Decision creation time (DATA-02)")
    valid_until: Optional[datetime] = Field(None, description="Decision validity expiry (DATA-02)")

    # ── Field validators ──────────────────────────────────────────────────

    @field_validator("decision_id", "request_id", "seller_id", "lead_id",
                     "experiment_id", "bundle_id", "model_compatibility_id")
    @classmethod
    def _validate_id_field(cls, v: str | None) -> str | None:
        """Validate ID fields are nonempty strings <= 128 UTF-8 bytes (DATA-01)."""
        if v is not None:
            return _validate_id(v, field_name=cls.model_fields.get(v, FieldInfo()).alias or v)
        return v

    @field_validator("scheduled_at", "secondary_at", "created_at", "valid_until", mode="before")
    @classmethod
    def _validate_timestamp_field(cls, v: Any) -> datetime | None:
        """Validate ISO 8601 timestamps with explicit offset (DATA-02).

        Runs in 'before' mode so the raw string is validated before
        Pydantic coerces it to a ``datetime``.
        """
        if v is None:
            return None
        return _validate_iso_offset(v, field_name="timestamp")

    @model_validator(mode="after")
    def _validate_no_action_fields(self) -> RecommendResponse:
        """Enforce SRS §10.2: no-slot/stop/review/superseded responses
        have null scheduled_at, secondary_at, and valid_until.

        When status is NO_ELIGIBLE_SLOT, STOP, MANUAL_REVIEW, or SUPERSEDED,
        timestamps must be null and candidate_count must be zero.
        """
        no_action_statuses = {DecisionStatus.NO_ELIGIBLE_SLOT, DecisionStatus.STOP,
                              DecisionStatus.MANUAL_REVIEW, DecisionStatus.SUPERSEDED}
        if self.status in no_action_statuses:
            if self.scheduled_at is not None:
                raise ValueError(
                    "scheduled_at must be null for status "
                    f"{self.status.value} (SRS §10.2)"
                )
            if self.secondary_at is not None:
                raise ValueError(
                    "secondary_at must be null for status "
                    f"{self.status.value} (SRS §10.2)"
                )
            if self.valid_until is not None:
                raise ValueError(
                    "valid_until must be null for status "
                    f"{self.status.value} (SRS §10.2)"
                )
            if self.candidate_count != 0:
                raise ValueError(
                    "candidate_count must be 0 for status "
                    f"{self.status.value} (SRS §10.2)"
                )
            if self.action_probability is not None:
                raise ValueError(
                    "action_probability must be null for status "
                    f"{self.status.value} (SRS §10.2)"
                )

        # Shadow mode constraints (SRS §10.2)
        if self.assignment is Assignment.SHADOW:
            if self.experiment_id is not None:
                raise ValueError(
                    "experiment_id must be null for SHADOW assignment (SRS §10.2)"
                )
            if self.action_probability is not None:
                raise ValueError(
                    "action_probability must be null for SHADOW assignment (SRS §10.2)"
                )
            if self.assignment_probability is not None:
                raise ValueError(
                    "assignment_probability must be null for SHADOW assignment (SRS §10.2)"
                )
            if self.ope_eligible:
                raise ValueError(
                    "ope_eligible must be False for SHADOW assignment (SRS §10.2)"
                )

        return self


class RetryRequest(_StrictBase):
    """Retry request (SRS §8.2, RET-04).

    Requests a retry recommendation based on the latest outcome of an
    attempt. The request is keyed by ``(source, attempt_id, revision)``
    for idempotency.

    Attributes
    ----------
    request_id : str
        UUID request identifier (DATA-01).
    source : str
        Producer identifier (DATA-01).
    attempt_id : str
        Attempt identifier (DATA-01).
    expected_revision : int
        Expected revision number for optimistic concurrency control.
    """

    request_id: str = Field(..., description="UUID request identifier (DATA-01)")
    source: str = Field(..., description="Producer identifier (DATA-01)")
    attempt_id: str = Field(..., description="Attempt identifier (DATA-01)")
    expected_revision: int = Field(..., gt=0, description="Expected revision for concurrency control")

    @field_validator("request_id", "source", "attempt_id")
    @classmethod
    def _validate_id_field(cls, v: str) -> str:
        """Validate ID fields are nonempty strings <= 128 UTF-8 bytes (DATA-01)."""
        return _validate_id(v, field_name=cls.model_fields.get(v, FieldInfo()).alias or v)


class RetryResponse(_StrictBase):
    """Retry response (SRS §8.2, RET-04).

    Contains the retry decision result, including the proposed timestamp
    when applicable.

    Attributes
    ----------
    decision_id : str
        Unique retry decision identifier (DATA-01).
    request_id : str
        Echoed request identifier (DATA-01).
    source : str
        Producer identifier (DATA-01).
    attempt_id : str
        Attempt identifier (DATA-01).
    result : RetryResult
        Retry decision result (SRS §8.2).
    scheduled_at : datetime | None
        Proposed retry timestamp, nullable (DATA-02).
    reason_code : str
        Human-readable reason code.
    policy_version : str
        Version of the retry policy used.
    superseded_by : str | None
        Decision ID that superseded this one, nullable (DATA-01).
    """

    decision_id: str = Field(..., description="Unique retry decision ID (DATA-01)")
    request_id: str = Field(..., description="Echoed request identifier (DATA-01)")
    source: str = Field(..., description="Producer identifier (DATA-01)")
    attempt_id: str = Field(..., description="Attempt identifier (DATA-01)")
    result: RetryResult = Field(..., description="Retry decision result (SRS §8.2)")
    scheduled_at: Optional[datetime] = Field(None, description="Proposed retry timestamp (DATA-02)")
    reason_code: str = Field(..., min_length=1, description="Human-readable reason code")
    policy_version: str = Field(..., min_length=1, description="Retry policy version")
    superseded_by: Optional[str] = Field(None, description="Superseding decision ID (DATA-01)")

    @field_validator("decision_id", "request_id", "source", "attempt_id", "superseded_by")
    @classmethod
    def _validate_id_field(cls, v: str | None) -> str | None:
        """Validate ID fields are nonempty strings <= 128 UTF-8 bytes (DATA-01)."""
        if v is not None:
            return _validate_id(v, field_name=cls.model_fields.get(v, FieldInfo()).alias or v)
        return v

    @field_validator("scheduled_at", mode="before")
    @classmethod
    def _validate_scheduled_at(cls, v: Any) -> datetime | None:
        """Validate scheduled_at is ISO 8601 with offset when present (DATA-02).

        Runs in 'before' mode so the raw string is validated before
        Pydantic coerces it to a ``datetime``.
        """
        if v is None:
            return None
        return _validate_iso_offset(v, field_name="scheduled_at")


class ErrorResponseBody(_StrictBase):
    """Error response body (SRS §10.2, API-02).

    All error responses from the API use this structure. Internal
    stack traces are never included.

    Attributes
    ----------
    error_code : str
        Machine-readable error code (e.g. INVALID_RUNTIME_MODE).
    message : str
        Human-readable error message.
    request_id : str
        Request identifier for correlation (DATA-01).
    retryable : bool
        Whether the client should retry the request.
    """

    error_code: str = Field(..., min_length=1, description="Machine-readable error code (API-02)")
    message: str = Field(..., min_length=1, description="Human-readable error message")
    request_id: str = Field(..., description="Request identifier for correlation (DATA-01)")
    retryable: bool = Field(..., description="Whether the client should retry")

    @field_validator("request_id")
    @classmethod
    def _validate_request_id(cls, v: str) -> str:
        """Validate request_id is nonempty string <= 128 UTF-8 bytes (DATA-01)."""
        return _validate_id(v, field_name="request_id")


# ── Union type for any versioned schema ───────────────────────────────────────

VersionedSchema = Union[
    FinalizedOutcome,
    SellerProfile,
    RecommendRequest,
    RecommendResponse,
    RetryRequest,
    RetryResponse,
    ErrorResponseBody,
]
