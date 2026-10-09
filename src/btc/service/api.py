"""FastAPI service endpoints for the Best Time to Call system.

Implements SRS §10.1–10.2 and API-01 through API-02:

- POST /v1/best-time — POL-01/02/03: Generate best time recommendation
- POST /v1/outcomes — STATE-01: Record finalised outcome
- POST /v1/retries — RET-04: Generate retry decision
- GET /healthz — Process liveness
- GET /readyz — Readiness of model/config/stores
- GET /metrics — Operational metrics (no high-cardinality seller labels)

Authentication:
    Service identity via Bearer token (API-01).
    Authorization: outcomes producers, decision callers, model admins.

Error format (API-02):
    {error_code, message, request_id, retryable}
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from btc.config import Config, default_config
from btc.data.adapters import parse_timestamp
from btc.experiment import assignment as exp_assignment
from btc.experiment import logging as exp_logging
from btc.schemas import (
    Assignment,
    DecisionStatus,
    ErrorResponseBody,
    FinalizedOutcome,
    PolicyMode,
    RecommendRequest as SchemasRecommendRequest,
    RecommendResponse as SchemasRecommendResponse,
    RetryRequest as SchemasRetryRequest,
    RetryResponse as SchemasRetryResponse,
    RetryResult,
)

logger = logging.getLogger(__name__)


# ── Authentication ─────────────────────────────────────────────────────────────

# Development default — must be replaced with a secret management integration
# in production (API-01).
_DEFAULT_BEARER_TOKEN = "dev-token-change-me"


def _authenticate(bearer: Optional[str] = None) -> bool:
    """Validate the Bearer token.

    Parameters
    ----------
    bearer : str | None
        The Authorization Bearer token.

    Returns
    -------
    bool
        True if authenticated.
    """
    if bearer is None:
        return False
    # Constant-time comparison to prevent timing attacks
    return hashlib.sha256(bearer.encode("utf-8")).hexdigest() == hashlib.sha256(
        _DEFAULT_BEARER_TOKEN.encode("utf-8")
    ).hexdigest()


# ── Request/Response models ────────────────────────────────────────────────────


class RecommendRequest(BaseModel):
    """Recommendation request body.

    Mirrors SRS §10.1 recommendation request fields with Pydantic
    validation.
    """

    request_id: str = Field(..., description="UUID request identifier (DATA-01)")
    seller_id: str = Field(..., description="Seller identifier (DATA-01)")
    lead_id: str = Field(..., description="Lead identifier (DATA-01)")
    earliest_at: str = Field(..., description="Earliest feasible call time (ISO 8601, DATA-02)")
    latest_at: str = Field(..., description="Latest feasible call time (ISO 8601, DATA-02)")
    context_version: int = Field(default=1, description="Context version, default 1")
    lead_sent_time: Optional[str] = Field(
        default=None, description="When lead was sent (ISO 8601, nullable)"
    )
    lead_expiry: Optional[str] = Field(
        default=None, description="Lead expiry time (ISO 8601, nullable)"
    )
    calls_already_today: int = Field(
        default=0, description="Calls already placed today"
    )
    attempt_number: int = Field(
        default=1, description="Current attempt number"
    )

    class Config:
        extra = "forbid"


class RecommendResponse(BaseModel):
    """Recommendation response body (SRS §10.2).

    Contains the full decision payload including model diagnostics,
    experiment metadata, and scheduling information.
    """

    decision_id: str = Field(..., description="Unique decision identifier (DATA-01)")
    request_id: str = Field(..., description="Echoed request identifier (DATA-01)")
    seller_id: str = Field(..., description="Seller identifier (DATA-01)")
    lead_id: str = Field(..., description="Lead identifier (DATA-01)")
    status: str = Field(..., description="Decision status (SRS §10.2)")
    scheduled_at: Optional[str] = Field(
        default=None, description="Primary call timestamp (ISO 8601, nullable)"
    )
    secondary_at: Optional[str] = Field(
        default=None, description="Secondary call timestamp (ISO 8601, nullable)"
    )
    reason_code: str = Field(..., min_length=1, description="Human-readable reason code")
    mode: str = Field(..., description="Policy mode (SRS §10.2)")
    assignment: str = Field(..., description="Experiment assignment arm (EXP-01)")
    experiment_id: Optional[str] = Field(
        default=None, description="Experiment identifier (nullable)"
    )
    policy_version: str = Field(..., min_length=1, description="Policy version used")
    bundle_id: str = Field(..., description="Model bundle identifier (DATA-01)")
    model_compatibility_id: str = Field(
        ..., description="Model compatibility ID (DATA-01)"
    )
    profile_version: str = Field(..., description="Seller profile version")
    calendar_version: str = Field(..., description="Business calendar version")
    context_version: int = Field(..., description="Context version")
    state_version: int = Field(..., description="Seller model state version")
    n_attempts: int = Field(..., description="Number of finalised attempts")
    n_eff: int = Field(..., description="Effective number of observations")
    prior_level: float = Field(..., description="Prior level (shrinkage target)")
    prior_weight: float = Field(..., description="Prior weight in posterior")
    expected_reward: float = Field(..., description="Expected reward for chosen action")
    latent_std: float = Field(..., description="Latent standard deviation")
    predictive_std: float = Field(..., description="Predictive standard deviation")
    candidate_count: int = Field(..., description="Number of eligible candidates")
    action_probability: Optional[float] = Field(
        default=None, description="Action selection probability"
    )
    assignment_probability: Optional[float] = Field(
        default=None, description="Assignment arm probability"
    )
    ope_eligible: bool = Field(..., description="Eligible for off-policy evaluation")
    created_at: str = Field(..., description="Decision creation time (ISO 8601)")
    valid_until: Optional[str] = Field(
        default=None, description="Decision validity expiry (ISO 8601, nullable)"
    )

    class Config:
        extra = "forbid"


class OutcomeRequest(BaseModel):
    """Finalized outcome request body.

    Mirrors the FinalizedOutcome schema fields from SRS §4.2.
    """

    seller_id: str = Field(..., description="Seller identifier (DATA-01)")
    lead_id: str = Field(..., description="Lead identifier (DATA-01)")
    attempt_id: str = Field(..., description="Attempt identifier (DATA-01)")
    source: str = Field(..., description="Producer identifier (DATA-01)")
    revision: int = Field(..., gt=0, description="Positive revision number")
    event_id: str = Field(..., description="Unique delivery identifier (DATA-01)")
    finalized_at: str = Field(..., description="Outcome availability time (ISO 8601)")
    call_start_time: str = Field(..., description="Dial start time (ISO 8601)")
    call_end_time: str = Field(..., description="Call end time (ISO 8601)")
    lead_sent_time: str = Field(..., description="Lead sent time (ISO 8601)")
    attempt_number: int = Field(..., gt=0, description="Attempt number")
    answered: bool = Field(..., description="Whether call was answered")
    disposition: str = Field(..., description="Call disposition code")
    meeting_fixed: bool = Field(..., description="Whether meeting was fixed")
    requested_callback_at: Optional[str] = Field(
        default=None, description="Seller-requested callback time (ISO 8601, nullable)"
    )
    decision_id: Optional[str] = Field(
        default=None, description="Links to scheduling decision (nullable)"
    )
    duration_s: Optional[int] = Field(
        default=None, ge=0, description="Duration in seconds (nullable)"
    )
    dialer_version: Optional[str] = Field(
        default=None, description="Dialer version for diagnostics (nullable)"
    )
    source_bucket: Optional[str] = Field(
        default=None, description="Source bucket for diagnostics (nullable)"
    )

    class Config:
        extra = "forbid"


class OutcomeResponse(BaseModel):
    """Outcome processing response body."""

    status: str = Field(
        ...,
        description="APPLIED, DUPLICATE, or STALE_REVISION",
    )
    state_version: int = Field(..., description="New seller state version")
    attempt_id: str = Field(..., description="Processed attempt identifier")
    revision: int = Field(..., description="Accepted revision number")
    n_attempts: int = Field(..., description="Total finalised attempts for seller")

    class Config:
        extra = "forbid"


class RetryRequest(BaseModel):
    """Retry request body (SRS §8.2, RET-04)."""

    request_id: str = Field(..., description="UUID request identifier (DATA-01)")
    source: str = Field(..., description="Producer identifier (DATA-01)")
    attempt_id: str = Field(..., description="Attempt identifier (DATA-01)")
    expected_revision: int = Field(
        ..., gt=0, description="Expected revision for concurrency control"
    )

    class Config:
        extra = "forbid"


class RetryResponse(BaseModel):
    """Retry response body (SRS §8.2, RET-04)."""

    decision_id: str = Field(..., description="Unique retry decision ID (DATA-01)")
    request_id: str = Field(..., description="Echoed request identifier (DATA-01)")
    source: str = Field(..., description="Producer identifier (DATA-01)")
    attempt_id: str = Field(..., description="Attempt identifier (DATA-01)")
    result: str = Field(..., description="Retry decision result (SRS §8.2)")
    scheduled_at: Optional[str] = Field(
        default=None, description="Proposed retry timestamp (ISO 8601, nullable)"
    )
    reason_code: str = Field(..., min_length=1, description="Human-readable reason code")
    policy_version: str = Field(..., min_length=1, description="Retry policy version")
    superseded_by: Optional[str] = Field(
        default=None, description="Superseding decision ID (nullable)"
    )

    class Config:
        extra = "forbid"


# ── Metrics tracking ───────────────────────────────────────────────────────────


class _Metrics:
    """In-memory operational metrics (no high-cardinality seller labels)."""

    def __init__(self) -> None:
        self._lock: Any = None
        try:
            import threading
            self._lock = threading.Lock()
        except ImportError:
            pass
        self.request_count: int = 0
        self.outcome_count: int = 0
        self.retry_count: int = 0
        self.error_count: int = 0
        self.status_counts: dict[str, int] = {}
        self.assignment_counts: dict[str, int] = {}
        self.mode_counts: dict[str, int] = {}
        self.start_time: float = time.time()

    def record_decision(
        self,
        status: str,
        assignment: str,
        mode: str,
    ) -> None:
        """Record a decision outcome in metrics."""
        if self._lock:
            with self._lock:
                self.request_count += 1
                self.status_counts[status] = self.status_counts.get(status, 0) + 1
                self.assignment_counts[assignment] = (
                    self.assignment_counts.get(assignment, 0) + 1
                )
                self.mode_counts[mode] = self.mode_counts.get(mode, 0) + 1

    def record_outcome(self) -> None:
        """Record an outcome processing event."""
        if self._lock:
            with self._lock:
                self.outcome_count += 1

    def record_retry(self) -> None:
        """Record a retry event."""
        if self._lock:
            with self._lock:
                self.retry_count += 1

    def record_error(self) -> None:
        """Record an error event."""
        if self._lock:
            with self._lock:
                self.error_count += 1

    def snapshot(self) -> dict:
        """Return a metrics snapshot (no seller-level data).

        Returns
        -------
        dict
            Metrics dictionary.
        """
        if self._lock:
            with self._lock:
                return {
                    "uptime_seconds": time.time() - self.start_time,
                    "request_count": self.request_count,
                    "outcome_count": self.outcome_count,
                    "retry_count": self.retry_count,
                    "error_count": self.error_count,
                    "status_counts": dict(self.status_counts),
                    "assignment_counts": dict(self.assignment_counts),
                    "mode_counts": dict(self.mode_counts),
                }
        return {
            "uptime_seconds": time.time() - self.start_time,
            "request_count": self.request_count,
            "outcome_count": self.outcome_count,
            "retry_count": self.retry_count,
            "error_count": self.error_count,
            "status_counts": dict(self.status_counts),
            "assignment_counts": dict(self.assignment_counts),
            "mode_counts": dict(self.mode_counts),
        }


# ── Store and config ───────────────────────────────────────────────────────────


class _ServiceState:
    """Mutable service state injected into the FastAPI app."""

    def __init__(
        self,
        config: Config,
        store: Any = None,
        model_store: Any = None,
        metrics: Optional[_Metrics] = None,
    ) -> None:
        self.config = config
        self.store = store  # Database connection / outcome store
        self.model_store = model_store  # Seller state store
        self.metrics = metrics or _Metrics()
        self.experiment_id = "btc-timing-exp-1"
        self.salt = "release-1-salt"
        self.policy_version = "1.0.0"
        self.bundle_id = "btc-bundle-1.0.0"
        self.model_compatibility_id = "btc-compat-1.0.0"
        self.profile_version = "1.0.0"
        self.calendar_version = "1.0.0"
        self.ready = True
        self._ready_checks: list[tuple[str, bool]] = []


# ── Clock ──────────────────────────────────────────────────────────────────────


def _utc_now() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


# ── Helper: build recommendation response ──────────────────────────────────────


def _build_recommend_response(
    decision_id: str,
    request: RecommendRequest,
    status: str,
    scheduled_at: Optional[datetime],
    secondary_at: Optional[datetime],
    reason_code: str,
    mode: str,
    assignment: str,
    experiment_id: Optional[str],
    state_version: int,
    n_attempts: int,
    n_eff: int,
    prior_level: float,
    prior_weight: float,
    expected_reward: float,
    latent_std: float,
    predictive_std: float,
    candidate_count: int,
    action_probability: Optional[float],
    assignment_probability: Optional[float],
    ope_eligible: bool,
    valid_until: Optional[datetime],
    created_at: datetime,
    state: _ServiceState,
) -> RecommendResponse:
    """Build a RecommendResponse from internal decision data.

    Parameters
    ----------
    decision_id : str
        Unique decision identifier.
    request : RecommendRequest
        Original request.
    status : str
        Decision status.
    scheduled_at : datetime | None
        Primary recommended timestamp.
    secondary_at : datetime | None
        Secondary recommended timestamp.
    reason_code : str
        Reason code.
    mode : str
        Policy mode.
    assignment : str
        Experiment assignment.
    experiment_id : str | None
        Experiment identifier.
    state_version : int
        Seller state version.
    n_attempts : int
        Number of finalised attempts.
    n_eff : int
        Effective observations.
    prior_level : float
        Prior level.
    prior_weight : float
        Prior weight.
    expected_reward : float
        Expected reward.
    latent_std : float
        Latent standard deviation.
    predictive_std : float
        Predictive standard deviation.
    candidate_count : int
        Number of candidates.
    action_probability : float | None
        Action selection probability.
    assignment_probability : float | None
        Assignment arm probability.
    ope_eligible : bool
        OPE eligibility.
    valid_until : datetime | None
        Decision validity expiry.
    created_at : datetime
        Decision creation time.
    state : _ServiceState
        Service state.

    Returns
    -------
    RecommendResponse
        Formatted response.
    """
    def _fmt(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    return RecommendResponse(
        decision_id=decision_id,
        request_id=request.request_id,
        seller_id=request.seller_id,
        lead_id=request.lead_id,
        status=status,
        scheduled_at=_fmt(scheduled_at),
        secondary_at=_fmt(secondary_at),
        reason_code=reason_code,
        mode=mode,
        assignment=assignment,
        experiment_id=experiment_id,
        policy_version=state.policy_version,
        bundle_id=state.bundle_id,
        model_compatibility_id=state.model_compatibility_id,
        profile_version=state.profile_version,
        calendar_version=state.calendar_version,
        context_version=request.context_version,
        state_version=state_version,
        n_attempts=n_attempts,
        n_eff=n_eff,
        prior_level=prior_level,
        prior_weight=prior_weight,
        expected_reward=expected_reward,
        latent_std=latent_std,
        predictive_std=predictive_std,
        candidate_count=candidate_count,
        action_probability=action_probability,
        assignment_probability=assignment_probability,
        ope_eligible=ope_eligible,
        created_at=_fmt(created_at),
        valid_until=_fmt(valid_until),
    )


# ── Helper: log decision to experiment store ───────────────────────────────────


def _log_to_experiment_store(
    state: _ServiceState,
    request: RecommendRequest,
    response: RecommendResponse,
    candidate_timestamps: list,
    action_probability: Optional[float],
    assignment: str,
    experiment_id: Optional[str],
    state_version: int,
    clock,
    evaluation_status: Optional[str] = None,
) -> Optional[str]:
    """Log a decision to the experiment store (EXP-03).

    Parameters
    ----------
    state : _ServiceState
        Service state.
    request : RecommendRequest
        Original request.
    response : RecommendResponse
        Response to log.
    candidate_timestamps : list
        Sorted candidate timestamps.
    action_probability : float | None
        Action selection probability.
    assignment : str
        Assignment arm.
    experiment_id : str | None
        Experiment identifier.
    state_version : int
        Seller state version.
    clock : callable
        Server time.
    evaluation_status : str | None
        EXP-04 evaluation status.

    Returns
    -------
    str | None
        Decision ID, or None if logging is skipped.
    """
    try:
        request_hash = exp_logging.compute_canonical_request_hash(
            {
                "seller_id": request.seller_id,
                "lead_id": request.lead_id,
                "earliest_at": request.earliest_at,
                "latest_at": request.latest_at,
                "context_version": request.context_version,
            }
        )

        decision_id = exp_logging.log_decision(
            request_id=request.request_id,
            request_hash=request_hash,
            response=response.model_dump(mode="json", exclude_unset=True),
            candidate_timestamps=candidate_timestamps,
            action_probability=action_probability,
            assignment=assignment,
            experiment_id=experiment_id,
            state_version=state_version,
            model_compatibility_id=state.model_compatibility_id,
            bundle_id=state.bundle_id,
            clock=clock,
            evaluation_status=evaluation_status,
        )
        return decision_id
    except exp_logging.DecisionConflictError as exc:
        logger.warning("Decision conflict for request_id=%s: %s", request.request_id, exc)
        raise
    except Exception as exc:
        logger.error("Decision logging failed for request_id=%s: %s", request.request_id, exc)
        # EXP-03: Failing durable decision logging fails the request with 503
        raise HTTPException(
            status_code=503,
            detail=ErrorResponseBody(
                error_code="DECISION_LOGGING_FAILED",
                message=f"Decision logging failed: {exc}",
                request_id=request.request_id,
                retryable=True,
            ).model_dump(),
        ) from exc


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Best Time to Call Service",
    description="SRS §10: Best Time to Call prediction and retry service",
    version="1.0.0",
)

# Service state — attached to the app for dependency injection.
state: _ServiceState


@app.on_event("startup")
async def _startup() -> None:
    """Initialise service state on startup."""
    global state
    cfg = default_config()
    # Validate config
    from btc.config import validate_config
    try:
        validate_config(cfg)
    except Exception as exc:
        logger.error("Configuration validation failed: %s", exc)
    state = _ServiceState(config=cfg)


def _get_state() -> _ServiceState:
    """Get the current service state from the FastAPI app.

    Returns
    -------
    _ServiceState
        Service state.
    """
    return app.state


# ── Middleware ─────────────────────────────────────────────────────────────────


@app.middleware("http")
async def _auth_middleware(request: Request, call_next) -> Any:
    """API-01: Authenticate service identity via Bearer token.

    Health endpoints (/healthz, /readyz, /metrics) do not require auth.
    All other endpoints require a valid Bearer token.
    """
    # Skip auth for health/readiness/metrics endpoints
    if request.url.path in ("/healthz", "/readyz", "/metrics", "/openapi.json", "/docs", "/redoc"):
        return await call_next(request)

    # Check Bearer token
    authorization = request.headers.get("authorization", "")
    bearer = None
    if authorization.startswith("Bearer "):
        bearer = authorization[7:]

    if not _authenticate(bearer):
        return _error_response(
            error_code="UNAUTHORIZED",
            message="Missing or invalid authentication token",
            request_id="",
            retryable=False,
            status_code=401,
        )

    return await call_next(request)


def _error_response(
    error_code: str,
    message: str,
    request_id: str,
    retryable: bool,
    status_code: int = 422,
) -> Any:
    """API-02: Create an error response.

    Parameters
    ----------
    error_code : str
        Machine-readable error code.
    message : str
        Human-readable error message.
    request_id : str
        Request identifier for correlation.
    retryable : bool
        Whether the client should retry.
    status_code : int
        HTTP status code.

    Returns
    -------
    Any
        JSON response.
    """
    from fastapi.responses import JSONResponse

    body = ErrorResponseBody(
        error_code=error_code,
        message=message,
        request_id=request_id,
        retryable=retryable,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.post(
    "/v1/best-time",
    response_model=RecommendResponse,
    responses={
        401: {"model": dict, "description": "Unauthorized"},
        409: {"model": dict, "description": "Request conflict"},
        422: {"model": dict, "description": "Schema or interval violation"},
        503: {"model": dict, "description": "Required dependency unavailable"},
    },
    tags=["Best Time"],
)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """POL-01/02/03: Generate best time recommendation.

    Computes eligible candidates, resolves experiment assignment,
    selects an action, logs the decision (EXP-03), and returns the
    recommendation response (SRS §10.2).

    Request processing:
    1. Validate request fields.
    2. Resolve experiment assignment (EXP-01).
    3. Generate candidates with calendar/support constraints.
    4. Select action based on assignment and seller state.
    5. Log decision to experiment store (EXP-03).
    6. Return recommendation response.

    Parameters
    ----------
    request : RecommendRequest
        Recommendation request body.

    Returns
    -------
    RecommendResponse
        SRS §10.2 recommendation response.

    Raises
    ------
    HTTPException
        401: Unauthorized (API-02).
        409: Request conflict — same request_id, different payload (API-01).
        422: Schema violation or invalid time range (API-02).
        503: Decision logging failure (EXP-03).
    """
    svc = _get_state()
    request_id = request.request_id

    try:
        # ── Parse timestamps ───────────────────────────────────────────
        earliest_at = parse_timestamp(request.earliest_at)
        latest_at = parse_timestamp(request.latest_at)

        if earliest_at > latest_at:
            raise HTTPException(
                status_code=422,
                detail=_error_body(
                    "INVALID_TIME_RANGE",
                    "earliest_at must be <= latest_at",
                    request_id,
                    False,
                ),
            )

        # ── Parse optional fields ──────────────────────────────────────
        lead_sent_time = None
        if request.lead_sent_time:
            try:
                lead_sent_time = parse_timestamp(request.lead_sent_time)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=_error_body(
                        "INVALID_TIMESTAMP",
                        f"Invalid lead_sent_time: {request.lead_sent_time}",
                        request_id,
                        False,
                    ),
                )

        lead_expiry = None
        if request.lead_expiry:
            try:
                lead_expiry = parse_timestamp(request.lead_expiry)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=_error_body(
                        "INVALID_TIMESTAMP",
                        f"Invalid lead_expiry: {request.lead_expiry}",
                        request_id,
                        False,
                    ),
                )

        # ── Resolve experiment assignment (EXP-01) ─────────────────────
        if exp_assignment.is_experiment_enabled(svc.config):
            assignment = exp_assignment.assign_seller(
                svc.experiment_id, svc.salt, request.seller_id
            )
            experiment_id = svc.experiment_id
        else:
            assignment = Assignment.CONTROL.value
            experiment_id = None

        # ── Get seller state ───────────────────────────────────────────
        state_version = 0
        n_attempts = 0
        n_eff = 0
        prior_level = 0.5
        prior_weight = 1.0
        is_cold = True

        if svc.model_store is not None:
            seller_state = svc.model_store.get_state(
                request.seller_id, svc.model_compatibility_id
            )
            if seller_state is not None:
                state_version = seller_state.state_version
                n_attempts = seller_state.n
                n_eff = seller_state.n
                prior_weight = seller_state.prior_weight
                prior_level = seller_state.prior_level
                is_cold = seller_state.is_cold_start

        # ── Generate candidates ────────────────────────────────────────
        from btc.model.policy import (
            generate_candidates,
            recommend as policy_recommend,
            CandidateSet,
        )
        from btc.model.posterior import Posterior, compute_posterior

        now = _utc_now()
        support_mask = {}  # Empty support mask — all bins supported

        candidate_set = generate_candidates(
            earliest_at=earliest_at,
            latest_at=latest_at,
            calendar=svc.config.calendar,
            support_mask=support_mask,
            clock=lambda: now,
            max_calls_per_day=svc.config.max_calls_per_seller_per_day,
            calls_already_today=request.calls_already_today,
            lead_expiry=lead_expiry,
            dispatch_lead_time_seconds=svc.config.dispatch_lead_time_seconds,
            initial_delay_minutes=svc.config.initial_delay_maximum_minutes,
            lead_sent_time=lead_sent_time,
        )

        # ── No candidates → NO_ELIGIBLE_SLOT ───────────────────────────
        if not candidate_set.timestamps:
            response = _build_recommend_response(
                decision_id=str(uuid.uuid4()),
                request=request,
                status=DecisionStatus.NO_ELIGIBLE_SLOT.value,
                scheduled_at=None,
                secondary_at=None,
                reason_code="NO_ELIGIBLE_SLOT",
                mode=PolicyMode.NONE.value,
                assignment=assignment,
                experiment_id=experiment_id,
                state_version=state_version,
                n_attempts=n_attempts,
                n_eff=n_eff,
                prior_level=prior_level,
                prior_weight=prior_weight,
                expected_reward=0.0,
                latent_std=0.0,
                predictive_std=0.0,
                candidate_count=0,
                action_probability=None,
                assignment_probability=None,
                ope_eligible=False,
                valid_until=None,
                created_at=now,
                state=svc,
            )

            # Log the decision
            try:
                _log_to_experiment_store(
                    state=svc,
                    request=request,
                    response=response,
                    candidate_timestamps=[],
                    action_probability=None,
                    assignment=assignment,
                    experiment_id=experiment_id,
                    state_version=state_version,
                    clock=lambda: now,
                )
            except exp_logging.DecisionConflictError:
                # Idempotent — return the existing decision
                pass

            svc.metrics.record_decision(
                status=DecisionStatus.NO_ELIGIBLE_SLOT.value,
                assignment=assignment,
                mode=PolicyMode.NONE.value,
            )
            return response

        # ── Build posterior ────────────────────────────────────────────
        from btc.model.stats import Prior, SellerState

        prior = svc.model_store.get_prior() if svc.model_store else Prior.diagonal_prior(
            d=2 * svc.config.model.k + 1, alpha=svc.config.model.alpha
        )

        if is_cold:
            # Cold start: use prior directly
            posterior = Posterior(
                mu=prior.mu.copy(),
                L=prior.L.copy(),
                sigma2=svc.config.model.reward_config.sigma2,
                prior_weight=1.0,
                n=0,
                d=prior.d,
            )
        else:
            seller_state = svc.model_store.get_state(
                request.seller_id, svc.model_compatibility_id
            )
            posterior = compute_posterior(
                seller_state, prior, svc.config.model.reward_config.sigma2
            )

        # ── Select action ──────────────────────────────────────────────
        import numpy as np

        rng = np.random.default_rng(hash(request.seller_id) ^ hash(now.isoformat()))

        policy_decision = policy_recommend(
            candidates=candidate_set,
            posterior=posterior,
            assignment=assignment,
            sigma2=svc.config.model.reward_config.sigma2,
            rng=rng,
            clock=lambda: now,
        )

        # ── Compute valid_until ────────────────────────────────────────
        valid_until = None
        if policy_decision.scheduled_at is not None:
            tolerance = timedelta(minutes=svc.config.decision_execution_tolerance_minutes)
            valid_until = policy_decision.scheduled_at + tolerance
            if lead_expiry is not None and lead_expiry < valid_until:
                valid_until = lead_expiry

        # ── Build response ─────────────────────────────────────────────
        response = _build_recommend_response(
            decision_id=policy_decision.decision_id,
            request=request,
            status=policy_decision.status,
            scheduled_at=policy_decision.scheduled_at,
            secondary_at=policy_decision.secondary_at,
            reason_code=policy_decision.reason_code,
            mode=policy_decision.mode,
            assignment=policy_decision.assignment,
            experiment_id=experiment_id,
            state_version=state_version,
            n_attempts=policy_decision.n_attempts,
            n_eff=n_eff,
            prior_level=prior_level,
            prior_weight=policy_decision.prior_weight,
            expected_reward=policy_decision.expected_reward,
            latent_std=policy_decision.latent_std,
            predictive_std=policy_decision.predictive_std,
            candidate_count=policy_decision.candidate_count,
            action_probability=policy_decision.action_probability,
            assignment_probability=policy_decision.assignment_probability,
            ope_eligible=policy_decision.ope_eligible,
            valid_until=valid_until,
            created_at=now,
            state=svc,
        )

        # ── Log to experiment store (EXP-03) ───────────────────────────
        candidate_timestamps = [
            ts.isoformat() for ts in candidate_set.timestamps
        ]

        try:
            _log_to_experiment_store(
                state=svc,
                request=request,
                response=response,
                candidate_timestamps=candidate_timestamps,
                action_probability=policy_decision.action_probability,
                assignment=policy_decision.assignment,
                experiment_id=experiment_id,
                state_version=state_version,
                clock=lambda: now,
            )
        except exp_logging.DecisionConflictError:
            # Idempotent — return the existing decision
            pass

        svc.metrics.record_decision(
            status=policy_decision.status,
            assignment=policy_decision.assignment,
            mode=policy_decision.mode,
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Recommendation failed for request_id=%s: %s", request_id, exc)
        raise HTTPException(
            status_code=503,
            detail=_error_body(
                "RECOMMENDATION_FAILED",
                f"Recommendation failed: {exc}",
                request_id,
                True,
            ),
        )


@app.post(
    "/v1/outcomes",
    response_model=OutcomeResponse,
    responses={
        409: {"model": dict, "description": "Revision conflict"},
        503: {"model": dict, "description": "Database unavailable"},
    },
    tags=["Outcomes"],
)
async def handle_outcome(request: OutcomeRequest) -> OutcomeResponse:
    """STATE-01: Record a finalised outcome.

    Processes the outcome atomically: validates, computes contribution,
    updates seller state, and appends outbox record.

    Parameters
    ----------
    request : OutcomeRequest
        Finalised outcome data.

    Returns
    -------
    OutcomeResponse
        Status (APPLIED/DUPLICATE/STALE_REVISION) and state metadata.

    Raises
    ------
    HTTPException
        409: Same revision with different payload.
        503: Database unavailable.
    """
    svc = _get_state()
    request_id = ""

    try:
        # ── Parse timestamps ───────────────────────────────────────────
        def _parse_field(field_name: str) -> datetime:
            val = getattr(request, field_name)
            try:
                return parse_timestamp(val)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=_error_body(
                        "INVALID_TIMESTAMP",
                        f"Invalid {field_name}: {val}",
                        request_id,
                        False,
                    ),
                )

        finalized_at = _parse_field("finalized_at")
        call_start_time = _parse_field("call_start_time")
        call_end_time = _parse_field("call_end_time")
        lead_sent_time = _parse_field("lead_sent_time")

        requested_callback_at = None
        if request.requested_callback_at:
            requested_callback_at = parse_timestamp(request.requested_callback_at)

        # ── Build outcome dict ─────────────────────────────────────────
        outcome = {
            "seller_id": request.seller_id,
            "lead_id": request.lead_id,
            "attempt_id": request.attempt_id,
            "source": request.source,
            "revision": request.revision,
            "event_id": request.event_id,
            "finalized_at": finalized_at,
            "call_start_time": call_start_time,
            "call_end_time": call_end_time,
            "lead_sent_time": lead_sent_time,
            "attempt_number": request.attempt_number,
            "answered": request.answered,
            "disposition": request.disposition,
            "meeting_fixed": request.meeting_fixed,
            "requested_callback_at": requested_callback_at,
            "duration_s": request.duration_s,
            "dialer_version": request.dialer_version,
            "source_bucket": request.source_bucket,
        }

        # ── Process outcome ────────────────────────────────────────────
        if svc.store is not None:
            from btc.store.transactions import record_outcome

            result = record_outcome(
                conn=svc.store,
                outcome=outcome,
                seller_state_store=svc.model_store,
                model_config=svc.config.model,
                reward_config=svc.config.model.reward_config,
                model_compatibility_id=svc.model_compatibility_id,
            )

            svc.metrics.record_outcome()

            return OutcomeResponse(
                status=result.status,
                state_version=result.state_version,
                attempt_id=result.attempt_id,
                revision=result.revision,
                n_attempts=result.n_attempts,
            )
        else:
            # No database — return a synthetic applied result
            svc.metrics.record_outcome()
            return OutcomeResponse(
                status="APPLIED",
                state_version=0,
                attempt_id=request.attempt_id,
                revision=request.revision,
                n_attempts=0,
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Outcome processing failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=_error_body(
                "OUTCOME_PROCESSING_FAILED",
                f"Outcome processing failed: {exc}",
                request_id,
                True,
            ),
        )


@app.post(
    "/v1/retries",
    response_model=RetryResponse,
    responses={
        409: {"model": dict, "description": "Revision conflict"},
        503: {"model": dict, "description": "Required dependency unavailable"},
    },
    tags=["Retries"],
)
async def handle_retry(request: RetryRequest) -> RetryResponse:
    """RET-04: Generate a retry decision.

    Reads the authoritative outcome from the database, applies retry
    rules, and returns a persisted retry result.

    Parameters
    ----------
    request : RetryRequest
        Retry request body.

    Returns
    -------
    RetryResponse
        Retry decision result.

    Raises
    ------
    HTTPException
        409: Revision conflict.
        503: Required dependency unavailable.
    """
    svc = _get_state()
    request_id = request.request_id

    try:
        svc.metrics.record_retry()

        # ── Read latest outcome from database ──────────────────────────
        if svc.store is not None:
            row = svc.store.fetchone(
                "SELECT * FROM attempt_latest WHERE source = %s AND attempt_id = %s ORDER BY revision DESC LIMIT 1",
                (request.source, request.attempt_id),
            )
        else:
            row = None

        if row is None:
            # No outcome found — cannot determine retry
            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.STOP.value,
                scheduled_at=None,
                reason_code="NO_OUTCOME_FOUND",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        # Check expected revision for optimistic concurrency
        existing_revision = row.get("revision", 0)
        if existing_revision != request.expected_revision:
            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.SUPERSEDED.value,
                scheduled_at=None,
                reason_code=f"REVISION_MISMATCH: expected={request.expected_revision}, existing={existing_revision}",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        # ── Apply retry rules ──────────────────────────────────────────
        disposition = row.get("disposition", "")
        call_end_time_str = row.get("call_end_time")
        attempt_number = row.get("attempt_number", 1)

        now = _utc_now()

        if disposition in ("MEETING_FIXED", "NOT_INTERESTED", "GENERAL"):
            # Terminal dispositions — no automatic retry
            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.STOP.value,
                scheduled_at=None,
                reason_code=f"TERMINAL_DISPOSITION:{disposition}",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        if disposition == "UNKNOWN":
            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.MANUAL_REVIEW.value,
                scheduled_at=None,
                reason_code="UNKNOWN_DISPOSITION",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        # Parse call_end_time for delay calculations
        call_end_time = None
        if call_end_time_str:
            try:
                call_end_time = parse_timestamp(call_end_time_str)
            except ValueError:
                pass

        if call_end_time is None:
            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.MANUAL_REVIEW.value,
                scheduled_at=None,
                reason_code="INVALID_CALL_END_TIME",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        # Not Answered — first attempt
        if disposition == "NOT_ANSWERED" and attempt_number == 1:
            proposed = call_end_time + timedelta(minutes=15)
            if proposed <= now:
                proposed = now + timedelta(
                    seconds=svc.config.dispatch_lead_time_seconds
                    + svc.config.minimum_inter_call_gap_minutes * 60
                )
            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.SCHEDULED.value,
                scheduled_at=proposed.isoformat(),
                reason_code="NOT_ANSWERED_FIRST_RETRY",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        # Not Answered — subsequent attempts
        if disposition == "NOT_ANSWERED" and attempt_number >= 2:
            # Next eligible working day
            from btc.model.policy import _to_kolkata, _is_calendar_day, _is_in_calendar_window

            candidate = call_end_time
            # Move to next day
            candidate = candidate.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            # Find next calendar-allowed day
            for _ in range(7):
                local = _to_kolkata(candidate)
                if _is_calendar_day(local, svc.config.calendar) and _is_in_calendar_window(local, svc.config.calendar):
                    break
                candidate += timedelta(days=1)

            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.SCHEDULED.value,
                scheduled_at=candidate.isoformat(),
                reason_code=f"NOT_ANSWERED_SUBSEQUENT:attempt={attempt_number}",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        # Call Later / Busy — use explicit callback or propose
        if disposition == "CALL_LATER_BUSY":
            callback = row.get("requested_callback_at")
            if callback:
                if isinstance(callback, str):
                    try:
                        callback = parse_timestamp(callback)
                    except ValueError:
                        callback = None
                if callback and callback > now:
                    return RetryResponse(
                        decision_id=str(uuid.uuid4()),
                        request_id=request.request_id,
                        source=request.source,
                        attempt_id=request.attempt_id,
                        result=RetryResult.SCHEDULED.value,
                        scheduled_at=callback.isoformat(),
                        reason_code="EXPLICIT_CALLBACK",
                        policy_version=svc.policy_version,
                        superseded_by=None,
                    )

            # No callback — propose call_end + 120m
            proposed = call_end_time + timedelta(minutes=2)
            if proposed <= now:
                proposed = now + timedelta(seconds=svc.config.dispatch_lead_time_seconds)
            return RetryResponse(
                decision_id=str(uuid.uuid4()),
                request_id=request.request_id,
                source=request.source,
                attempt_id=request.attempt_id,
                result=RetryResult.SCHEDULED.value,
                scheduled_at=proposed.isoformat(),
                reason_code="CALL_LATER_NO_CALLBACK",
                policy_version=svc.policy_version,
                superseded_by=None,
            )

        # Default — manual review
        return RetryResponse(
            decision_id=str(uuid.uuid4()),
            request_id=request.request_id,
            source=request.source,
            attempt_id=request.attempt_id,
            result=RetryResult.MANUAL_REVIEW.value,
            scheduled_at=None,
            reason_code=f"UNHANDLED_DISPOSITION:{disposition}",
            policy_version=svc.policy_version,
            superseded_by=None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Retry processing failed for request_id=%s: %s", request_id, exc)
        raise HTTPException(
            status_code=503,
            detail=_error_body(
                "RETRY_PROCESSING_FAILED",
                f"Retry processing failed: {exc}",
                request_id,
                True,
            ),
        )


@app.get("/healthz", tags=["Health"])
async def health() -> dict:
    """Process liveness check.

    Returns a simple pong to indicate the process is alive.
    Does NOT expose seller data or credentials (API-01).

    Returns
    -------
    dict
        {status: "ok"}
    """
    return {"status": "ok"}


@app.get("/readyz", tags=["Health"])
async def ready() -> dict:
    """Readiness check.

    Reports readiness of model, config, and required stores/adapters.
    Does NOT expose seller data or credentials (API-01).

    Returns
    -------
    dict
        Readiness status with component checks.
    """
    svc = _get_state()

    checks = {
        "config": True,
        "model_store": svc.model_store is not None,
        "database": svc.store is not None,
    }

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }


@app.get("/metrics", tags=["Operational"])
async def metrics() -> dict:
    """Operational metrics endpoint.

    Returns system-level metrics without high-cardinality seller labels
    (SRS §10.1).

    Returns
    -------
    dict
        Metrics including request counts, status distribution,
        assignment distribution, and uptime.
    """
    svc = _get_state()
    return svc.metrics.snapshot()


# ── Error body helper ──────────────────────────────────────────────────────────


def _error_body(
    error_code: str,
    message: str,
    request_id: str,
    retryable: bool,
) -> dict:
    """Create an error body dict (API-02).

    Parameters
    ----------
    error_code : str
        Machine-readable error code.
    message : str
        Human-readable message.
    request_id : str
        Request identifier.
    retryable : bool
        Whether the request is retryable.

    Returns
    -------
    dict
        Error body dictionary.
    """
    return ErrorResponseBody(
        error_code=error_code,
        message=message,
        request_id=request_id,
        retryable=retryable,
    ).model_dump()
