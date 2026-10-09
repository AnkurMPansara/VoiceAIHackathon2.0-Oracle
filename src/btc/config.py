"""Configuration loader, validator, and defaults for the Best Time to Call system.

Implements SRS §14 (CFG-01) configuration requirements:
- Runtime mode, timezone, calendar, delay, and cap settings
- Reward model parameters (weights, costs, noise variance)
- Bayesian model hyperparameters (K, gamma, smoothness, prior)
- Startup validation with specific error codes
- Content hashing for bundle compatibility tracking

All defaults marked "development only" per SRS §2.2 and §14.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional

import yaml


# ── Error types ──────────────────────────────────────────────────────────────

class ConfigError(ValueError):
    """Base exception for configuration validation failures.

    Each subclass encodes an ``error_code`` suitable for SRS §10.2 error
    response payloads (422 schema violations).
    """

    def __init__(self, message: str, error_code: str = "CONFIG_ERROR") -> None:
        super().__init__(message)
        self.error_code = error_code


class RuntimeModeError(ConfigError):
    """``runtime_mode`` must be ``shadow`` or ``live``."""

    def __init__(self, value: Any) -> None:
        super().__init__(
            f"runtime_mode must be 'shadow' or 'live', got {value!r}",
            error_code="INVALID_RUNTIME_MODE",
        )


class GammaError(ConfigError):
    """``gamma`` must equal 1.0 in release 1 (SRS MOD-04)."""

    def __init__(self, value: Any) -> None:
        super().__init__(
            f"gamma must be 1.0 in release 1, got {value!r}",
            error_code="UNSUPPORTED_GAMMA",
        )


class Sigma2Error(ConfigError):
    """``sigma2`` must be strictly positive (SRS MOD-03)."""

    def __init__(self, value: Any) -> None:
        super().__init__(
            f"sigma2 must be > 0, got {value!r}",
            error_code="ZERO_OR_NEGATIVE_SIGMA2",
        )


class KError(ConfigError):
    """``k`` must be in {2, 3, 4} (SRS MOD-02, §14)."""

    def __init__(self, value: Any) -> None:
        super().__init__(
            f"k (basis order) must be in [2, 3, 4], got {value!r}",
            error_code="INVALID_K",
        )


class WeightError(ConfigError):
    """A weight or cost parameter is negative (SRS MOD-01, §14)."""

    def __init__(self, param: str, value: Any) -> None:
        super().__init__(
            f"{param} must be non-negative, got {value!r}",
            error_code="NEGATIVE_WEIGHT_OR_COST",
        )


class CalendarError(ConfigError):
    """Calendar window is inconsistent (SRS §14, POL-02)."""

    def __init__(self, message: str, error_code: str = "INVALID_CALENDAR") -> None:
        super().__init__(message, error_code=error_code)


class DuplicateHolidayError(ConfigError):
    """Holidays list contains duplicate dates (SRS §14)."""

    def __init__(self, date: str) -> None:
        super().__init__(
            f"Duplicate holiday date: {date}",
            error_code="DUPLICATE_HOLIDAY",
        )


class UnknownKeyError(ConfigError):
    """YAML contains keys not recognised by the schema (SRS §14)."""

    def __init__(self, key: str, parent: str = "root") -> None:
        super().__init__(
            f"Unknown configuration key '{key}' in {parent}",
            error_code="UNKNOWN_CONFIG_KEY",
        )


# ── RewardConfig ─────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """Reward model parameters (SRS MOD-01).

    Computes a utility score for each finalised attempt:

        y = w_meeting * meeting_fixed
          + w_answered * answered
          - c_dial
          - w_not_interested * I(disposition == NOT_INTERESTED)

    Defaults: (1.0, 0.02, 0.02, 0.0) — development only (SRS §2.2).
    """

    w_meeting: float = 1.0
    """Weight for a meeting-fixed outcome."""

    w_answered: float = 0.1
    """Weight for any answered call."""

    c_dial: float = 0.02
    """Cost per dial attempt."""

    w_not_interested: float = 0.0
    """Penalty for NOT_INTERESTED disposition."""

    sigma2: float = 0.06
    """Gaussian working noise variance; must be > 0 (SRS MOD-03)."""

    def __post_init__(self) -> None:
        """Validate reward parameters (SRS MOD-01, MOD-03, §14).

        Floors sigma2 at 1e-4 per SRS TRAIN-04.
        """
        if self.w_meeting < 0:
            raise WeightError("w_meeting", self.w_meeting)
        if self.w_answered < 0:
            raise WeightError("w_answered", self.w_answered)
        if self.c_dial < 0:
            raise WeightError("c_dial", self.c_dial)
        if self.w_not_interested < 0:
            raise WeightError("w_not_interested", self.w_not_interested)
        if self.sigma2 <= 0:
            raise Sigma2Error(self.sigma2)
        # SRS TRAIN-04: floor at 1e-4
        if self.sigma2 < 1e-4:
            self.sigma2 = 1e-4


# ── ModelConfig ──────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Bayesian model hyperparameters (SRS MOD-02 through TRAIN-05).

    Attributes
    ----------
    k : int
        Fourier basis order. Candidates {2, 3, 4}; development default 4.
    reward_config : RewardConfig
        Reward parameters used to build seller sufficient statistics.
    gamma : float
        Discount factor. MUST equal 1.0 in release 1 (SRS MOD-04).
    lambda_smooth : float
        Smoothness penalty on Fourier coefficients (SRS TRAIN-02).
    lambda_parent : float
        Penalty for deviation from parent segment mean (SRS TRAIN-02).
    alpha : float
        Shrinkage scale for the regularised covariance prior (SRS TRAIN-03).
    segment_min_attempts : int
        Minimum finalised attempts for a segment to be eligible (SRS TRAIN-01).
    segment_min_sellers : int
        Minimum distinct sellers for a segment to be eligible (SRS TRAIN-01).
    support_bin_min_attempts : int
        Minimum attempts in a 15-minute support bin (SRS TRAIN-05).
    support_bin_min_sellers : int
        Minimum distinct sellers in a 15-minute support bin (SRS TRAIN-05).
    state_history_start : str
        Inclusive start of the state-history interval (ISO date, SRS TRAIN-06).
    state_history_end : str
        Exclusive end of the state-history interval (ISO date, SRS TRAIN-06).
    """

    k: int = 4
    reward_config: RewardConfig = field(default_factory=lambda: RewardConfig())
    gamma: float = 1.0
    lambda_smooth: float = 1.0
    lambda_parent: float = 10.0
    alpha: float = 0.1
    segment_min_attempts: int = 2000
    segment_min_sellers: int = 200
    support_bin_min_attempts: int = 50
    support_bin_min_sellers: int = 30
    state_history_start: str = "2026-04-01"
    state_history_end: str = "2026-10-01"

    def __post_init__(self) -> None:
        """Validate model hyperparameters (SRS MOD-02, MOD-04, §14)."""
        if self.gamma != 1.0:
            raise GammaError(self.gamma)
        if self.k not in (2, 3, 4):
            raise KError(self.k)
        # Validate nested reward config
        self.reward_config.__post_init__()


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Calendar:
    """Business calendar window (SRS §14, POL-02).

    Attributes
    ----------
    days_of_week : list[int]
        Allowed weekdays as ISO numbers (0=Mon … 6=Sun).
    start_hour : int
        Inclusive start of the calling window (0–23).
    end_hour : int
        Exclusive end of the calling window (0–23).
    holidays : list[str]
        ISO date strings (YYYY-MM-DD) excluded from the calendar.
    """

    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    start_hour: int = 8
    end_hour: int = 18
    holidays: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate calendar constraints (SRS §14, POL-02)."""
        if self.start_hour >= self.end_hour:
            raise CalendarError(
                f"calendar.start_hour ({self.start_hour}) must be < "
                f"calendar.end_hour ({self.end_hour})",
                error_code="INVALID_CALENDAR_WINDOW",
            )
        # Check for duplicate holidays
        seen: set[str] = set()
        for h in self.holidays:
            if h in seen:
                raise DuplicateHolidayError(h)
            seen.add(h)


@dataclass
class Config:
    """Top-level configuration (SRS §14, CFG-01).

    All fields are validated on construction via ``__post_init__`` and
    separately by ``validate_config`` for YAML-loaded instances.

    Attributes
    ----------
    runtime_mode : str
        ``"shadow"`` or ``"live"``. Default ``"shadow"`` (development only).
    timezone : str
        Business timezone for calendar/candidate calculations.
    calendar : Calendar
        Allowed calling days, hours, and holidays.
    initial_delay_maximum_minutes : int
        Maximum initial delay before first call (minutes).
    expiry_when_upstream_absent : str | None
        Lead expiry when upstream does not supply one.
    max_calls_per_seller_per_day : int
        Hard cap on calls per seller per local day.
    max_attempts_per_lead : int
        Maximum retry attempts per seller–lead pair.
    minimum_inter_call_gap_minutes : int
        Minimum gap between consecutive calls (minutes).
    dispatch_lead_time_seconds : int
        Scheduler dispatch lead time (seconds).
    decision_execution_tolerance_minutes : int
        Dispatch execution tolerance window (minutes).
    cache_staleness_seconds : int
        Maximum acceptable cache age before revalidation (seconds).
    experiment_enabled : bool
        Whether the experiment arm is active.
    double_call_enabled : bool
        Whether the double-call experiment is active.
    terminal_retry_dispositions : list[str]
        Dispositions that stop automatic retries.
    retention_horizon_days : int
        Retention/replay horizon for deduplication (days).
    model : ModelConfig
        Bayesian model hyperparameters.
    """

    runtime_mode: str = "shadow"
    timezone: str = "Asia/Kolkata"
    calendar: Calendar = field(default_factory=Calendar)
    initial_delay_maximum_minutes: int = 15
    expiry_when_upstream_absent: Optional[str] = None
    max_calls_per_seller_per_day: int = 3
    max_attempts_per_lead: int = 5
    minimum_inter_call_gap_minutes: int = 15
    dispatch_lead_time_seconds: int = 5
    decision_execution_tolerance_minutes: int = 5
    cache_staleness_seconds: int = 5
    experiment_enabled: bool = False
    double_call_enabled: bool = False
    terminal_retry_dispositions: List[str] = field(
        default_factory=lambda: ["Meeting Fixed", "Not Interested", "General"]
    )
    retention_horizon_days: int = 90
    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    def __post_init__(self) -> None:
        """Validate all configuration constraints (SRS §14, CFG-01)."""
        validate_config(self)


# ── YAML loading ─────────────────────────────────────────────────────────────

def _unwrap_calendar(data: Dict[str, Any]) -> Calendar:
    """Convert a YAML dict into a Calendar instance."""
    return Calendar(
        days_of_week=data.get("days_of_week", [0, 1, 2, 3, 4, 5]),
        start_hour=data.get("start_hour", 8),
        end_hour=data.get("end_hour", 18),
        holidays=data.get("holidays", []),
    )


def _unwrap_reward_config(data: Dict[str, Any]) -> RewardConfig:
    """Convert a YAML dict into a RewardConfig instance."""
    return RewardConfig(
        w_meeting=data.get("w_meeting", 1.0),
        w_answered=data.get("w_answered", 0.1),
        c_dial=data.get("c_dial", 0.02),
        w_not_interested=data.get("w_not_interested", 0.0),
        sigma2=data.get("sigma2", 0.06),
    )


def _unwrap_model_config(data: Dict[str, Any]) -> ModelConfig:
    """Convert a YAML dict into a ModelConfig instance."""
    reward_data = data.get("reward_config", {})
    reward_config = _unwrap_reward_config(reward_data) if isinstance(reward_data, dict) else RewardConfig()

    return ModelConfig(
        k=data.get("k", 4),
        reward_config=reward_config,
        gamma=data.get("gamma", 1.0),
        lambda_smooth=data.get("lambda_smooth", 1.0),
        lambda_parent=data.get("lambda_parent", 10.0),
        alpha=data.get("alpha", 0.1),
        segment_min_attempts=data.get("segment_min_attempts", 2000),
        segment_min_sellers=data.get("segment_min_sellers", 200),
        support_bin_min_attempts=data.get("support_bin_min_attempts", 50),
        support_bin_min_sellers=data.get("support_bin_min_sellers", 30),
        state_history_start=data.get("state_history_start", "2026-04-01"),
        state_history_end=data.get("state_history_end", "2026-10-01"),
    )


# Known top-level keys for strict YAML parsing (SRS §14 — reject unknown keys)
_KNOWN_TOP_KEYS = frozenset({
    "runtime_mode", "timezone", "calendar", "initial_delay_maximum_minutes",
    "expiry_when_upstream_absent", "max_calls_per_seller_per_day",
    "max_attempts_per_lead", "minimum_inter_call_gap_minutes",
    "dispatch_lead_time_seconds", "decision_execution_tolerance_minutes",
    "cache_staleness_seconds", "experiment_enabled", "double_call_enabled",
    "terminal_retry_dispositions", "retention_horizon_days", "model",
})

_KNOWN_CALENDAR_KEYS = frozenset({"days_of_week", "start_hour", "end_hour", "holidays"})

_KNOWN_MODEL_KEYS = frozenset({
    "k", "reward_config", "gamma", "lambda_smooth", "lambda_parent", "alpha",
    "segment_min_attempts", "segment_min_sellers", "support_bin_min_attempts",
    "support_bin_min_sellers", "state_history_start", "state_history_end",
})

_KNOWN_REWARD_KEYS = frozenset({
    "w_meeting", "w_answered", "c_dial", "w_not_interested", "sigma2",
})


def _check_unknown_keys(data: Dict[str, Any], parent: str) -> None:
    """Raise UnknownKeyError for any key not in the expected schema."""
    for key in data:
        if key not in _KNOWN_TOP_KEYS and parent == "root":
            raise UnknownKeyError(key, parent)
        if key not in _KNOWN_CALENDAR_KEYS and parent == "calendar":
            raise UnknownKeyError(key, parent)
        if key not in _KNOWN_MODEL_KEYS and parent == "model":
            raise UnknownKeyError(key, parent)
        if key not in _KNOWN_REWARD_KEYS and parent == "reward_config":
            raise UnknownKeyError(key, parent)


def load_config(path: str) -> Config:
    """Load configuration from a YAML file and validate it (SRS §14).

    Parameters
    ----------
    path : str
        File system path to the YAML configuration file.

    Returns
    -------
    Config
        Validated configuration instance.

    Raises
    ------
    ConfigError
        If any configuration value violates SRS constraints.
    FileNotFoundError
        If the YAML file does not exist.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Configuration file must contain a YAML mapping at the top level")

    _check_unknown_keys(raw, "root")

    # Calendar
    cal_data = raw.get("calendar", {})
    if isinstance(cal_data, dict):
        _check_unknown_keys(cal_data, "calendar")
        calendar = _unwrap_calendar(cal_data)
    else:
        calendar = Calendar()

    # Model
    model_data = raw.get("model", {})
    if isinstance(model_data, dict):
        _check_unknown_keys(model_data, "model")
        reward_data = model_data.get("reward_config", {})
        if isinstance(reward_data, dict):
            _check_unknown_keys(reward_data, "reward_config")
        model = _unwrap_model_config(model_data)
    else:
        model = ModelConfig()

    config = Config(
        runtime_mode=raw.get("runtime_mode", "shadow"),
        timezone=raw.get("timezone", "Asia/Kolkata"),
        calendar=calendar,
        initial_delay_maximum_minutes=raw.get("initial_delay_maximum_minutes", 15),
        expiry_when_upstream_absent=raw.get("expiry_when_upstream_absent"),
        max_calls_per_seller_per_day=raw.get("max_calls_per_seller_per_day", 3),
        max_attempts_per_lead=raw.get("max_attempts_per_lead", 5),
        minimum_inter_call_gap_minutes=raw.get("minimum_inter_call_gap_minutes", 15),
        dispatch_lead_time_seconds=raw.get("dispatch_lead_time_seconds", 5),
        decision_execution_tolerance_minutes=raw.get("decision_execution_tolerance_minutes", 5),
        cache_staleness_seconds=raw.get("cache_staleness_seconds", 5),
        experiment_enabled=raw.get("experiment_enabled", False),
        double_call_enabled=raw.get("double_call_enabled", False),
        terminal_retry_dispositions=raw.get("terminal_retry_dispositions",
                                           ["Meeting Fixed", "Not Interested", "General"]),
        retention_horizon_days=raw.get("retention_horizon_days", 90),
        model=model,
    )

    validate_config(config)
    return config


# ── Validation ───────────────────────────────────────────────────────────────

def validate_config(config: Config) -> None:
    """Validate all configuration constraints (SRS §14, CFG-01).

    Checks every constraint listed in CFG-01 and raises a
    :class:`ConfigError` subclass with a specific ``error_code`` for
    each violation.

    Parameters
    ----------
    config : Config
        The configuration to validate.

    Raises
    ------
    RuntimeModeError
        ``runtime_mode`` is not ``"shadow"`` or ``"live"``.
    GammaError
        ``gamma`` is not 1.0.
    Sigma2Error
        ``sigma2`` is not strictly positive.
    KError
        ``k`` is not in {2, 3, 4}.
    WeightError
        A weight or cost is negative.
    CalendarError
        ``start_hour >= end_hour``.
    DuplicateHolidayError
        Duplicate holiday dates.
    """
    # runtime_mode
    if config.runtime_mode not in ("shadow", "live"):
        raise RuntimeModeError(config.runtime_mode)

    # Model-level validations
    model = config.model
    if model.gamma != 1.0:
        raise GammaError(model.gamma)
    if model.k not in (2, 3, 4):
        raise KError(model.k)

    # Reward params
    rc = model.reward_config
    if rc.w_meeting < 0:
        raise WeightError("w_meeting", rc.w_meeting)
    if rc.w_answered < 0:
        raise WeightError("w_answered", rc.w_answered)
    if rc.c_dial < 0:
        raise WeightError("c_dial", rc.c_dial)
    if rc.w_not_interested < 0:
        raise WeightError("w_not_interested", rc.w_not_interested)
    if rc.sigma2 <= 0:
        raise Sigma2Error(rc.sigma2)

    # Calendar
    cal = config.calendar
    if cal.start_hour >= cal.end_hour:
        raise CalendarError(
            f"calendar.start_hour ({cal.start_hour}) must be < "
            f"calendar.end_hour ({cal.end_hour})",
            error_code="INVALID_CALENDAR_WINDOW",
        )
    seen_holidays: set[str] = set()
    for h in cal.holidays:
        if h in seen_holidays:
            raise DuplicateHolidayError(h)
        seen_holidays.add(h)


# ── Hashing ──────────────────────────────────────────────────────────────────

def _serialise(config: Config) -> Dict[str, Any]:
    """Convert a Config instance to a plain dict (dataclass → dict)."""
    result: Dict[str, Any] = {}
    for f in dataclass_fields(config):
        val = getattr(config, f.name)
        if isinstance(val, (Calendar, RewardConfig, ModelConfig)):
            result[f.name] = _serialise_nested(val)
        elif isinstance(val, list):
            result[f.name] = list(val)
        else:
            result[f.name] = val
    return result


def _serialise_nested(obj: Any) -> Dict[str, Any]:
    """Convert a nested dataclass to a plain dict."""
    if not hasattr(obj, "__dataclass_fields__"):
        return obj
    result: Dict[str, Any] = {}
    for f in obj.__dataclass_fields__.values():
        val = getattr(obj, f.name)
        if hasattr(val, "__dataclass_fields__"):
            result[f.name] = _serialise_nested(val)
        elif isinstance(val, list):
            result[f.name] = list(val)
        else:
            result[f.name] = val
    return result


def config_hash(config: Config) -> str:
    """Compute a SHA-256 content hash of the configuration (SRS §14).

    Serialises the configuration to canonical JSON (sorted keys,
    no whitespace) and returns the hex digest.  This hash is used
    to detect configuration changes that require model rebuilds
    (SRS TRAIN-08).

    Parameters
    ----------
    config : Config
        The configuration to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    data = _serialise(config)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Defaults ─────────────────────────────────────────────────────────────────

def default_config() -> Config:
    """Return a Config instance with all SRS §14 development defaults.

    Returns
    -------
    Config
        Validated default configuration.
    """
    return Config()


def default_reward_config() -> RewardConfig:
    """Return a RewardConfig with SRS MOD-01 defaults.

    Returns
    -------
    RewardConfig
        (w_meeting=1.0, w_answered=0.1, c_dial=0.02, w_not_interested=0.0, sigma2=0.06)
    """
    return RewardConfig()


def default_model_config() -> ModelConfig:
    """Return a ModelConfig with SRS MOD-02 / TRAIN-01–05 defaults.

    Returns
    -------
    ModelConfig
        (k=4, gamma=1.0, lambda_smooth=1.0, lambda_parent=10.0, alpha=0.1, …)
    """
    return ModelConfig()
