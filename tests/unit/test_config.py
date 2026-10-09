"""Comprehensive unit tests for src/btc/config.py — SRS §14 (CFG-01).

Tests cover:
  - All default values (CFG-01 table)
  - Every validation rule (invalid runtime_mode, gamma, sigma2, k, weights/costs,
    calendar window, duplicate holidays, unknown YAML keys)
  - config_hash consistency
  - YAML round-trip load
  - sigma2 flooring at 1e-4 (SRS TRAIN-04)
  - ModelConfig with k=2 valid path
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
import yaml

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from btc.config import (
    Calendar,
    Config,
    ConfigError,
    DuplicateHolidayError,
    GammaError,
    KError,
    ModelConfig,
    RewardConfig,
    RuntimeModeError,
    Sigma2Error,
    UnknownKeyError,
    WeightError,
    config_hash,
    default_config,
    default_model_config,
    default_reward_config,
    load_config,
    validate_config,
)


# ── 1. default_config() returns all correct defaults ─────────────────────────

class TestDefaultConfig:
    def test_runtime_mode(self):
        cfg = default_config()
        assert cfg.runtime_mode == "shadow"

    def test_timezone(self):
        cfg = default_config()
        assert cfg.timezone == "Asia/Kolkata"

    def test_calendar_days(self):
        cfg = default_config()
        assert cfg.calendar.days_of_week == [0, 1, 2, 3, 4, 5]

    def test_calendar_hours(self):
        cfg = default_config()
        assert cfg.calendar.start_hour == 8
        assert cfg.calendar.end_hour == 18

    def test_calendar_no_holidays(self):
        cfg = default_config()
        assert cfg.calendar.holidays == []

    def test_initial_delay_maximum(self):
        cfg = default_config()
        assert cfg.initial_delay_maximum_minutes == 15

    def test_max_calls_per_seller_per_day(self):
        cfg = default_config()
        assert cfg.max_calls_per_seller_per_day == 3

    def test_max_attempts_per_lead(self):
        cfg = default_config()
        assert cfg.max_attempts_per_lead == 5

    def test_minimum_inter_call_gap(self):
        cfg = default_config()
        assert cfg.minimum_inter_call_gap_minutes == 15

    def test_dispatch_lead_time(self):
        cfg = default_config()
        assert cfg.dispatch_lead_time_seconds == 5

    def test_decision_execution_tolerance(self):
        cfg = default_config()
        assert cfg.decision_execution_tolerance_minutes == 5

    def test_cache_staleness(self):
        cfg = default_config()
        assert cfg.cache_staleness_seconds == 5

    def test_experiment_disabled(self):
        cfg = default_config()
        assert cfg.experiment_enabled is False

    def test_double_call_disabled(self):
        cfg = default_config()
        assert cfg.double_call_enabled is False

    def test_terminal_retry_dispositions(self):
        cfg = default_config()
        assert cfg.terminal_retry_dispositions == [
            "Meeting Fixed", "Not Interested", "General"
        ]

    def test_retention_horizon_days(self):
        cfg = default_config()
        assert cfg.retention_horizon_days == 90

    def test_model_defaults(self):
        cfg = default_config()
        assert cfg.model.k == 4
        assert cfg.model.gamma == 1.0
        assert cfg.model.lambda_smooth == 1.0
        assert cfg.model.lambda_parent == 10.0
        assert cfg.model.alpha == 0.1

    def test_expiry_when_upstream_absent(self):
        cfg = default_config()
        assert cfg.expiry_when_upstream_absent is None


# ── 2. default_reward_config() returns correct defaults ──────────────────────

class TestDefaultRewardConfig:
    def test_w_meeting(self):
        rc = default_reward_config()
        assert rc.w_meeting == 1.0

    def test_w_answered(self):
        rc = default_reward_config()
        assert rc.w_answered == 0.1

    def test_c_dial(self):
        rc = default_reward_config()
        assert rc.c_dial == 0.02

    def test_w_not_interested(self):
        rc = default_reward_config()
        assert rc.w_not_interested == 0.0

    def test_sigma2(self):
        rc = default_reward_config()
        assert rc.sigma2 == 0.06


# ── 3. default_model_config() returns correct defaults ───────────────────────

class TestDefaultModelConfig:
    def test_k(self):
        mc = default_model_config()
        assert mc.k == 4

    def test_gamma(self):
        mc = default_model_config()
        assert mc.gamma == 1.0

    def test_lambda_smooth(self):
        mc = default_model_config()
        assert mc.lambda_smooth == 1.0

    def test_lambda_parent(self):
        mc = default_model_config()
        assert mc.lambda_parent == 10.0

    def test_alpha(self):
        mc = default_model_config()
        assert mc.alpha == 0.1

    def test_segment_min_attempts(self):
        mc = default_model_config()
        assert mc.segment_min_attempts == 2000

    def test_segment_min_sellers(self):
        mc = default_model_config()
        assert mc.segment_min_sellers == 200

    def test_support_bin_min_attempts(self):
        mc = default_model_config()
        assert mc.support_bin_min_attempts == 50

    def test_support_bin_min_sellers(self):
        mc = default_model_config()
        assert mc.support_bin_min_sellers == 30

    def test_state_history_start(self):
        mc = default_model_config()
        assert mc.state_history_start == "2026-04-01"

    def test_state_history_end(self):
        mc = default_model_config()
        assert mc.state_history_end == "2026-10-01"


# ── 4. validate_config passes with all defaults ─────────────────────────────

class TestValidateConfigPass:
    def test_defaults_pass(self):
        cfg = default_config()
        validate_config(cfg)  # should not raise

    def test_explicit_valid_config(self):
        cfg = Config(
            runtime_mode="shadow",
            timezone="Asia/Kolkata",
            model=ModelConfig(k=3, gamma=1.0),
        )
        validate_config(cfg)


# ── 5. validate_config rejects invalid runtime_mode ─────────────────────────

class TestValidateRuntimeMode:
    def test_rejects_shadow_is_valid(self):
        cfg = default_config()
        cfg.runtime_mode = "shadow"
        validate_config(cfg)  # should pass

    def test_rejects_live_is_valid(self):
        cfg = default_config()
        cfg.runtime_mode = "live"
        validate_config(cfg)  # should pass

    def test_rejects_invalid_mode(self):
        cfg = default_config()
        cfg.runtime_mode = "testing"
        with pytest.raises(RuntimeModeError) as exc:
            validate_config(cfg)
        assert "shadow" in str(exc.value)
        assert "live" in str(exc.value)
        assert exc.value.error_code == "INVALID_RUNTIME_MODE"

    def test_rejects_empty_mode(self):
        cfg = default_config()
        cfg.runtime_mode = ""
        with pytest.raises(RuntimeModeError):
            validate_config(cfg)

    def test_rejects_none_mode(self):
        cfg = default_config()
        cfg.runtime_mode = None  # type: ignore
        with pytest.raises(RuntimeModeError):
            validate_config(cfg)


# ── 6. validate_config rejects gamma != 1.0 ─────────────────────────────────

class TestValidateGamma:
    def test_gamma_1_0_passes(self):
        cfg = default_config()
        cfg.model.gamma = 1.0
        validate_config(cfg)

    def test_rejects_gamma_0_99(self):
        cfg = default_config()
        cfg.model.gamma = 0.99
        with pytest.raises(GammaError) as exc:
            validate_config(cfg)
        assert "1.0" in str(exc.value)
        assert exc.value.error_code == "UNSUPPORTED_GAMMA"

    def test_rejects_gamma_1_1(self):
        cfg = default_config()
        cfg.model.gamma = 1.1
        with pytest.raises(GammaError):
            validate_config(cfg)

    def test_rejects_gamma_0(self):
        cfg = default_config()
        cfg.model.gamma = 0.0
        with pytest.raises(GammaError):
            validate_config(cfg)


# ── 7, 8. validate_config rejects sigma2 <= 0 ───────────────────────────────

class TestValidateSigma2:
    def test_sigma2_zero_rejected(self):
        cfg = default_config()
        cfg.model.reward_config.sigma2 = 0
        with pytest.raises(Sigma2Error) as exc:
            validate_config(cfg)
        assert "0" in str(exc.value)
        assert exc.value.error_code == "ZERO_OR_NEGATIVE_SIGMA2"

    def test_sigma2_negative_rejected(self):
        cfg = default_config()
        cfg.model.reward_config.sigma2 = -0.01
        with pytest.raises(Sigma2Error):
            validate_config(cfg)

    def test_sigma2_very_small_positive_passes(self):
        cfg = default_config()
        cfg.model.reward_config.sigma2 = 1e-10
        validate_config(cfg)  # passes the >0 check


# ── 9. validate_config rejects k not in {2, 3, 4} ──────────────────────────

class TestValidateK:
    def test_k_2_passes(self):
        cfg = default_config()
        cfg.model.k = 2
        validate_config(cfg)

    def test_k_3_passes(self):
        cfg = default_config()
        cfg.model.k = 3
        validate_config(cfg)

    def test_k_4_passes(self):
        cfg = default_config()
        cfg.model.k = 4
        validate_config(cfg)

    def test_rejects_k_5(self):
        cfg = default_config()
        cfg.model.k = 5
        with pytest.raises(KError) as exc:
            validate_config(cfg)
        assert "2, 3, 4" in str(exc.value) or "[2, 3, 4]" in str(exc.value)
        assert exc.value.error_code == "INVALID_K"

    def test_rejects_k_1(self):
        cfg = default_config()
        cfg.model.k = 1
        with pytest.raises(KError):
            validate_config(cfg)

    def test_rejects_k_0(self):
        cfg = default_config()
        cfg.model.k = 0
        with pytest.raises(KError):
            validate_config(cfg)


# ── 10, 11. validate_config rejects negative weights/costs ──────────────────

class TestValidateWeights:
    def test_w_meeting_negative_rejected(self):
        cfg = default_config()
        cfg.model.reward_config.w_meeting = -1.0
        with pytest.raises(WeightError) as exc:
            validate_config(cfg)
        assert "w_meeting" in str(exc.value)
        assert exc.value.error_code == "NEGATIVE_WEIGHT_OR_COST"

    def test_w_answered_negative_rejected(self):
        cfg = default_config()
        cfg.model.reward_config.w_answered = -0.1
        with pytest.raises(WeightError):
            validate_config(cfg)

    def test_c_dial_negative_rejected(self):
        cfg = default_config()
        cfg.model.reward_config.c_dial = -0.01
        with pytest.raises(WeightError) as exc:
            validate_config(cfg)
        assert "c_dial" in str(exc.value)

    def test_w_not_interested_negative_rejected(self):
        cfg = default_config()
        cfg.model.reward_config.w_not_interested = -0.5
        with pytest.raises(WeightError):
            validate_config(cfg)

    def test_all_non_negative_passes(self):
        cfg = default_config()
        cfg.model.reward_config.w_meeting = 0.0
        cfg.model.reward_config.w_answered = 0.0
        cfg.model.reward_config.c_dial = 0.0
        cfg.model.reward_config.w_not_interested = 0.0
        validate_config(cfg)


# ── 12. validate_config rejects inverted calendar window ────────────────────

class TestValidateCalendar:
    def test_valid_calendar_passes(self):
        cfg = default_config()
        validate_config(cfg)

    def test_rejects_start_gte_end(self):
        cfg = default_config()
        cfg.calendar.start_hour = 20
        cfg.calendar.end_hour = 10
        with pytest.raises(Exception) as exc:
            validate_config(cfg)
        assert "start_hour" in str(exc.value).lower() or "must be" in str(exc.value).lower()

    def test_rejects_equal_hours(self):
        cfg = default_config()
        cfg.calendar.start_hour = 12
        cfg.calendar.end_hour = 12
        with pytest.raises(Exception):
            validate_config(cfg)

    def test_rejects_zero_end(self):
        cfg = default_config()
        cfg.calendar.end_hour = 0
        with pytest.raises(Exception):
            validate_config(cfg)

    def test_duplicate_holidays_rejected(self):
        cfg = default_config()
        cfg.calendar.holidays = ["2026-08-15", "2026-08-15"]
        with pytest.raises(DuplicateHolidayError) as exc:
            validate_config(cfg)
        assert "Duplicate holiday" in str(exc.value)
        assert exc.value.error_code == "DUPLICATE_HOLIDAY"

    def test_no_duplicate_holidays_passes(self):
        cfg = default_config()
        cfg.calendar.holidays = ["2026-08-15", "2026-10-02", "2026-01-26"]
        validate_config(cfg)


# ── 14. config_hash returns consistent hash ─────────────────────────────────

class TestConfigHash:
    def test_same_config_same_hash(self):
        cfg1 = default_config()
        cfg2 = default_config()
        assert config_hash(cfg1) == config_hash(cfg2)

    def test_hash_is_sha256_hex(self):
        cfg = default_config()
        h = config_hash(cfg)
        assert len(h) == 64  # SHA-256 hex digest
        int(h, 16)  # should not raise — valid hex

    def test_different_config_different_hash(self):
        cfg1 = default_config()
        cfg2 = Config(runtime_mode="live")
        assert config_hash(cfg1) != config_hash(cfg2)

    def test_hash_is_deterministic(self):
        cfg = default_config()
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        h3 = config_hash(cfg)
        assert h1 == h2 == h3

    def test_hash_canonical_json(self):
        cfg = default_config()
        h = config_hash(cfg)
        expected = hashlib.sha256(
            json.dumps(
                {
                    "runtime_mode": "shadow",
                    "timezone": "Asia/Kolkata",
                    "calendar": {
                        "days_of_week": [0, 1, 2, 3, 4, 5],
                        "start_hour": 8,
                        "end_hour": 18,
                        "holidays": [],
                    },
                    "initial_delay_maximum_minutes": 15,
                    "expiry_when_upstream_absent": None,
                    "max_calls_per_seller_per_day": 3,
                    "max_attempts_per_lead": 5,
                    "minimum_inter_call_gap_minutes": 15,
                    "dispatch_lead_time_seconds": 5,
                    "decision_execution_tolerance_minutes": 5,
                    "cache_staleness_seconds": 5,
                    "experiment_enabled": False,
                    "double_call_enabled": False,
                    "terminal_retry_dispositions": [
                        "Meeting Fixed", "Not Interested", "General"
                    ],
                    "retention_horizon_days": 90,
                    "model": {
                        "k": 4,
                        "reward_config": {
                            "w_meeting": 1.0,
                            "w_answered": 0.1,
                            "c_dial": 0.02,
                            "w_not_interested": 0.0,
                            "sigma2": 0.06,
                        },
                        "gamma": 1.0,
                        "lambda_smooth": 1.0,
                        "lambda_parent": 10.0,
                        "alpha": 0.1,
                        "segment_min_attempts": 2000,
                        "segment_min_sellers": 200,
                        "support_bin_min_attempts": 50,
                        "support_bin_min_sellers": 30,
                        "state_history_start": "2026-04-01",
                        "state_history_end": "2026-10-01",
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        assert h == expected


# ── 15. YAML load with unknown keys raises error ────────────────────────────

class TestYAMLUnknownKeys:
    def test_top_level_unknown_key(self):
        yaml_content = "runtime_mode: shadow\nunknown_key: value\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            with pytest.raises(UnknownKeyError) as exc:
                load_config(tmp)
            assert "unknown_key" in str(exc.value)
        finally:
            os.unlink(tmp)

    def test_calendar_unknown_key(self):
        yaml_content = (
            "calendar:\n"
            "  start_hour: 8\n"
            "  end_hour: 18\n"
            "  unknown_cal_key: true\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            with pytest.raises(UnknownKeyError) as exc:
                load_config(tmp)
            assert "unknown_cal_key" in str(exc.value)
        finally:
            os.unlink(tmp)

    def test_model_unknown_key(self):
        yaml_content = (
            "model:\n"
            "  k: 4\n"
            "  gamma: 1.0\n"
            "  unknown_model_key: 42\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            with pytest.raises(UnknownKeyError) as exc:
                load_config(tmp)
            assert "unknown_model_key" in str(exc.value)
        finally:
            os.unlink(tmp)

    def test_reward_config_unknown_key(self):
        yaml_content = (
            "model:\n"
            "  reward_config:\n"
            "    w_meeting: 1.0\n"
            "    unknown_reward_key: 99\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            with pytest.raises(UnknownKeyError) as exc:
                load_config(tmp)
            assert "unknown_reward_key" in str(exc.value)
        finally:
            os.unlink(tmp)


# ── 16. Create a valid YAML config file and load it ─────────────────────────

class TestYAMLLoad:
    def test_load_valid_yaml(self):
        yaml_content = """\
runtime_mode: shadow
timezone: Asia/Kolkata
calendar:
  days_of_week: [0, 1, 2, 3, 4, 5]
  start_hour: 8
  end_hour: 18
  holidays: []
initial_delay_maximum_minutes: 15
max_calls_per_seller_per_day: 3
max_attempts_per_lead: 5
minimum_inter_call_gap_minutes: 15
dispatch_lead_time_seconds: 5
decision_execution_tolerance_minutes: 5
cache_staleness_seconds: 5
experiment_enabled: false
double_call_enabled: false
terminal_retry_dispositions:
  - Meeting Fixed
  - Not Interested
  - General
retention_horizon_days: 90
model:
  k: 4
  gamma: 1.0
  lambda_smooth: 1.0
  lambda_parent: 10.0
  alpha: 0.1
  segment_min_attempts: 2000
  segment_min_sellers: 200
  support_bin_min_attempts: 50
  support_bin_min_sellers: 30
  state_history_start: "2026-04-01"
  state_history_end: "2026-10-01"
  reward_config:
    w_meeting: 1.0
    w_answered: 0.1
    c_dial: 0.02
    w_not_interested: 0.0
    sigma2: 0.06
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = load_config(tmp)
            assert cfg.runtime_mode == "shadow"
            assert cfg.timezone == "Asia/Kolkata"
            assert cfg.model.k == 4
            assert cfg.model.gamma == 1.0
            assert cfg.model.reward_config.w_meeting == 1.0
        finally:
            os.unlink(tmp)

    def test_load_minimal_yaml(self):
        yaml_content = "{}"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = load_config(tmp)
            assert cfg.runtime_mode == "shadow"
        finally:
            os.unlink(tmp)

    def test_load_yaml_with_custom_values(self):
        yaml_content = """\
runtime_mode: live
initial_delay_maximum_minutes: 30
max_calls_per_seller_per_day: 5
model:
  k: 3
  gamma: 1.0
  lambda_smooth: 0.1
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            cfg = load_config(tmp)
            assert cfg.runtime_mode == "live"
            assert cfg.initial_delay_maximum_minutes == 30
            assert cfg.max_calls_per_seller_per_day == 5
            assert cfg.model.k == 3
            assert cfg.model.lambda_smooth == 0.1
        finally:
            os.unlink(tmp)


# ── 17. ModelConfig with k=2, lambda_smooth=0.1, gamma=1.0 → valid ─────────

class TestModelConfigValid:
    def test_k2_valid(self):
        mc = ModelConfig(k=2, gamma=1.0, lambda_smooth=0.1)
        assert mc.k == 2
        assert mc.gamma == 1.0
        assert mc.lambda_smooth == 0.1

    def test_k3_valid(self):
        mc = ModelConfig(k=3, gamma=1.0)
        assert mc.k == 3

    def test_k4_valid(self):
        mc = ModelConfig(k=4, gamma=1.0)
        assert mc.k == 4


# ── 18. RewardConfig with sigma2=1e-4 → valid (floored at 1e-4) ────────────

class TestRewardConfigSigma2Floor:
    def test_sigma2_1e4_passes(self):
        rc = RewardConfig(sigma2=1e-4)
        assert rc.sigma2 >= 1e-4

    def test_sigma2_very_small_is_floored(self):
        rc = RewardConfig(sigma2=1e-10)
        assert rc.sigma2 == 1e-4

    def test_sigma2_below_floor_is_floored(self):
        rc = RewardConfig(sigma2=5e-5)
        assert rc.sigma2 == 1e-4

    def test_sigma2_at_floor_unchanged(self):
        rc = RewardConfig(sigma2=0.0001)
        assert rc.sigma2 == 0.0001


# ── Bonus: Config construction via __post_init__ ────────────────────────────

class TestConfigPostInit:
    def test_config_construction_validates(self):
        with pytest.raises(GammaError):
            Config(model=ModelConfig(gamma=0.99))

    def test_config_construction_invalid_mode(self):
        with pytest.raises(RuntimeModeError):
            Config(runtime_mode="bad_mode")

    def test_config_construction_invalid_calendar(self):
        with pytest.raises(Exception):
            Config(calendar=Calendar(start_hour=20, end_hour=10))

    def test_config_construction_duplicate_holidays(self):
        with pytest.raises(DuplicateHolidayError):
            Config(calendar=Calendar(holidays=["2026-01-01", "2026-01-01"]))

    def test_config_construction_invalid_k(self):
        with pytest.raises(KError):
            Config(model=ModelConfig(k=5))

    def test_config_construction_negative_weight(self):
        with pytest.raises(WeightError):
            Config(model=ModelConfig(reward_config=RewardConfig(w_meeting=-1.0)))

    def test_config_construction_invalid_sigma2(self):
        with pytest.raises(Sigma2Error):
            Config(model=ModelConfig(reward_config=RewardConfig(sigma2=0)))


# ── Bonus: RewardConfig __post_init__ validation ────────────────────────────

class TestRewardConfigPostInit:
    def test_valid_reward_config(self):
        rc = RewardConfig()
        assert rc.w_meeting == 1.0

    def test_zero_weights_allowed(self):
        rc = RewardConfig(w_meeting=0.0, w_answered=0.0, c_dial=0.0, w_not_interested=0.0)
        assert rc.w_meeting == 0.0

    def test_positive_sigma2_allowed(self):
        rc = RewardConfig(sigma2=0.5)
        assert rc.sigma2 == 0.5


# ── Bonus: ModelConfig __post_init__ validation ─────────────────────────────

class TestModelConfigPostInit:
    def test_valid_model_config(self):
        mc = ModelConfig()
        assert mc.k == 4

    def test_nested_reward_validation(self):
        with pytest.raises(Sigma2Error):
            ModelConfig(reward_config=RewardConfig(sigma2=-1.0))

    def test_nested_reward_validation_via_model(self):
        with pytest.raises(WeightError):
            ModelConfig(reward_config=RewardConfig(w_meeting=-5.0))


# ── Bonus: Error code coverage ──────────────────────────────────────────────

class TestErrorCodeCoverage:
    def test_runtime_mode_error_code(self):
        with pytest.raises(RuntimeModeError) as exc:
            validate_config(Config(runtime_mode="invalid"))
        assert exc.value.error_code == "INVALID_RUNTIME_MODE"

    def test_gamma_error_code(self):
        with pytest.raises(GammaError) as exc:
            validate_config(Config(model=ModelConfig(gamma=0.9)))
        assert exc.value.error_code == "UNSUPPORTED_GAMMA"

    def test_sigma2_error_code(self):
        with pytest.raises(Sigma2Error) as exc:
            validate_config(Config(model=ModelConfig(reward_config=RewardConfig(sigma2=0))))
        assert exc.value.error_code == "ZERO_OR_NEGATIVE_SIGMA2"

    def test_k_error_code(self):
        with pytest.raises(KError) as exc:
            validate_config(Config(model=ModelConfig(k=5)))
        assert exc.value.error_code == "INVALID_K"

    def test_weight_error_code(self):
        with pytest.raises(WeightError) as exc:
            validate_config(Config(model=ModelConfig(reward_config=RewardConfig(w_meeting=-1.0))))
        assert exc.value.error_code == "NEGATIVE_WEIGHT_OR_COST"

    def test_duplicate_holiday_error_code(self):
        with pytest.raises(DuplicateHolidayError) as exc:
            validate_config(Config(calendar=Calendar(holidays=["2026-01-01", "2026-01-01"])))
        assert exc.value.error_code == "DUPLICATE_HOLIDAY"

    def test_unknown_key_error_code(self):
        yaml_content = "bad_key: 1\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp = f.name
        try:
            with pytest.raises(UnknownKeyError) as exc:
                load_config(tmp)
            assert exc.value.error_code == "UNKNOWN_CONFIG_KEY"
        finally:
            os.unlink(tmp)


# ── Bonus: Calendar validation ──────────────────────────────────────────────

class TestCalendarValidation:
    def test_calendar_equal_hours_fails(self):
        with pytest.raises(Exception):
            Calendar(start_hour=10, end_hour=10)

    def test_calendar_start_after_end_fails(self):
        with pytest.raises(Exception):
            Calendar(start_hour=20, end_hour=6)

    def test_calendar_valid_range(self):
        cal = Calendar(start_hour=6, end_hour=22)
        assert cal.start_hour == 6
        assert cal.end_hour == 22

    def test_calendar_default_days(self):
        cal = Calendar()
        assert cal.days_of_week == [0, 1, 2, 3, 4, 5]

    def test_calendar_custom_days(self):
        cal = Calendar(days_of_week=[0, 1, 2, 3, 4])
        assert cal.days_of_week == [0, 1, 2, 3, 4]
