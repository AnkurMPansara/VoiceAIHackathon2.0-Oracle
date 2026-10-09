"""Tests for src/btc/model/reward.py — reward computation and validation.

Covers SRS MOD-01 (reward formula), DATA-04 (cross-field constraints),
and SRS T02 (expected reward values).
"""

import pytest

from btc.config import RewardConfig
from btc.model.reward import (
    batch_compute_rewards,
    compute_reward,
    validate_outcome_consistency,
)

# ── Default and custom configs ────────────────────────────────────────────────

DEFAULT_CONFIG = RewardConfig()  # (1.0, 0.1, 0.02, 0.0)

CUSTOM_CONFIG = RewardConfig(
    w_meeting=2.0,
    w_answered=0.5,
    c_dial=0.1,
    w_not_interested=0.3,
)

# ── T02: Reward computation with defaults ─────────────────────────────────────


class TestComputeRewardDefaults:
    """T02: Verify expected reward values with default weights (1.0, 0.1, 0.02, 0.0)."""

    def test_unanswered(self):
        """answered=False, meeting_fixed=False → reward = -0.02"""
        reward = compute_reward(
            answered=False,
            meeting_fixed=False,
            disposition="NOT_ANSWERED",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(-0.02)
        assert isinstance(reward, float)

    def test_answered_no_meeting(self):
        """answered=True, meeting_fixed=False → reward = 0.1 - 0.02 = 0.08"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="GENERAL",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(0.08)
        assert isinstance(reward, float)

    def test_meeting_fixed(self):
        """answered=True, meeting_fixed=True → reward = 1.0 + 0.1 - 0.02 = 1.08"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=True,
            disposition="MEETING_FIXED",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(1.08)
        assert isinstance(reward, float)

    def test_not_interested_default_penalty_zero(self):
        """answered=True, meeting_fixed=False, disposition=NOT_INTERESTED,
        w_not_interested=0 → reward = 0.1 - 0.02 = 0.08"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="NOT_INTERESTED",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(0.08)

    def test_not_interested_with_penalty(self):
        """answered=True, meeting_fixed=False, disposition=NOT_INTERESTED,
        w_not_interested=0.5 → reward = 0.1 - 0.02 - 0.5 = -0.42"""
        config = RewardConfig(w_not_interested=0.5)
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="NOT_INTERESTED",
            reward_config=config,
        )
        assert reward == pytest.approx(-0.42)

    def test_general_disposition(self):
        """answered=True, meeting_fixed=False, disposition=GENERAL → 0.08"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="GENERAL",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(0.08)

    def test_call_later_busy(self):
        """answered=True, meeting_fixed=False, disposition=CALL_LATER_BUSY → 0.08"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="CALL_LATER_BUSY",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(0.08)

    def test_unknown_disposition(self):
        """answered=True, meeting_fixed=False, disposition=UNKNOWN → 0.08"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="UNKNOWN",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(0.08)

    def test_unanswered_not_interested_disposition(self):
        """answered=False, disposition=NOT_INTERESTED → -0.02 (no answered bonus, no meeting)."""
        reward = compute_reward(
            answered=False,
            meeting_fixed=False,
            disposition="NOT_INTERESTED",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(-0.02)

    def test_unanswered_call_later_busy_disposition(self):
        """answered=False, disposition=CALL_LATER_BUSY → -0.02."""
        reward = compute_reward(
            answered=False,
            meeting_fixed=False,
            disposition="CALL_LATER_BUSY",
            reward_config=DEFAULT_CONFIG,
        )
        assert reward == pytest.approx(-0.02)


# ── Custom reward config ─────────────────────────────────────────────────────


class TestComputeRewardCustomConfig:
    """Test reward computation with non-default weights."""

    def test_meeting_custom(self):
        """w_meeting=2.0, w_answered=0.5, c_dial=0.1 → 2.0 + 0.5 - 0.1 = 2.4"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=True,
            disposition="MEETING_FIXED",
            reward_config=CUSTOM_CONFIG,
        )
        assert reward == pytest.approx(2.4)

    def test_not_interested_custom(self):
        """w_meeting=2.0, w_answered=0.5, c_dial=0.1, w_not_interested=0.3
        → 0 + 0.5 - 0.1 - 0.3 = 0.1"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="NOT_INTERESTED",
            reward_config=CUSTOM_CONFIG,
        )
        assert reward == pytest.approx(0.1)

    def test_unanswered_custom(self):
        """w_meeting=2.0, w_answered=0.5, c_dial=0.1 → 0 + 0 - 0.1 = -0.1"""
        reward = compute_reward(
            answered=False,
            meeting_fixed=False,
            disposition="NOT_ANSWERED",
            reward_config=CUSTOM_CONFIG,
        )
        assert reward == pytest.approx(-0.1)

    def test_general_custom(self):
        """w_meeting=2.0, w_answered=0.5, c_dial=0.1 → 0 + 0.5 - 0.1 = 0.4"""
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="GENERAL",
            reward_config=CUSTOM_CONFIG,
        )
        assert reward == pytest.approx(0.4)


# ── DATA-04: Validation ─────────────────────────────────────────────────────


class TestValidateOutcomeConsistency:
    """DATA-04: Cross-field constraint validation."""

    def test_valid_meeting_fixed(self):
        """True, True, MEETING_FIXED → (True, '')"""
        valid, msg = validate_outcome_consistency(True, True, "MEETING_FIXED")
        assert valid is True
        assert msg == ""

    def test_meeting_fixed_requires_answered(self):
        """False, True, MEETING_FIXED → (False, error)"""
        valid, msg = validate_outcome_consistency(False, True, "MEETING_FIXED")
        assert valid is False
        assert "meeting_fixed=True requires answered=True" in msg

    def test_meeting_fixed_requires_disposition(self):
        """True, True, GENERAL → (False, error)"""
        valid, msg = validate_outcome_consistency(True, True, "GENERAL")
        assert valid is False
        assert "meeting_fixed=True requires disposition='MEETING_FIXED'" in msg

    def test_not_answered_requires_not_answered(self):
        """True, False, NOT_ANSWERED → (False, error)"""
        valid, msg = validate_outcome_consistency(True, False, "NOT_ANSWERED")
        assert valid is False
        assert "NOT_ANSWERED disposition requires answered=False" in msg

    def test_meeting_fixed_disposition_requires_answered(self):
        """False, False, MEETING_FIXED → (False, error)"""
        valid, msg = validate_outcome_consistency(False, False, "MEETING_FIXED")
        assert valid is False
        assert "disposition='MEETING_FIXED' requires answered=True" in msg

    def test_unanswered_valid(self):
        """False, False, NOT_ANSWERED → (True, '')"""
        valid, msg = validate_outcome_consistency(False, False, "NOT_ANSWERED")
        assert valid is True
        assert msg == ""

    def test_answered_no_meeting_valid(self):
        """True, False, GENERAL → (True, '')"""
        valid, msg = validate_outcome_consistency(True, False, "GENERAL")
        assert valid is True
        assert msg == ""

    def test_call_later_busy_valid(self):
        """True, False, CALL_LATER_BUSY → (True, '')"""
        valid, msg = validate_outcome_consistency(True, False, "CALL_LATER_BUSY")
        assert valid is True
        assert msg == ""

    def test_unanswered_call_later_busy_valid(self):
        """False, False, CALL_LATER_BUSY → (True, '') — CALL_LATER_BUSY
        SHALL NOT itself determine whether the call was answered."""
        valid, msg = validate_outcome_consistency(False, False, "CALL_LATER_BUSY")
        assert valid is True
        assert msg == ""

    def test_unknown_disposition_valid(self):
        """True, False, UNKNOWN → (True, '')"""
        valid, msg = validate_outcome_consistency(True, False, "UNKNOWN")
        assert valid is True
        assert msg == ""

    def test_not_interested_valid(self):
        """True, False, NOT_INTERESTED → (True, '')"""
        valid, msg = validate_outcome_consistency(True, False, "NOT_INTERESTED")
        assert valid is True
        assert msg == ""


class TestComputeRewardValidationErrors:
    """compute_reward should raise ValueError on invalid inputs."""

    def test_meeting_fixed_without_answered_raises(self):
        """meeting_fixed=True, answered=False → ValueError"""
        with pytest.raises(ValueError, match="meeting_fixed=True requires answered=True"):
            compute_reward(
                answered=False,
                meeting_fixed=True,
                disposition="MEETING_FIXED",
                reward_config=DEFAULT_CONFIG,
            )

    def test_meeting_fixed_wrong_disposition_raises(self):
        """meeting_fixed=True, disposition='GENERAL' → ValueError"""
        with pytest.raises(ValueError, match="meeting_fixed=True requires disposition"):
            compute_reward(
                answered=True,
                meeting_fixed=True,
                disposition="GENERAL",
                reward_config=DEFAULT_CONFIG,
            )

    def test_answered_with_not_answered_disposition_raises(self):
        """answered=True, disposition='NOT_ANSWERED' → ValueError"""
        with pytest.raises(ValueError, match="NOT_ANSWERED disposition requires answered=False"):
            compute_reward(
                answered=True,
                meeting_fixed=False,
                disposition="NOT_ANSWERED",
                reward_config=DEFAULT_CONFIG,
            )

    def test_meeting_fixed_disposition_without_answered_raises(self):
        """answered=False, disposition='MEETING_FIXED' → ValueError"""
        with pytest.raises(ValueError, match="disposition='MEETING_FIXED' requires answered=True"):
            compute_reward(
                answered=False,
                meeting_fixed=False,
                disposition="MEETING_FIXED",
                reward_config=DEFAULT_CONFIG,
            )

    def test_invalid_disposition_raises(self):
        """Unknown disposition string → ValueError"""
        with pytest.raises(ValueError, match="Invalid disposition"):
            compute_reward(
                answered=True,
                meeting_fixed=False,
                disposition="SOME_UNKNOWN_VALUE",
                reward_config=DEFAULT_CONFIG,
            )


# ── Batch computation ────────────────────────────────────────────────────────


class TestBatchComputeRewards:
    """batch_compute_rewards functionality."""

    def test_batch_valid_outcomes(self):
        """Valid list → returns list of rewards."""
        outcomes = [
            {"answered": False, "meeting_fixed": False, "disposition": "NOT_ANSWERED"},
            {"answered": True, "meeting_fixed": False, "disposition": "GENERAL"},
            {"answered": True, "meeting_fixed": True, "disposition": "MEETING_FIXED"},
        ]
        rewards = batch_compute_rewards(outcomes, DEFAULT_CONFIG)
        assert len(rewards) == 3
        assert rewards[0] == pytest.approx(-0.02)
        assert rewards[1] == pytest.approx(0.08)
        assert rewards[2] == pytest.approx(1.08)
        assert all(isinstance(r, float) for r in rewards)

    def test_batch_single_outcome(self):
        """Single outcome → list with one reward."""
        outcomes = [
            {"answered": True, "meeting_fixed": False, "disposition": "GENERAL"},
        ]
        rewards = batch_compute_rewards(outcomes, DEFAULT_CONFIG)
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(0.08)

    def test_batch_empty(self):
        """Empty list → empty list."""
        rewards = batch_compute_rewards([], DEFAULT_CONFIG)
        assert rewards == []

    def test_batch_invalid_raises_on_first(self):
        """Invalid entry → raises ValueError on first invalid outcome."""
        outcomes = [
            {"answered": True, "meeting_fixed": False, "disposition": "GENERAL"},
            {"answered": False, "meeting_fixed": True, "disposition": "MEETING_FIXED"},
        ]
        with pytest.raises(ValueError):
            batch_compute_rewards(outcomes, DEFAULT_CONFIG)

    def test_batch_missing_field_raises(self):
        """Outcome missing required field → ValueError with index info."""
        outcomes = [
            {"answered": True, "meeting_fixed": False},  # missing 'disposition'
        ]
        with pytest.raises(ValueError, match="missing required field"):
            batch_compute_rewards(outcomes, DEFAULT_CONFIG)

    def test_batch_all_dispositions(self):
        """All valid dispositions produce rewards."""
        dispositions = [
            "MEETING_FIXED",
            "NOT_INTERESTED",
            "GENERAL",
            "CALL_LATER_BUSY",
            "NOT_ANSWERED",
            "UNKNOWN",
        ]
        outcomes = [
            {
                "answered": d in ("MEETING_FIXED", "NOT_INTERESTED", "GENERAL", "CALL_LATER_BUSY", "UNKNOWN"),
                "meeting_fixed": d == "MEETING_FIXED",
                "disposition": d,
            }
            for d in dispositions
        ]
        rewards = batch_compute_rewards(outcomes, DEFAULT_CONFIG)
        assert len(rewards) == len(dispositions)
        assert all(isinstance(r, float) for r in rewards)


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_all_dispositions_return_float(self):
        """Every disposition returns a float type."""
        for disposition in ["MEETING_FIXED", "NOT_INTERESTED", "GENERAL", "CALL_LATER_BUSY", "NOT_ANSWERED", "UNKNOWN"]:
            answered = disposition != "NOT_ANSWERED"
            meeting_fixed = disposition == "MEETING_FIXED"
            reward = compute_reward(
                answered=answered,
                meeting_fixed=meeting_fixed,
                disposition=disposition,
                reward_config=DEFAULT_CONFIG,
            )
            assert isinstance(reward, float), f"{disposition} did not return float"

    def test_constant_cost_does_not_alter_ranking(self):
        """Different c_dial values preserve the same ordering of rewards."""
        outcomes = [
            (False, False, "NOT_ANSWERED"),
            (True, False, "GENERAL"),
            (True, True, "MEETING_FIXED"),
        ]
        rewards_by_c_dial = {}
        for c_dial in [0.0, 0.01, 0.02, 0.05, 0.1, 1.0]:
            config = RewardConfig(c_dial=c_dial)
            rewards = [
                compute_reward(answered, meeting_fixed, disposition, config)
                for answered, meeting_fixed, disposition in outcomes
            ]
            rewards_by_c_dial[c_dial] = rewards

        # Check that ranking is the same for all c_dial values
        for c_dial_a, c_dial_b in [(0.0, 0.02), (0.02, 0.1), (0.0, 1.0), (0.05, 0.1)]:
            ra = rewards_by_c_dial[c_dial_a]
            rb = rewards_by_c_dial[c_dial_b]
            # answered_no_meeting > unanswered, meeting > answered_no_meeting
            assert ra[2] > ra[1] > ra[0], f"Ranking broken for c_dial={c_dial_a}"
            assert rb[2] > rb[1] > rb[0], f"Ranking broken for c_dial={c_dial_b}"
            # The ordering should be identical
            assert (ra[0] < ra[1] < ra[2]) == (rb[0] < rb[1] < rb[2])

    def test_zero_weights(self):
        """All weights zero → only -c_dial remains."""
        config = RewardConfig(w_meeting=0.0, w_answered=0.0, c_dial=0.02, w_not_interested=0.0)
        reward = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="GENERAL",
            reward_config=config,
        )
        assert reward == pytest.approx(-0.02)

    def test_high_weights(self):
        """Large weights → large rewards."""
        config = RewardConfig(w_meeting=100.0, w_answered=50.0, c_dial=0.01, w_not_interested=10.0)
        reward = compute_reward(
            answered=True,
            meeting_fixed=True,
            disposition="MEETING_FIXED",
            reward_config=config,
        )
        assert reward == pytest.approx(149.99)

    def test_negative_not_interested_penalty(self):
        """w_not_interested=0 means no penalty, not_interested == answered_no_meeting."""
        reward_meeting = compute_reward(
            answered=True,
            meeting_fixed=False,
            disposition="MEETING_FIXED",  # This will fail validation
            reward_config=DEFAULT_CONFIG,
        )

    def test_return_type_is_always_float(self):
        """compute_reward always returns float, not bool or int."""
        reward = compute_reward(
            answered=False,
            meeting_fixed=False,
            disposition="NOT_ANSWERED",
            reward_config=DEFAULT_CONFIG,
        )
        assert type(reward) is float  # not just isinstance


class TestSRSFormulaCompliance:
    """Verify the implementation matches the MOD-01 formula exactly."""

    def test_formula_structure_unanswered(self):
        """y = 0*w_meeting + 0*w_answered - c_dial - 0*w_not_interested = -c_dial"""
        config = RewardConfig(w_meeting=1.0, w_answered=0.1, c_dial=0.02, w_not_interested=0.0)
        reward = compute_reward(False, False, "NOT_ANSWERED", config)
        expected = -0.02
        assert reward == pytest.approx(expected)

    def test_formula_structure_answered_no_meeting(self):
        """y = 0*w_meeting + 1*w_answered - c_dial - 0*w_not_interested = w_answered - c_dial"""
        config = RewardConfig(w_meeting=1.0, w_answered=0.1, c_dial=0.02, w_not_interested=0.0)
        reward = compute_reward(True, False, "GENERAL", config)
        expected = 0.1 - 0.02
        assert reward == pytest.approx(expected)

    def test_formula_structure_meeting(self):
        """y = 1*w_meeting + 1*w_answered - c_dial - 0*w_not_interested = w_meeting + w_answered - c_dial"""
        config = RewardConfig(w_meeting=1.0, w_answered=0.1, c_dial=0.02, w_not_interested=0.0)
        reward = compute_reward(True, True, "MEETING_FIXED", config)
        expected = 1.0 + 0.1 - 0.02
        assert reward == pytest.approx(expected)

    def test_formula_structure_not_interested(self):
        """y = 0*w_meeting + 1*w_answered - c_dial - 1*w_not_interested = w_answered - c_dial - w_not_interested"""
        config = RewardConfig(w_meeting=1.0, w_answered=0.1, c_dial=0.02, w_not_interested=0.5)
        reward = compute_reward(True, False, "NOT_INTERESTED", config)
        expected = 0.1 - 0.02 - 0.5
        assert reward == pytest.approx(expected)
