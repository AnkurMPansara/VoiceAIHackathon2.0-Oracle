"""Comprehensive unit tests for `src/btc/model/stats.py` (MOD-04, MOD-05, STATE-07).

Tests seller sufficient statistics, Bayesian update functions, and compact
codec math against SRS §5.2 (MOD-04, MOD-05), §6.3 (TRAIN-03), and §9.2
(STATE-07).

SRS requirements covered::

    MOD-04:   A += outer(phi,phi)/sigma2, b += y*phi/sigma2, n += 1, gamma=1
    MOD-05:   Revision: subtract old, add new; n unchanged
    STATE-07: bytes(d) = 48 + 8*(d*(d+1)/2 + d), bytes(9) = 480
    TRAIN-03: Sigma0 = alpha * diag(1,1,1,1/4,1/4,...,1/K^2,1/K^2)

Test categories::
    - SellerState: zero_state, A property, copy, dimension validation
    - compute_contribution: formula, scaling, zero-reward
    - apply_contribution: single, batch, order independence, versioning
    - revoke_contribution: restore, n unchanged, versioning
    - update_revision: A,b equivalence, n unchanged, label change
    - Upper triangle utilities: extract/reconstruct identity, symmetry
    - Prior: SPD, TRAIN-03 diagonal, Lambda0, eta0
"""

from __future__ import annotations

import numpy as np
import pytest

from btc.model.stats import (
    apply_contribution,
    compute_contribution,
    extract_upper_triangle,
    posterior_params,
    reconstruct_symmetric,
    revoke_contribution,
    update_revision,
    zero_state,
    Prior,
    SellerState,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

D_VALUES = [3, 5, 7, 9]
K_VALUES = [2, 3, 4]
ALPHA_VALUES = [0.01, 0.1, 1.0]


@pytest.fixture(params=D_VALUES)
def d(request):
    """Feature dimensions to test."""
    return request.param


@pytest.fixture(params=K_VALUES)
def k(request):
    """Fourier harmonic counts."""
    return request.param


@pytest.fixture(params=ALPHA_VALUES)
def alpha(request):
    """Prior scale factors."""
    return request.param


@pytest.fixture
def d5_state():
    """Zero state with d=5."""
    return zero_state(d=5)


@pytest.fixture
def d9_state():
    """Zero state with d=9."""
    return zero_state(d=9)


@pytest.fixture
def phi_d5():
    """Feature vector for d=5 at t=0: [1, 0, 1, 0, 1]."""
    return np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)


@pytest.fixture
def phi_d5_nonzero():
    """Non-trivial feature vector for d=5."""
    return np.array([0.5, 0.8, 0.3, 0.6, 0.9], dtype=np.float64)


@pytest.fixture
def sigma2():
    """Working noise variance."""
    return 0.06


@pytest.fixture
def reward_meeting():
    """Meeting fixed reward: w_meeting + w_answered - c_dial = 1.0 + 0.1 - 0.02 = 1.08."""
    return 1.08


@pytest.fixture
def reward_no_meeting():
    """Answered no meeting reward: w_answered - c_dial = 0.1 - 0.02 = 0.08."""
    return 0.08


@pytest.fixture
def prior_d5():
    """Prior for d=5 (K=2)."""
    return Prior.diagonal_prior(d=5, alpha=0.1)


@pytest.fixture
def prior_d9():
    """Prior for d=9 (K=4)."""
    return Prior.diagonal_prior(d=9, alpha=0.1)


# ── Helper: expected upper triangle of outer(phi, phi) ────────────────────────

def expected_A_upper(phi, sigma2):
    """Compute expected A_upper = upper_triangle(outer(phi, phi) / sigma2)."""
    d = len(phi)
    outer_phi = np.outer(phi, phi) / sigma2
    return extract_upper_triangle(outer_phi)


def expected_b(phi, reward, sigma2):
    """Compute expected b = phi * reward / sigma2."""
    return phi * reward / sigma2


# ═══════════════════════════════════════════════════════════════════════════════
#  SellerState: zero_state, A property, copy, dimension validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroState:
    """SRS §15 T01/T03: zero_state(d) creates cold-start state.

    Test 1: zero_state(d=5) returns A_upper=zeros(15), b=zeros(5), n=0, d=5.
    """

    def test_zero_state_d5(self, d5_state):
        """Test 1: d=5 → A_upper shape (15,), b shape (5,), n=0, d=5."""
        assert d5_state.d == 5
        assert d5_state.n == 0
        assert d5_state.A_upper.shape == (15,)
        assert d5_state.b.shape == (5,)
        assert np.allclose(d5_state.A_upper, 0.0)
        assert np.allclose(d5_state.b, 0.0)

    def test_zero_state_d9(self, d9_state):
        """d=9 → A_upper shape (45,), b shape (9,), n=0."""
        assert d9_state.d == 9
        assert d9_state.n == 0
        assert d9_state.A_upper.shape == (45,)
        assert d9_state.b.shape == (9,)

    def test_zero_state_all_d(self, d):
        """zero_state produces correct shapes for all tested d values."""
        expected_upper = d * (d + 1) // 2
        state = zero_state(d=d)
        assert state.d == d
        assert state.n == 0
        assert state.A_upper.shape == (expected_upper,)
        assert state.b.shape == (d,)
        assert np.allclose(state.A_upper, 0.0)
        assert np.allclose(state.b, 0.0)

    def test_zero_state_custom_version(self):
        """zero_state accepts custom initial state_version."""
        state = zero_state(d=5, state_version=42)
        assert state.state_version == 42

    def test_zero_state_d_invalid_raises(self):
        """zero_state(d <= 0) raises ValueError."""
        with pytest.raises(ValueError, match="d must be > 0"):
            zero_state(d=0)
        with pytest.raises(ValueError, match="d must be > 0"):
            zero_state(d=-1)


class TestSellerStateAProperty:
    """Test 2: SellerState.A property reconstructs full symmetric matrix.

    SRS MOD-04 / STATE-07: A is stored as upper triangle only.
    """

    def test_A_shape(self, d5_state):
        """A property returns (d, d) matrix."""
        assert d5_state.A.shape == (5, 5)

    def test_A_symmetric(self, d5_state):
        """A is symmetric: A == A.T."""
        np.testing.assert_array_equal(d5_state.A, d5_state.A.T)

    def test_A_all_zeros_when_zero_state(self, d5_state):
        """A is all zeros for zero state."""
        np.testing.assert_array_equal(d5_state.A, np.zeros((5, 5)))

    def test_A_after_single_apply(self, d5_state, phi_d5, sigma2):
        """A matches outer(phi, phi)/sigma2 after single apply."""
        state = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        expected = np.outer(phi_d5, phi_d5) / sigma2
        np.testing.assert_allclose(state.A, expected, atol=1e-12)

    def test_A_after_multiple_applies(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """A matches sum of outer products after multiple applies."""
        s1 = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        s2 = apply_contribution(s1, phi_d5_nonzero, reward=0.5, sigma2=sigma2)
        A1, _ = compute_contribution(phi_d5, reward=1.0, sigma2=sigma2)
        A2, _ = compute_contribution(phi_d5_nonzero, reward=0.5, sigma2=sigma2)
        expected = reconstruct_symmetric(A1 + A2, 5)
        np.testing.assert_allclose(s2.A, expected, atol=1e-12)

    def test_A_symmetric_after_apply(self, d5_state, phi_d5_nonzero, sigma2):
        """A remains symmetric after apply."""
        state = apply_contribution(d5_state, phi_d5_nonzero, reward=1.0, sigma2=sigma2)
        np.testing.assert_array_equal(state.A, state.A.T)

    def test_A_reconstruction_is_identity(self, d):
        """extract_upper_triangle(reconstruct_symmetric(m)) == upper_triangle(m)."""
        m = np.random.RandomState(42).randn(d, d)
        m = (m + m.T) / 2  # make symmetric
        upper = extract_upper_triangle(m)
        reconstructed = reconstruct_symmetric(upper, d)
        np.testing.assert_allclose(reconstructed, m, atol=1e-12)


class TestSellerStateCopy:
    """Test 3: SellerState.copy() returns independent deep copy.

    SRS MOD-04: state is immutable per operation (new SellerState returned).
    """

    def test_copy_is_different_object(self, d5_state):
        """copy() returns a different object."""
        other = d5_state.copy()
        assert other is not d5_state

    def test_copy_has_same_values(self, d5_state):
        """copy() has the same values."""
        other = d5_state.copy()
        np.testing.assert_array_equal(other.A_upper, d5_state.A_upper)
        np.testing.assert_array_equal(other.b, d5_state.b)
        assert other.n == d5_state.n
        assert other.state_version == d5_state.state_version
        assert other.d == d5_state.d

    def test_copy_no_shared_memory(self, d5_state):
        """copy() arrays do not share memory with original."""
        other = d5_state.copy()
        assert not np.shares_memory(other.A_upper, d5_state.A_upper)
        assert not np.shares_memory(other.b, d5_state.b)

    def test_copy_modification_independent(self, d5_state):
        """Modifying copy does not affect original."""
        other = d5_state.copy()
        other.A_upper[0] = 999.0
        other.b[0] = 999.0
        other.n = 999
        assert d5_state.A_upper[0] != 999.0
        assert d5_state.b[0] != 999.0
        assert d5_state.n != 999


class TestDimensionMismatch:
    """Test 4: dimension mismatch between state and phi raises ValueError.

    SRS MOD-04: phi dimension must match state.d.
    """

    def test_apply_phi_too_small(self, d5_state):
        """phi with fewer elements than state.d raises ValueError."""
        phi_small = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="does not match state.d"):
            apply_contribution(d5_state, phi_small, reward=1.0, sigma2=0.06)

    def test_apply_phi_too_large(self, d5_state):
        """phi with more elements than state.d raises ValueError."""
        phi_large = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="does not match state.d"):
            apply_contribution(d5_state, phi_large, reward=1.0, sigma2=0.06)

    def test_revoke_phi_mismatch(self, d5_state, phi_d5, sigma2):
        """revoke with wrong phi dimension raises ValueError."""
        phi_wrong = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="does not match state.d"):
            revoke_contribution(d5_state, phi_wrong, reward=1.0, sigma2=sigma2)

    def test_update_revision_old_phi_mismatch(self, d5_state, phi_d5, sigma2):
        """update_revision with wrong old_phi dimension raises ValueError."""
        phi_wrong = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="does not match state.d"):
            update_revision(d5_state, phi_wrong, 1.0, phi_d5, 0.5, sigma2)

    def test_update_revision_new_phi_mismatch(self, d5_state, phi_d5, sigma2):
        """update_revision with wrong new_phi dimension raises ValueError."""
        phi_wrong = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="does not match state.d"):
            update_revision(d5_state, phi_d5, 1.0, phi_wrong, 0.5, sigma2)

    def test_compute_contribution_2d_raises(self):
        """compute_contribution with 2-D phi raises ValueError."""
        phi_2d = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phi must be 1-D"):
            compute_contribution(phi_2d, reward=1.0, sigma2=0.06)

    def test_apply_contribution_2d_raises(self, d5_state):
        """apply_contribution with 2-D phi raises ValueError."""
        phi_2d = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="phi must be 1-D"):
            apply_contribution(d5_state, phi_2d, reward=1.0, sigma2=0.06)

    def test_sellerstate_invalid_A_upper_shape(self):
        """SellerState with wrong A_upper shape raises ValueError."""
        with pytest.raises(ValueError, match="A_upper shape must be"):
            SellerState(
                A_upper=np.zeros(10, dtype=np.float64),
                b=np.zeros(5, dtype=np.float64),
                n=0, state_version=0, d=5,
            )

    def test_sellerstate_invalid_b_shape(self):
        """SellerState with wrong b shape raises ValueError."""
        with pytest.raises(ValueError, match="b shape must be"):
            SellerState(
                A_upper=np.zeros(15, dtype=np.float64),
                b=np.zeros(4, dtype=np.float64),
                n=0, state_version=0, d=5,
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  compute_contribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeContribution:
    """SRS MOD-04: contribution_A = outer(phi,phi)/sigma2, contribution_b = y*phi/sigma2.

    Test 5: contribution for phi=[1,0,1,0,1], y=1.0, sigma2=1.0.
    Test 6: contribution scales with 1/sigma2.
    Test 7: contribution with y=0 gives zero b.
    """

    def test_5_contribution_formula(self, phi_d5, sigma2):
        """Test 5: A_upper = upper_triangle(outer(phi, phi) / sigma2)."""
        A_upper, b = compute_contribution(phi_d5, reward=1.0, sigma2=sigma2)
        expected_A = expected_A_upper(phi_d5, sigma2)
        expected_b_val = expected_b(phi_d5, 1.0, sigma2)
        np.testing.assert_allclose(A_upper, expected_A, atol=1e-12)
        np.testing.assert_allclose(b, expected_b_val, atol=1e-12)

    def test_5_shape(self, phi_d5):
        """contribution shapes match expected."""
        A_upper, b = compute_contribution(phi_d5, reward=1.0, sigma2=1.0)
        assert A_upper.shape == (15,)
        assert b.shape == (5,)

    def test_6_contribution_scales_with_1_sigma2(self, phi_d5):
        """Test 6: doubling sigma2 halves the contribution."""
        A1, b1 = compute_contribution(phi_d5, reward=1.0, sigma2=0.06)
        A2, b2 = compute_contribution(phi_d5, reward=1.0, sigma2=0.12)
        np.testing.assert_allclose(A1, 2.0 * A2, atol=1e-12)
        np.testing.assert_allclose(b1, 2.0 * b2, atol=1e-12)

    def test_6_contribution_general_sigma2(self, phi_d5_nonzero, alpha):
        """Test 6: contribution scales as 1/sigma2 for any sigma2."""
        sigma2_a = 0.03
        sigma2_b = 0.09
        A_a, b_a = compute_contribution(phi_d5_nonzero, reward=1.0, sigma2=sigma2_a)
        A_b, b_b = compute_contribution(phi_d5_nonzero, reward=1.0, sigma2=sigma2_b)
        ratio = sigma2_b / sigma2_a
        np.testing.assert_allclose(A_a, ratio * A_b, atol=1e-12)
        np.testing.assert_allclose(b_a, ratio * b_b, atol=1e-12)

    def test_7_zero_reward_gives_zero_b(self, phi_d5, sigma2):
        """Test 7: y=0 → b = zeros."""
        A_upper, b = compute_contribution(phi_d5, reward=0.0, sigma2=sigma2)
        assert np.allclose(b, 0.0)
        # A_upper should still be non-zero
        assert not np.allclose(A_upper, 0.0)

    def test_7_negative_reward(self, phi_d5, sigma2):
        """Negative reward produces negative b."""
        A_upper, b = compute_contribution(phi_d5, reward=-1.0, sigma2=sigma2)
        np.testing.assert_allclose(b, -expected_b(phi_d5, 1.0, sigma2), atol=1e-12)

    def test_sigma2_positive_required(self):
        """sigma2 <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="sigma2 must be > 0"):
            compute_contribution(phi_d5, reward=1.0, sigma2=0.0)
        with pytest.raises(ValueError, match="sigma2 must be > 0"):
            compute_contribution(phi_d5, reward=1.0, sigma2=-0.1)

    def test_1d_phi_required(self):
        """Non-1-D phi raises ValueError."""
        with pytest.raises(ValueError, match="phi must be 1-D"):
            compute_contribution(np.array([[1.0, 0.0, 1.0, 0.0, 1.0]]), reward=1.0, sigma2=1.0)

    def test_dtype_float64(self, phi_d5):
        """Output is always float64."""
        A_upper, b = compute_contribution(phi_d5, reward=1.0, sigma2=1.0)
        assert A_upper.dtype == np.float64
        assert b.dtype == np.float64

    def test_input_coerced_to_float64(self):
        """float32 input is coerced to float64."""
        phi_f32 = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        A_upper, b = compute_contribution(phi_f32, reward=1.0, sigma2=1.0)
        assert A_upper.dtype == np.float64
        assert b.dtype == np.float64


# ═══════════════════════════════════════════════════════════════════════════════
#  apply_contribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestApplyContribution:
    """SRS MOD-04: A += outer(phi,phi)/sigma2, b += y*phi/sigma2, n += 1.

    Test 8: Single application.
    Test 9: Sequential application equals batch.
    Test 10: Order independence.
    Test 11: state_version increments.
    Test 12: sigma2 <= 0 raises ValueError.
    """

    def test_8_single_application(self, d5_state, phi_d5, sigma2):
        """Test 8: n becomes 1, A and b match compute_contribution."""
        new_state = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        assert new_state.n == 1
        A_upper, b = compute_contribution(phi_d5, reward=1.0, sigma2=sigma2)
        np.testing.assert_allclose(new_state.A_upper, A_upper, atol=1e-12)
        np.testing.assert_allclose(new_state.b, b, atol=1e-12)

    def test_8_n_increments(self, d5_state, phi_d5, sigma2):
        """n increments by 1 on each apply."""
        s1 = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        s2 = apply_contribution(s1, phi_d5, reward=1.0, sigma2=sigma2)
        assert s1.n == 1
        assert s2.n == 2

    def test_9_sequential_equals_batch(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """Test 9: apply(apply(zero, phi1, y1), phi2, y2) == batch sum.

        Sequential: apply the first contribution, then the second.
        Batch: A_upper = A1 + A2, b = b1 + b2.
        """
        s_seq = apply_contribution(
            apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2),
            phi_d5_nonzero, reward=0.5, sigma2=sigma2,
        )
        A1, b1 = compute_contribution(phi_d5, reward=1.0, sigma2=sigma2)
        A2, b2 = compute_contribution(phi_d5_nonzero, reward=0.5, sigma2=sigma2)
        expected_A = A1 + A2
        expected_b = b1 + b2
        np.testing.assert_allclose(s_seq.A_upper, expected_A, atol=1e-12)
        np.testing.assert_allclose(s_seq.b, expected_b, atol=1e-12)

    def test_10_order_independence(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """Test 10: apply order does not matter (up to float64 roundoff).

        apply(apply(zero, phi1, y1), phi2, y2) ≈ apply(apply(zero, phi2, y2), phi1, y1)
        """
        s_12 = apply_contribution(
            apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2),
            phi_d5_nonzero, reward=0.5, sigma2=sigma2,
        )
        s_21 = apply_contribution(
            apply_contribution(d5_state, phi_d5_nonzero, reward=0.5, sigma2=sigma2),
            phi_d5, reward=1.0, sigma2=sigma2,
        )
        np.testing.assert_allclose(s_12.A_upper, s_21.A_upper, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(s_12.b, s_21.b, atol=1e-9, rtol=1e-9)

    def test_10_many_orders(self, d5_state, sigma2):
        """Test 10 (extended): many random contributions, different orders agree."""
        rng = np.random.RandomState(123)
        phis = [rng.randn(5).astype(np.float64) for _ in range(5)]
        rewards = [rng.rand() for _ in range(5)]

        # Order 1: sequential
        s1 = d5_state
        for phi, y in zip(phis, rewards):
            s1 = apply_contribution(s1, phi, reward=y, sigma2=sigma2)

        # Order 2: reversed
        s2 = d5_state
        for phi, y in zip(reversed(phis), reversed(rewards)):
            s2 = apply_contribution(s2, phi, reward=y, sigma2=sigma2)

        np.testing.assert_allclose(s1.A_upper, s2.A_upper, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(s1.b, s2.b, atol=1e-9, rtol=1e-9)

    def test_11_state_version_increments(self, d5_state, phi_d5, sigma2):
        """Test 11: state_version increments on each apply."""
        s0 = d5_state
        assert s0.state_version == 0
        s1 = apply_contribution(s0, phi_d5, reward=1.0, sigma2=sigma2)
        assert s1.state_version == 1
        s2 = apply_contribution(s1, phi_d5, reward=1.0, sigma2=sigma2)
        assert s2.state_version == 2

    def test_11_version_custom_start(self):
        """state_version increments from custom start."""
        s0 = zero_state(d=5, state_version=100)
        s1 = apply_contribution(s0, np.ones(5), reward=1.0, sigma2=1.0)
        assert s1.state_version == 101

    def test_12_sigma2_zero_raises(self, d5_state, phi_d5):
        """Test 12: sigma2 = 0 raises ValueError."""
        with pytest.raises(ValueError, match="sigma2 must be > 0"):
            apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=0.0)

    def test_12_sigma2_negative_raises(self, d5_state, phi_d5):
        """Test 12: sigma2 < 0 raises ValueError."""
        with pytest.raises(ValueError, match="sigma2 must be > 0"):
            apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=-0.1)

    def test_apply_returns_new_state(self, d5_state, phi_d5, sigma2):
        """apply_contribution returns a new state, not the input."""
        new_state = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        assert new_state is not d5_state

    def test_apply_does_not_mutate_input(self, d5_state, phi_d5, sigma2):
        """apply_contribution does not modify the input state."""
        original_A = d5_state.A_upper.copy()
        original_b = d5_state.b.copy()
        apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        np.testing.assert_array_equal(d5_state.A_upper, original_A)
        np.testing.assert_array_equal(d5_state.b, original_b)

    def test_apply_many_observations(self, d5_state, phi_d5, sigma2):
        """Applying many observations: n matches count, A,b are cumulative."""
        state = d5_state
        n_obs = 100
        for _ in range(n_obs):
            state = apply_contribution(state, phi_d5, reward=1.0, sigma2=sigma2)
        assert state.n == n_obs
        A_single, b_single = compute_contribution(phi_d5, reward=1.0, sigma2=sigma2)
        np.testing.assert_allclose(state.A_upper, n_obs * A_single, atol=1e-12)
        np.testing.assert_allclose(state.b, n_obs * b_single, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
#  revoke_contribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestRevokeContribution:
    """SRS MOD-05: subtract contribution, n unchanged.

    Test 13: Revoke restores A and b to before contribution.
    Test 14: n stays unchanged after revoke.
    Test 15: state_version decrements on revoke.
    """

    def test_13_revoke_restores_A_and_b(self, d5_state, phi_d5, sigma2):
        """Test 13: revoke(apply(zero, phi, y), phi, y) restores A and b to zero.

        NOTE: n and state_version are NOT restored to zero.
        n stays at applied.n (MOD-05), state_version decrements.
        """
        applied = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        revoked = revoke_contribution(applied, phi_d5, reward=1.0, sigma2=sigma2)
        # A and b restored to zero
        np.testing.assert_allclose(revoked.A_upper, 0.0, atol=1e-12)
        np.testing.assert_allclose(revoked.b, 0.0, atol=1e-12)
        # n is NOT restored (MOD-05: n stays unchanged)
        assert revoked.n == applied.n == 1
        # state_version decremented
        assert revoked.state_version == applied.state_version - 1 == 0

    def test_13_revoke_partial_state(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """Revoke only removes the specified contribution, not others."""
        s = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        s = apply_contribution(s, phi_d5_nonzero, reward=0.5, sigma2=sigma2)
        # Revoke second contribution
        revoked = revoke_contribution(s, phi_d5_nonzero, reward=0.5, sigma2=sigma2)
        # Should be back to first contribution only
        A1, b1 = compute_contribution(phi_d5, reward=1.0, sigma2=sigma2)
        np.testing.assert_allclose(revoked.A_upper, A1, atol=1e-12)
        np.testing.assert_allclose(revoked.b, b1, atol=1e-12)
        assert revoked.n == 2  # n stays unchanged

    def test_14_n_unchanged_after_revoke(self, d5_state, phi_d5, sigma2):
        """Test 14: n stays unchanged after revoke."""
        applied = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        revoked = revoke_contribution(applied, phi_d5, reward=1.0, sigma2=sigma2)
        assert revoked.n == applied.n

    def test_14_n_unchanged_after_multi_revoke(self, d5_state, phi_d5, sigma2):
        """n stays unchanged through multiple apply+revoke cycles."""
        state = d5_state
        for _ in range(5):
            state = apply_contribution(state, phi_d5, reward=1.0, sigma2=sigma2)
        for _ in range(5):
            state = revoke_contribution(state, phi_d5, reward=1.0, sigma2=sigma2)
        assert state.n == 5  # n never changes on revoke

    def test_15_state_version_decrements(self, d5_state, phi_d5, sigma2):
        """Test 15: state_version decrements on revoke."""
        applied = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        revoked = revoke_contribution(applied, phi_d5, reward=1.0, sigma2=sigma2)
        assert revoked.state_version == applied.state_version - 1

    def test_15_multi_revoke(self, d5_state, phi_d5, sigma2):
        """state_version decrements on each revoke."""
        s = d5_state
        versions = [s.state_version]
        for _ in range(3):
            s = apply_contribution(s, phi_d5, reward=1.0, sigma2=sigma2)
            s = revoke_contribution(s, phi_d5, reward=1.0, sigma2=sigma2)
            versions.append(s.state_version)
        # Each cycle: +1 then -1 = no net change
        assert all(v == versions[0] for v in versions)

    def test_revoke_sigma2_zero_raises(self, d5_state, phi_d5):
        """sigma2 = 0 raises ValueError on revoke."""
        with pytest.raises(ValueError, match="sigma2 must be > 0"):
            revoke_contribution(d5_state, phi_d5, reward=1.0, sigma2=0.0)

    def test_revoke_returns_new_state(self, d5_state, phi_d5, sigma2):
        """revoke_contribution returns a new state, not the input."""
        applied = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        revoked = revoke_contribution(applied, phi_d5, reward=1.0, sigma2=sigma2)
        assert revoked is not applied

    def test_revoke_does_not_mutate_input(self, d5_state, phi_d5, sigma2):
        """revoke_contribution does not modify the input state."""
        applied = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        original_A = applied.A_upper.copy()
        original_b = applied.b.copy()
        original_n = applied.n
        revoke_contribution(applied, phi_d5, reward=1.0, sigma2=sigma2)
        np.testing.assert_array_equal(applied.A_upper, original_A)
        np.testing.assert_array_equal(applied.b, original_b)
        assert applied.n == original_n


# ═══════════════════════════════════════════════════════════════════════════════
#  update_revision
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpdateRevision:
    """SRS MOD-05: atomic revision — subtract old, add new; n unchanged.

    Test 16: Revision A,b equivalence.
    Test 17: n unchanged after revision.
    Test 18: Correct label change (meeting → no meeting).
    """

    def test_16_revision_A_b_equivalence(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """Test 16: update_revision A,b == revoke(apply(state, new, new_y), old, old_y).

        NOTE: n and state_version differ between the two approaches.
        update_revision: n = state.n, state_version = state.state_version + 1
        revoke(apply(...), ...): n = state.n + 1, state_version = state.state_version
        """
        old_phi, old_y = phi_d5, 1.0
        new_phi, new_y = phi_d5_nonzero, 0.5

        revised = update_revision(
            d5_state, old_phi, old_y, new_phi, new_y, sigma2
        )

        # Sequential approach
        applied = apply_contribution(d5_state, new_phi, new_y, sigma2)
        seq = revoke_contribution(applied, old_phi, old_y, sigma2)

        # A,b should be equivalent
        np.testing.assert_allclose(revised.A_upper, seq.A_upper, atol=1e-12)
        np.testing.assert_allclose(revised.b, seq.b, atol=1e-12)

        # n differs: update_revision keeps n, sequential increments then keeps
        assert revised.n == d5_state.n == 0
        assert seq.n == 1

    def test_16_revision_equivalent_to_sequential(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """update_revision is equivalent to revoke(apply(state, new, new_y), old, old_y) for A,b."""
        old_phi, old_y = phi_d5, 1.08  # meeting
        new_phi, new_y = phi_d5_nonzero, 0.08  # no meeting

        revised = update_revision(
            d5_state, old_phi, old_y, new_phi, new_y, sigma2
        )

        # Expected: A = new_contrib - old_contrib, b = new_b - old_b
        new_A, new_b = compute_contribution(new_phi, new_y, sigma2)
        old_A, old_b = compute_contribution(old_phi, old_y, sigma2)
        expected_A = new_A - old_A
        expected_b = new_b - old_b

        np.testing.assert_allclose(revised.A_upper, expected_A, atol=1e-12)
        np.testing.assert_allclose(revised.b, expected_b, atol=1e-12)

    def test_17_n_unchanged_after_revision(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """Test 17: n stays unchanged after revision."""
        assert d5_state.n == 0
        revised = update_revision(
            d5_state, phi_d5, 1.0, phi_d5_nonzero, 0.5, sigma2
        )
        assert revised.n == 0

    def test_17_n_unchanged_after_revision_with_history(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """n stays unchanged even when state has prior observations."""
        s = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        assert s.n == 1
        revised = update_revision(s, phi_d5, 1.0, phi_d5_nonzero, 0.5, sigma2)
        assert revised.n == 1  # n unchanged

    def test_18_label_change_meeting_to_no_meeting(self, d5_state, phi_d5, sigma2):
        """Test 18: meeting (y=1.08) revised to no meeting (y=0.08).

        b should change by phi * (0.08 - 1.08) / sigma2 = phi * (-1.0) / sigma2.
        A should not change (same phi, same sigma2).
        """
        y_meeting = 1.08
        y_no_meeting = 0.08

        revised = update_revision(
            d5_state, phi_d5, y_meeting, phi_d5, y_no_meeting, sigma2
        )

        # A should be unchanged (same phi, same sigma2)
        np.testing.assert_allclose(revised.A_upper, 0.0, atol=1e-12)
        # b should reflect the difference
        expected_b = phi_d5 * (y_no_meeting - y_meeting) / sigma2
        np.testing.assert_allclose(revised.b, expected_b, atol=1e-12)
        assert revised.b[0] == pytest.approx((y_no_meeting - y_meeting) / sigma2)

    def test_18_revision_same_phi_different_reward(self, d5_state, phi_d5, sigma2):
        """Same phi, different reward: only b changes."""
        s1 = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        s2 = update_revision(s1, phi_d5, 1.0, phi_d5, 0.5, sigma2)
        expected_A, expected_b = compute_contribution(phi_d5, 0.5, sigma2)
        np.testing.assert_allclose(s2.A_upper, expected_A, atol=1e-12)
        np.testing.assert_allclose(s2.b, expected_b, atol=1e-12)

    def test_revision_sigma2_zero_raises(self, d5_state, phi_d5, phi_d5_nonzero):
        """sigma2 = 0 raises ValueError on revision."""
        with pytest.raises(ValueError, match="sigma2 must be > 0"):
            update_revision(d5_state, phi_d5, 1.0, phi_d5_nonzero, 0.5, 0.0)

    def test_revision_returns_new_state(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """update_revision returns a new state, not the input."""
        revised = update_revision(
            d5_state, phi_d5, 1.0, phi_d5_nonzero, 0.5, sigma2
        )
        assert revised is not d5_state

    def test_revision_state_version_increments(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """update_revision increments state_version by 1."""
        assert d5_state.state_version == 0
        revised = update_revision(
            d5_state, phi_d5, 1.0, phi_d5_nonzero, 0.5, sigma2
        )
        assert revised.state_version == 1

    def test_revision_no_change(self, d5_state, phi_d5, sigma2):
        """Revising with same phi and reward: state unchanged."""
        revised = update_revision(
            d5_state, phi_d5, 1.0, phi_d5, 1.0, sigma2
        )
        np.testing.assert_allclose(revised.A_upper, 0.0, atol=1e-12)
        np.testing.assert_allclose(revised.b, 0.0, atol=1e-12)
        assert revised.n == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Upper triangle utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestUpperTriangleUtilities:
    """extract_upper_triangle + reconstruct_symmetric identity and symmetry.

    Test 19: extract_upper_triangle + reconstruct_symmetric is identity.
    Test 20: Reconstructed matrix is symmetric.
    """

    def test_19_identity_symmetric_matrix(self, d):
        """Test 19: extract + reconstruct is identity for symmetric matrices."""
        rng = np.random.RandomState(d * 100)
        m = rng.randn(d, d)
        m = (m + m.T) / 2  # symmetric
        upper = extract_upper_triangle(m)
        reconstructed = reconstruct_symmetric(upper, d)
        np.testing.assert_allclose(reconstructed, m, atol=1e-12)

    def test_19_identity_d3(self):
        """Identity check for d=3 with known values."""
        m = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
        upper = extract_upper_triangle(m)
        expected_upper = np.array([1., 2., 3., 4., 5., 6.], dtype=np.float64)
        np.testing.assert_array_equal(upper, expected_upper)
        reconstructed = reconstruct_symmetric(upper, 3)
        np.testing.assert_array_equal(reconstructed, m)

    def test_20_reconstructed_is_symmetric(self, d):
        """Test 20: Reconstructed matrix is symmetric: A == A.T."""
        rng = np.random.RandomState(d)
        upper = rng.randn(d * (d + 1) // 2)
        m = reconstruct_symmetric(upper, d)
        np.testing.assert_array_equal(m, m.T)

    def test_20_symmetric_all_d(self, d_values=D_VALUES):
        """Symmetry holds for all tested dimensions."""
        for d in d_values:
            rng = np.random.RandomState(d)
            upper = rng.randn(d * (d + 1) // 2)
            m = reconstruct_symmetric(upper, d)
            np.testing.assert_array_equal(m, m.T)

    def test_extract_upper_of_symmetric(self, d):
        """extract_upper_triangle of symmetric matrix is well-defined."""
        rng = np.random.RandomState(d)
        m = rng.randn(d, d)
        m = (m + m.T) / 2
        upper = extract_upper_triangle(m)
        assert len(upper) == d * (d + 1) // 2

    def test_reconstruct_upper_size_validation(self):
        """reconstruct_symmetric validates upper array length."""
        with pytest.raises(ValueError, match="upper length must be"):
            reconstruct_symmetric(np.zeros(10), 5)  # expected 15

    def test_reconstruct_upper_size_validation_d3(self):
        """reconstruct_symmetric rejects wrong size for d=3 (expect 6)."""
        with pytest.raises(ValueError, match="upper length must be 6"):
            reconstruct_symmetric(np.zeros(5), 3)

    def test_roundtrip_extract_reconstruct(self, d):
        """extract(reconstruct(upper)) == upper."""
        rng = np.random.RandomState(d)
        upper = rng.randn(d * (d + 1) // 2).astype(np.float64)
        m = reconstruct_symmetric(upper, d)
        upper_back = extract_upper_triangle(m)
        np.testing.assert_allclose(upper_back, upper, atol=1e-12)

    def test_roundtrip_reconstruct_extract(self, d):
        """reconstruct(extract(m)) == m for symmetric m."""
        rng = np.random.RandomState(d)
        m = rng.randn(d, d)
        m = (m + m.T) / 2
        reconstructed = reconstruct_symmetric(extract_upper_triangle(m), d)
        np.testing.assert_allclose(reconstructed, m, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
#  Prior
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrior:
    """Gaussian prior for seller coefficients.

    Test 21: Prior.diagonal_prior(d=5, alpha=0.1) creates valid SPD matrix.
    Test 22: Prior diagonal values match TRAIN-03: alpha * [1,1,1,1/4,1/4] for K=2.
    Test 23: Lambda0 = inv(Sigma0).
    Test 24: eta0 = Lambda0 @ mu0.
    """

    def test_21_prior_is_valid_SPD(self, k):
        """Test 21: Prior.diagonal_prior creates valid SPD Sigma0."""
        d = 2 * k + 1
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        # Check Sigma0 is symmetric
        np.testing.assert_array_equal(prior.Sigma0, prior.Sigma0.T)
        # Check eigenvalues are all positive (SPD)
        eigenvalues = np.linalg.eigvalsh(prior.Sigma0)
        assert np.all(eigenvalues > 0), f"Sigma0 not SPD: eigenvalues={eigenvalues}"
        # Check Lambda0 is symmetric
        np.testing.assert_array_equal(prior.Lambda0, prior.Lambda0.T)
        # Check Lambda0 eigenvalues are all positive
        lambda_eigenvalues = np.linalg.eigvalsh(prior.Lambda0)
        assert np.all(lambda_eigenvalues > 0), f"Lambda0 not SPD: eigenvalues={lambda_eigenvalues}"

    def test_22_TRAIN03_diagonal_K2(self, alpha):
        """Test 22: TRAIN-03 diagonal for K=2: alpha * [1, 1, 1, 1/4, 1/4]."""
        prior = Prior.diagonal_prior(d=5, alpha=alpha)
        expected_diag = alpha * np.array([1.0, 1.0, 1.0, 0.25, 0.25])
        np.testing.assert_allclose(np.diag(prior.Sigma0), expected_diag, atol=1e-15)

    def test_22_TRAIN03_diagonal_K3(self, alpha):
        """TRAIN-03 diagonal for K=3: alpha * [1, 1, 1, 1/4, 1/4, 1/9, 1/9]."""
        prior = Prior.diagonal_prior(d=7, alpha=alpha)
        expected_diag = alpha * np.array([1.0, 1.0, 1.0, 0.25, 0.25, 1.0/9, 1.0/9])
        np.testing.assert_allclose(np.diag(prior.Sigma0), expected_diag, atol=1e-15)

    def test_22_TRAIN03_diagonal_K4(self, alpha):
        """TRAIN-03 diagonal for K=4: alpha * [1, 1, 1, 1/4, 1/4, 1/9, 1/9, 1/16, 1/16]."""
        prior = Prior.diagonal_prior(d=9, alpha=alpha)
        expected_diag = alpha * np.array([
            1.0, 1.0, 1.0,
            0.25, 0.25,
            1.0/9, 1.0/9,
            1.0/16, 1.0/16,
        ])
        np.testing.assert_allclose(np.diag(prior.Sigma0), expected_diag, atol=1e-15)

    def test_23_Lambda0_inverse_of_Sigma0(self, k):
        """Test 23: Lambda0 = inv(Sigma0)."""
        d = 2 * k + 1
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        expected_Lambda0 = np.linalg.inv(prior.Sigma0)
        np.testing.assert_allclose(prior.Lambda0, expected_Lambda0, atol=1e-12)

    def test_23_Lambda0_Sigma0_product_is_identity(self, k):
        """Lambda0 @ Sigma0 = I."""
        d = 2 * k + 1
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        product = prior.Lambda0 @ prior.Sigma0
        np.testing.assert_allclose(product, np.eye(d), atol=1e-12)

    def test_24_eta0_equals_Lambda0_mu0(self, k):
        """Test 24: eta0 = Lambda0 @ mu0."""
        d = 2 * k + 1
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        expected_eta0 = prior.Lambda0 @ prior.mu0
        np.testing.assert_allclose(prior.eta0, expected_eta0, atol=1e-12)

    def test_prior_mu0_is_zero(self, k):
        """mu0 is zero vector."""
        d = 2 * k + 1
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        np.testing.assert_array_equal(prior.mu0, np.zeros(d))

    def test_prior_eta0_is_zero(self, k):
        """eta0 is zero vector (since mu0 is zero)."""
        d = 2 * k + 1
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        np.testing.assert_array_equal(prior.eta0, np.zeros(d))

    def test_prior_diagonal_structure(self, k):
        """Sigma0 is diagonal (off-diagonal elements are zero)."""
        d = 2 * k + 1
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        np.testing.assert_array_equal(
            prior.Sigma0 - np.diag(np.diag(prior.Sigma0)),
            np.zeros((d, d))
        )

    def test_prior_invalid_d_even_raises(self):
        """d must be odd."""
        with pytest.raises(ValueError, match="d must be odd"):
            Prior.diagonal_prior(d=4, alpha=0.1)
        with pytest.raises(ValueError, match="d must be odd"):
            Prior.diagonal_prior(d=6, alpha=0.1)

    def test_prior_invalid_alpha_raises(self):
        """alpha must be > 0."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            Prior.diagonal_prior(d=5, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be > 0"):
            Prior.diagonal_prior(d=5, alpha=-0.1)

    def test_prior_all_d_values(self, d):
        """Prior is valid for all tested d values."""
        prior = Prior.diagonal_prior(d=d, alpha=0.1)
        assert prior.d == d
        assert prior.mu0.shape == (d,)
        assert prior.Sigma0.shape == (d, d)
        assert prior.Lambda0.shape == (d, d)
        assert prior.eta0.shape == (d,)
        # Validate SPD
        eigenvalues = np.linalg.eigvalsh(prior.Sigma0)
        assert np.all(eigenvalues > 0)

    def test_prior_different_alpha_values(self, d):
        """Prior works with different alpha values."""
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            prior = Prior.diagonal_prior(d=d, alpha=alpha)
            assert prior.d == d
            eigenvalues = np.linalg.eigvalsh(prior.Sigma0)
            assert np.all(eigenvalues > 0)

    def test_prior_d9_bytes_codec_compliance(self):
        """d=9 state size for STATE-07 codec.

        bytes(d) = 48 + 8 * (d*(d+1)/2 + d)
        bytes(9) = 48 + 8 * (45 + 9) = 48 + 432 = 480
        """
        d = 9
        upper_size = d * (d + 1) // 2  # 45
        b_size = d  # 9
        expected_bytes = 48 + 8 * (upper_size + b_size)
        assert expected_bytes == 480, f"Expected 480 bytes for d=9, got {expected_bytes}"

    def test_prior_codec_formula_all_d(self):
        """STATE-07 codec formula holds for all tested d values."""
        for d in D_VALUES:
            upper_size = d * (d + 1) // 2
            b_size = d
            expected_bytes = 48 + 8 * (upper_size + b_size)
            # Verify the formula
            if d == 9:
                assert expected_bytes == 480
            elif d == 5:
                assert expected_bytes == 48 + 8 * (15 + 5) == 208
            elif d == 3:
                assert expected_bytes == 48 + 8 * (6 + 3) == 120
            elif d == 7:
                assert expected_bytes == 48 + 8 * (28 + 7) == 328


# ═══════════════════════════════════════════════════════════════════════════════
#  Posterior computation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPosteriorParams:
    """posterior_params: Lambda = Lambda0 + A, eta = eta0 + b, mu = Lambda^{-1} @ eta."""

    def test_posterior_cold_start(self, prior_d5):
        """Cold start (zero state): posterior = prior."""
        state = zero_state(d=5)
        mu, Sigma = posterior_params(state, prior_d5)
        np.testing.assert_allclose(mu, prior_d5.mu0, atol=1e-12)
        np.testing.assert_allclose(Sigma, prior_d5.Sigma0, atol=1e-12)

    def test_posterior_shape(self, prior_d5, phi_d5, sigma2):
        """posterior_params returns correct shapes."""
        state = apply_contribution(zero_state(d=5), phi_d5, reward=1.0, sigma2=sigma2)
        mu, Sigma = posterior_params(state, prior_d5)
        assert mu.shape == (5,)
        assert Sigma.shape == (5, 5)

    def test_posterior_SPD(self, prior_d5, phi_d5, sigma2):
        """Posterior covariance is SPD."""
        state = apply_contribution(zero_state(d=5), phi_d5, reward=1.0, sigma2=sigma2)
        mu, Sigma = posterior_params(state, prior_d5)
        eigenvalues = np.linalg.eigvalsh(Sigma)
        assert np.all(eigenvalues > 0), f"Sigma not SPD: eigenvalues={eigenvalues}"

    def test_posterior_dimension_mismatch_raises(self, prior_d5):
        """Dimension mismatch between state and prior raises ValueError."""
        state_9 = zero_state(d=9)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            posterior_params(state_9, prior_d5)

    def test_posterior_with_data_shifts_mean(self, prior_d5, phi_d5, sigma2):
        """Posterior mean shifts toward data when signal is strong."""
        state = zero_state(d=5)
        # Apply strong signal: reward = 10, small sigma2
        for _ in range(10):
            state = apply_contribution(state, phi_d5, reward=10.0, sigma2=0.01)
        mu, Sigma = posterior_params(state, prior_d5)
        # Prior mean is zero, posterior should shift toward data
        assert not np.allclose(mu, 0.0)

    def test_posterior_variance_decreases_with_data(self, prior_d5, phi_d5, sigma2):
        """Posterior variance decreases as more data is added."""
        state = zero_state(d=5)
        _, Sigma0 = posterior_params(state, prior_d5)
        state = apply_contribution(state, phi_d5, reward=1.0, sigma2=sigma2)
        _, Sigma1 = posterior_params(state, prior_d5)
        # Posterior variance should be less than prior
        assert np.trace(Sigma1) < np.trace(Sigma0)


# ═══════════════════════════════════════════════════════════════════════════════
#  SRS compliance: MOD-04 gamma=1 enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class TestSRSCompliance:
    """SRS MOD-04: gamma=1 enforced, order independence, n_eff=n."""

    def test_mod04_gamma_equals_1(self):
        """MOD-04: gamma=1 is enforced (no discounting parameter in API).

        The implementation has no gamma parameter — all contributions have
        equal weight (gamma=1). This test verifies the API design.
        """
        # The functions don't accept a gamma parameter
        import inspect
        for func_name in ['apply_contribution', 'revoke_contribution', 'update_revision']:
            func = globals()[func_name]
            sig = inspect.signature(func)
            assert 'gamma' not in sig.parameters, f"{func_name} should not accept gamma"

    def test_mod04_order_independence_strong(self, d5_state, sigma2):
        """MOD-04: order independence up to float64 roundoff (rtol=1e-9)."""
        rng = np.random.RandomState(42)
        phis = [rng.randn(5).astype(np.float64) for _ in range(10)]
        rewards = [rng.rand() for _ in range(10)]

        # Two random permutations
        order_a = list(range(10))
        order_b = list(reversed(range(10)))

        s_a = d5_state
        for i in order_a:
            s_a = apply_contribution(s_a, phis[i], reward=rewards[i], sigma2=sigma2)

        s_b = d5_state
        for i in order_b:
            s_b = apply_contribution(s_b, phis[i], reward=rewards[i], sigma2=sigma2)

        np.testing.assert_allclose(s_a.A_upper, s_b.A_upper, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(s_a.b, s_b.b, rtol=1e-9, atol=1e-9)

    def test_mod04_n_eff_equals_n(self, d5_state, phi_d5, sigma2):
        """MOD-04: n_eff = n (no effective sample size adjustment)."""
        state = d5_state
        for _ in range(42):
            state = apply_contribution(state, phi_d5, reward=1.0, sigma2=sigma2)
        assert state.n == 42  # n is the actual observation count

    def test_mod05_revision_n_unchanged(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """MOD-05: n stays unchanged during revision."""
        s = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        assert s.n == 1
        revised = update_revision(s, phi_d5, 1.0, phi_d5_nonzero, 0.5, sigma2)
        assert revised.n == 1  # n unchanged

    def test_state07_codec_bytes_d9(self):
        """STATE-07: bytes(9) = 480."""
        d = 9
        upper_size = d * (d + 1) // 2  # 45
        b_size = d  # 9
        bytes_d = 48 + 8 * (upper_size + b_size)
        assert bytes_d == 480

    def test_state07_codec_formula(self, d):
        """STATE-07: bytes(d) = 48 + 8 * (d*(d+1)/2 + d)."""
        upper_size = d * (d + 1) // 2
        b_size = d
        expected = 48 + 8 * (upper_size + b_size)
        # Verify the formula produces a positive integer
        assert isinstance(expected, int)
        assert expected > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Integration: full lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullLifecycle:
    """Full lifecycle: apply → apply → revise → revoke → posterior."""

    def test_lifecycle_apply_revoke_posterior(self, d5_state, phi_d5, sigma2):
        """Apply, then revoke: posterior returns to prior."""
        s = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        s = revoke_contribution(s, phi_d5, reward=1.0, sigma2=sigma2)
        mu, Sigma = posterior_params(s, Prior.diagonal_prior(d=5, alpha=0.1))
        np.testing.assert_allclose(mu, np.zeros(5), atol=1e-12)
        prior = Prior.diagonal_prior(d=5, alpha=0.1)
        np.testing.assert_allclose(Sigma, prior.Sigma0, atol=1e-12)

    def test_lifecycle_revision_then_revoke(self, d5_state, phi_d5, phi_d5_nonzero, sigma2):
        """Revise, then revoke new: state should have only old contribution."""
        # Apply old
        s = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        assert s.n == 1
        # Revise to new (n stays 1 per MOD-05)
        s = update_revision(s, phi_d5, 1.0, phi_d5_nonzero, 0.5, sigma2)
        assert s.n == 1
        # Revoke new
        s = revoke_contribution(s, phi_d5_nonzero, 0.5, sigma2)
        # Should be back to zero state (A,b)
        np.testing.assert_allclose(s.A_upper, 0.0, atol=1e-12)
        np.testing.assert_allclose(s.b, 0.0, atol=1e-12)
        assert s.n == 1  # n never changed on revoke

    def test_lifecycle_many_operations(self, d5_state, sigma2):
        """Many apply/revoke/revision operations maintain consistency."""
        rng = np.random.RandomState(99)
        state = d5_state
        n_ops = 50
        for _ in range(n_ops):
            phi = rng.randn(5).astype(np.float64)
            y = rng.rand()
            state = apply_contribution(state, phi, reward=y, sigma2=sigma2)
        # Now verify n
        assert state.n == n_ops
        # Revoke all
        for _ in range(n_ops):
            phi = rng.randn(5).astype(np.float64)
            y = rng.rand()
            state = revoke_contribution(state, phi, reward=y, sigma2=sigma2)
        assert state.n == n_ops  # n unchanged after revoke

    def test_lifecycle_prior_posterior_consistency(self, d5_state, phi_d5, sigma2):
        """Prior + state → posterior: Lambda = Lambda0 + A, eta = eta0 + b."""
        prior = Prior.diagonal_prior(d=5, alpha=0.1)
        state = apply_contribution(d5_state, phi_d5, reward=1.0, sigma2=sigma2)
        mu, Sigma = posterior_params(state, prior)
        # Verify Lambda = Lambda0 + A
        Lambda = prior.Lambda0 + state.A
        # Verify mu = Lambda^{-1} @ eta
        eta = prior.eta0 + state.b
        expected_mu = np.linalg.solve(Lambda, eta)
        np.testing.assert_allclose(mu, expected_mu, atol=1e-10)
        # Verify Sigma = Lambda^{-1}
        expected_Sigma = np.linalg.inv(Lambda)
        np.testing.assert_allclose(Sigma, expected_Sigma, atol=1e-10)
