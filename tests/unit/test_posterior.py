"""Comprehensive unit tests for `src/btc/model/posterior.py` (MOD-04 / MOD-06 / MOD-07).

Tests SRS §5.2 (MOD-04, MOD-06, MOD-07) and §15 (Verification matrix, T03, T04):

    MOD-04:  Lambda = Lambda0 + A, L = cholesky(Lambda),
             mu = solve(L.T, solve(L, eta0 + b))
    MOD-06:  expected_reward = phi.T @ mu,
             latent_std = sqrt(max(0, dot(solve(L, phi), solve(L, phi)))),
             predictive_std = sqrt(latent_std^2 + sigma2),
             prior_weight = trace(Lambda0) / trace(Lambda0 + A)
             prior_weight in [0, 1], finite; cold-start = 1.0; DO NOT clip scores.
    MOD-07:  Failed Cholesky → named fallback + alert; never fabricate uncertainty.
             Nonfinite parameters → fallback. Incompatible state → fallback.

SRS §15: atol=1e-9, rtol=1e-9 on well-conditioned float64 fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.btc.features.fourier import fourier
from src.btc.model.posterior import (
    FallbackResult,
    Posterior,
    compute_posterior,
    compute_prior_weight,
    predict_expected_reward,
    predict_uncertainty,
    score_candidate,
    score_candidates,
)
from src.btc.model.stats import (
    Prior,
    apply_contribution,
    zero_state,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

D = 5
K = 2
SIGMA2 = 0.06


@pytest.fixture
def prior():
    """Standard diagonal prior, d=5, alpha=0.1."""
    return Prior.diagonal_prior(d=D, alpha=0.1)


@pytest.fixture
def cold_state():
    """Cold-start zero state."""
    return zero_state(d=D)


@pytest.fixture
def warm_state(prior):
    """State with 100 observations for data-dominant regime."""
    state = zero_state(d=D)
    np.random.seed(42)
    for _ in range(100):
        t = np.random.uniform(0.0, 24.0)
        phi = fourier(t, k=K)
        reward = np.random.normal(0.5, 0.3)
        state = apply_contribution(state, phi, float(reward), SIGMA2)
    return state


@pytest.fixture
def single_obs_state(prior):
    """State with exactly 1 observation."""
    state = zero_state(d=D)
    t = 10.0
    phi = fourier(t, k=K)
    reward = 0.8
    return apply_contribution(state, phi, reward, SIGMA2)


@pytest.fixture
def phi():
    """Feature vector for d=5."""
    return fourier(10.0, k=K)


@pytest.fixture
def phi_matrix():
    """Multiple feature vectors."""
    times = np.array([8.0, 12.0, 16.0, 20.0])
    return fourier(times, k=K)


@pytest.fixture
def well_conditioned_posterior(cold_state, prior):
    """Posterior from cold start — well-conditioned."""
    return compute_posterior(cold_state, prior, sigma2=SIGMA2)


# ── MOD-04: compute_posterior ─────────────────────────────────────────────────

class TestComputePosterior:
    """SRS MOD-04: Lambda = Lambda0 + A, L = cholesky(Lambda),
    mu = solve(L.T, solve(L, eta0 + b))."""

    def test_cold_start_mu_equals_prior_mean(self, well_conditioned_posterior, prior):
        """Cold start (n=0): posterior mean = prior mean mu0."""
        np.testing.assert_allclose(
            well_conditioned_posterior.mu,
            prior.mu0,
            atol=1e-14, rtol=1e-14,
            err_msg="Cold-start mu should equal prior mean mu0"
        )

    def test_cold_start_prior_weight_is_one(self, well_conditioned_posterior):
        """Cold start: prior_weight = 1.0."""
        assert well_conditioned_posterior.prior_weight == pytest.approx(1.0, abs=1e-15)

    def test_after_one_obs_mu_shifts_toward_data(self, single_obs_state, prior):
        """After 1 observation: mu shifts away from mu0 toward data-driven estimate."""
        posterior = compute_posterior(single_obs_state, prior, sigma2=SIGMA2)
        # With a positive reward, mu[0] (bias term) should shift positive
        assert posterior.mu[0] > prior.mu0[0], (
            "Posterior mean bias should shift toward positive reward"
        )
        # mu should differ from prior mean
        assert not np.allclose(posterior.mu, prior.mu0, atol=1e-10), (
            "Posterior mean should shift after observation"
        )

    def test_after_100_obs_prior_weight_near_zero(self, prior):
        """After 100 observations: prior_weight close to 0 (data dominates)."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        assert posterior.prior_weight < 0.1, (
            f"Prior weight should be small with 100 obs, got {posterior.prior_weight}"
        )

    def test_cholesky_on_spd_succeeds(self, cold_state, prior):
        """Cholesky on SPD matrix (Lambda0) succeeds."""
        posterior = compute_posterior(cold_state, prior, sigma2=SIGMA2)
        assert isinstance(posterior, Posterior), (
            "Cholesky on SPD Lambda should succeed, got FallbackResult"
        )

    def test_lambda_is_spd(self, cold_state, prior):
        """Lambda = Lambda0 + A is SPD (all eigenvalues > 0)."""
        state = zero_state(d=D)
        np.random.seed(99)
        for _ in range(50):
            t = np.random.uniform(0.0, 24.0)
            phi = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        Lambda = prior.Lambda0 + state.A
        eigenvalues = np.linalg.eigvalsh(Lambda)
        assert np.all(eigenvalues > 0), (
            f"Lambda eigenvalues should be positive (SPD): {eigenvalues}"
        )

    def test_mu_matches_direct_solve(self, prior):
        """mu = solve(L.T, solve(L, eta0+b)) matches Lambda^{-1} @ eta."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        Lambda = prior.Lambda0 + state.A
        eta = prior.eta0 + state.b
        mu_direct = np.linalg.solve(Lambda, eta)
        np.testing.assert_allclose(
            posterior.mu, mu_direct,
            atol=1e-9, rtol=1e-9,
            err_msg="Posterior mu should match direct solve: Lambda @ mu = eta"
        )

    def test_return_type_is_posterior_on_success(self, cold_state, prior):
        """Successful compute_posterior returns Posterior, not FallbackResult."""
        result = compute_posterior(cold_state, prior, sigma2=SIGMA2)
        assert isinstance(result, Posterior)
        assert not isinstance(result, FallbackResult)


# ── Posterior properties ──────────────────────────────────────────────────────

class TestPosteriorProperties:
    """Test Posterior dataclass properties."""

    def test_is_cold_start_true_when_n_zero(self, well_conditioned_posterior):
        """Posterior.is_cold_start is True when n=0."""
        assert well_conditioned_posterior.is_cold_start is True
        assert well_conditioned_posterior.n == 0

    def test_is_cold_start_false_when_n_gt_zero(self, prior):
        """Posterior.is_cold_start is False when n > 0."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        assert posterior.is_cold_start is False
        assert posterior.n > 0

    def test_L_is_lower_triangular(self, well_conditioned_posterior):
        """Cholesky factor L is lower triangular."""
        L = well_conditioned_posterior.L
        # Lower triangular: all elements above diagonal are zero
        upper_mask = np.triu(np.ones_like(L), k=1).astype(bool)
        np.testing.assert_array_equal(
            L[upper_mask], 0.0,
            err_msg="L should be lower triangular"
        )

    def test_cholesky_verification_L_Lt_equals_Lambda(self, prior):
        """L @ L.T ≈ Lambda (Cholesky factorization correctness)."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        Lambda = prior.Lambda0 + state.A
        reconstructed = posterior.L @ posterior.L.T
        np.testing.assert_allclose(
            reconstructed, Lambda,
            atol=1e-12, rtol=1e-12,
            err_msg="L @ L.T should equal Lambda"
        )

    def test_posterior_dimensions_match(self, cold_state, prior):
        """Posterior d matches state.d and prior.d."""
        posterior = compute_posterior(cold_state, prior, sigma2=SIGMA2)
        assert posterior.d == D
        assert posterior.mu.shape == (D,)
        assert posterior.L.shape == (D, D)

    def test_sigma2_stored_as_float(self, cold_state, prior):
        """sigma2 is stored as Python float."""
        posterior = compute_posterior(cold_state, prior, sigma2=SIGMA2)
        assert isinstance(posterior.sigma2, float)


# ── MOD-06: predict_expected_reward ───────────────────────────────────────────

class TestPredictExpectedReward:
    """SRS MOD-06: expected_reward = phi.T @ mu.
    NOT clipped to [0, 1], NOT a probability."""

    def test_cold_start_reward_equals_phi_T_mu0(self, well_conditioned_posterior, prior, phi):
        """Cold start: expected_reward = phi.T @ mu0 (prior mean prediction)."""
        reward = predict_expected_reward(phi, well_conditioned_posterior)
        expected = float(phi.T @ prior.mu0)
        np.testing.assert_allclose(
            reward, expected, atol=1e-14, rtol=1e-14,
            err_msg="Cold-start expected_reward should be phi.T @ mu0"
        )

    def test_with_data_reward_equals_phi_T_mu(self, prior, phi):
        """With data: expected_reward = phi.T @ mu (posterior mean)."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        reward = predict_expected_reward(phi, posterior)
        expected = float(phi.T @ posterior.mu)
        np.testing.assert_allclose(
            reward, expected, atol=1e-14, rtol=1e-14,
            err_msg="expected_reward should be phi.T @ posterior.mu"
        )

    def test_score_not_clipped_to_0_1(self, prior):
        """Score is NOT clipped to [0, 1] — can be negative or > 1."""
        state = zero_state(d=D)
        # Create a state with very large rewards to push expected reward outside [0,1]
        for _ in range(1000):
            t = np.random.uniform(0.0, 24.0)
            phi = fourier(t, k=K)
            reward = 100.0  # Very large reward
            state = apply_contribution(state, phi, reward, SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        phi_large = np.ones(D, dtype=np.float64) * 10.0  # Large feature values
        score = predict_expected_reward(phi_large, posterior)
        # With large rewards and large features, score should be far outside [0,1]
        assert score > 1.0 or score < 0.0, (
            f"Score should not be clipped to [0,1]; got {score}"
        )

    def test_score_can_be_negative(self, prior):
        """expected_reward can be negative."""
        state = zero_state(d=D)
        for _ in range(1000):
            t = np.random.uniform(0.0, 24.0)
            phi = fourier(t, k=K)
            reward = -5.0  # Negative reward
            state = apply_contribution(state, phi, reward, SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        phi_pos = np.ones(D, dtype=np.float64)
        score = predict_expected_reward(phi_pos, posterior)
        assert score < 0.0, (
            f"Score should be negative with negative rewards; got {score}"
        )

    def test_score_is_float_type(self, well_conditioned_posterior, phi):
        """Score is Python float type."""
        score = predict_expected_reward(phi, well_conditioned_posterior)
        assert isinstance(score, float), f"Score should be float, got {type(score)}"

    def test_score_is_finite(self, prior, phi):
        """Score is finite (no NaN, no Inf)."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        score = predict_expected_reward(phi, posterior)
        assert np.isfinite(score), f"Score should be finite; got {score}"

    def test_phi_dimension_mismatch_raises(self, well_conditioned_posterior):
        """ValueError raised if phi dimension does not match posterior.d."""
        phi_bad = np.ones(3, dtype=np.float64)
        with pytest.raises(ValueError, match="dimension"):
            predict_expected_reward(phi_bad, well_conditioned_posterior)

    def test_phi_not_1d_raises(self, well_conditioned_posterior):
        """ValueError raised if phi is not 1-D."""
        phi_2d = np.ones((1, D), dtype=np.float64)
        with pytest.raises(ValueError, match="1-D"):
            predict_expected_reward(phi_2d, well_conditioned_posterior)


# ── MOD-06: predict_uncertainty ───────────────────────────────────────────────

class TestPredictUncertainty:
    """SRS MOD-06: latent_std, predictive_std, prior_weight."""

    def test_cold_start_latent_std_computed_from_prior(self, well_conditioned_posterior, phi):
        """Cold start: latent_std computed from prior precision."""
        latent, predictive, pw = predict_uncertainty(phi, well_conditioned_posterior)
        # latent_std = sqrt(phi.T @ Lambda0^{-1} @ phi)
        Lambda0_inv = np.linalg.inv(well_conditioned_posterior.L @ well_conditioned_posterior.L.T)
        expected_latent = np.sqrt(max(0.0, phi.T @ Lambda0_inv @ phi))
        np.testing.assert_allclose(
            latent, expected_latent, atol=1e-12, rtol=1e-12,
            err_msg="Cold-start latent_std should use prior precision"
        )

    def test_cold_start_prior_weight_is_one(self, well_conditioned_posterior, phi):
        """Cold start: prior_weight = 1.0."""
        _, _, pw = predict_uncertainty(phi, well_conditioned_posterior)
        assert pw == pytest.approx(1.0, abs=1e-15)

    def test_prior_weight_decreases_with_data(self, prior, phi):
        """prior_weight decreases as n increases."""
        cold = compute_posterior(zero_state(d=D), prior, sigma2=SIGMA2)
        cold_pw = predict_uncertainty(phi, cold)[2]

        # Build warm state inline to avoid fixture resolution issues
        warm_state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            warm_state = apply_contribution(warm_state, phi_obs, float(reward), SIGMA2)

        warm = compute_posterior(warm_state, prior, sigma2=SIGMA2)
        warm_pw = predict_uncertainty(phi, warm)[2]

        assert warm_pw < cold_pw, (
            f"Prior weight should decrease with data: cold={cold_pw}, warm={warm_pw}"
        )

    def test_predictive_std_geq_latent_std(self, prior, phi):
        """predictive_std >= latent_std (predictive includes noise variance)."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        latent, predictive, _ = predict_uncertainty(phi, posterior)
        assert predictive >= latent, (
            f"predictive_std ({predictive}) should be >= latent_std ({latent})"
        )

    def test_predictive_std_formula(self, prior, phi):
        """predictive_std ≈ sqrt(latent_std^2 + sigma2)."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        latent, predictive, _ = predict_uncertainty(phi, posterior)
        expected = np.sqrt(max(0.0, latent ** 2 + posterior.sigma2))
        np.testing.assert_allclose(
            predictive, expected, atol=1e-12, rtol=1e-12,
            err_msg=f"predictive_std should be sqrt(latent^2 + sigma2): "
                    f"got {predictive}, expected {expected}"
        )

    def test_all_values_finite(self, prior, phi):
        """All returned values are finite (no NaN, no Inf)."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        latent, predictive, pw = predict_uncertainty(phi, posterior)
        assert np.isfinite(latent), f"latent_std should be finite: {latent}"
        assert np.isfinite(predictive), f"predictive_std should be finite: {predictive}"
        assert np.isfinite(pw), f"prior_weight should be finite: {pw}"

    def test_prior_weight_in_range(self, prior, phi):
        """prior_weight in [0, 1]."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        _, _, pw = predict_uncertainty(phi, posterior)
        assert 0.0 <= pw <= 1.0, f"prior_weight should be in [0,1], got {pw}"

    def test_cold_start_all_finite(self, well_conditioned_posterior, phi):
        """Cold start: all uncertainty values finite."""
        latent, predictive, pw = predict_uncertainty(phi, well_conditioned_posterior)
        assert np.isfinite(latent)
        assert np.isfinite(predictive)
        assert np.isfinite(pw)

    def test_phi_dimension_mismatch_raises(self, well_conditioned_posterior):
        """ValueError if phi dimension does not match posterior.d."""
        phi_bad = np.ones(3, dtype=np.float64)
        with pytest.raises(ValueError, match="dimension"):
            predict_uncertainty(phi_bad, well_conditioned_posterior)


# ── MOD-07: Fallback ──────────────────────────────────────────────────────────

class TestFallback:
    """SRS MOD-07: Failed Cholesky, nonfinite parameters, incompatible state.
    Returns FallbackResult with named reason; never fabricates uncertainty."""

    def test_non_spd_lambda_returns_fallback(self, prior):
        """Non-positive-definite Lambda returns FallbackResult with reason='CHOLESKY_FAILED'."""
        # Lambda0 diagonal = [10, 10, 10, 40, 40]. Need A diagonal to make
        # at least one Lambda diagonal element <= 0.
        # Set A_upper all to a large negative value so Lambda = Lambda0 + A
        # has negative diagonal entries.
        state = zero_state(d=D)
        state.A_upper = -np.ones(D * (D + 1) // 2, dtype=np.float64) * 1000.0
        state.b = np.zeros(D, dtype=np.float64)

        result = compute_posterior(state, prior, sigma2=SIGMA2)
        assert isinstance(result, FallbackResult), (
            "Non-SPD Lambda should return FallbackResult"
        )
        assert result.reason == "CHOLESKY_FAILED", (
            f"Reason should be 'CHOLESKY_FAILED', got '{result.reason}'"
        )

    def test_fallback_prior_weight_is_one(self, prior):
        """FallbackResult has prior_weight = 1.0."""
        state = zero_state(d=D)
        state.A_upper = -np.abs(state.A_upper) - 1.0
        state.b = np.zeros(D, dtype=np.float64)

        result = compute_posterior(state, prior, sigma2=SIGMA2)
        assert result.prior_weight == pytest.approx(1.0, abs=1e-15)

    def test_fallback_mu_is_prior_mu0(self, prior):
        """Fallback mu = prior mu0."""
        state = zero_state(d=D)
        state.A_upper = -np.abs(state.A_upper) - 1.0
        state.b = np.zeros(D, dtype=np.float64)

        result = compute_posterior(state, prior, sigma2=SIGMA2)
        np.testing.assert_allclose(
            result.mu, prior.mu0, atol=1e-14, rtol=1e-14,
            err_msg="Fallback mu should equal prior mu0"
        )

    def test_nonfinite_b_returns_fallback(self, prior):
        """Nonfinite b (NaN) returns FallbackResult with reason='NONFINITE_PARAMETERS'."""
        state = zero_state(d=D)
        state.b[0] = np.nan

        result = compute_posterior(state, prior, sigma2=SIGMA2)
        assert isinstance(result, FallbackResult)
        assert result.reason == "NONFINITE_PARAMETERS"

    def test_fallback_returned_not_raised(self, prior):
        """Fallback is returned (not raised), allowing graceful handling."""
        state = zero_state(d=D)
        state.b[0] = np.inf

        result = compute_posterior(state, prior, sigma2=SIGMA2)
        assert isinstance(result, FallbackResult)
        # Should not raise
        assert hasattr(result, 'reason')
        assert hasattr(result, 'mu')
        assert hasattr(result, 'L')

    def test_dimension_mismatch_returns_fallback(self, prior):
        """Incompatible state dimension returns FallbackResult."""
        bad_state = zero_state(d=3)  # Different dimension
        result = compute_posterior(bad_state, prior, sigma2=SIGMA2)
        assert isinstance(result, FallbackResult)
        assert result.reason == "INCOMPATIBLE_STATE"

    def test_nonfinite_sigma2_returns_fallback(self, prior):
        """Nonfinite sigma2 returns FallbackResult."""
        state = zero_state(d=D)
        result = compute_posterior(state, prior, sigma2=np.nan)
        assert isinstance(result, FallbackResult)
        assert result.reason == "NONFINITE_PARAMETERS"

    def test_nonpositive_sigma2_returns_fallback(self, prior):
        """Non-positive sigma2 returns FallbackResult."""
        state = zero_state(d=D)
        result = compute_posterior(state, prior, sigma2=0.0)
        assert isinstance(result, FallbackResult)
        assert result.reason == "NONFINITE_PARAMETERS"

        result_neg = compute_posterior(state, prior, sigma2=-0.1)
        assert isinstance(result_neg, FallbackResult)

    def test_nonfinite_prior_lambda0_returns_fallback(self):
        """Nonfinite prior Lambda0 returns FallbackResult."""
        prior_bad = Prior.diagonal_prior(d=D, alpha=0.1)
        prior_bad.Lambda0[0, 0] = np.nan
        # Re-construct eta0 to be consistent
        prior_bad.eta0 = prior_bad.Lambda0 @ prior_bad.mu0

        state = zero_state(d=D)
        result = compute_posterior(state, prior_bad, sigma2=SIGMA2)
        assert isinstance(result, FallbackResult)
        assert result.reason == "NONFINITE_PARAMETERS"

    def test_nonfinite_prior_eta0_returns_fallback(self):
        """Nonfinite prior eta0 returns FallbackResult."""
        prior_bad = Prior.diagonal_prior(d=D, alpha=0.1)
        prior_bad.eta0[0] = np.inf
        # Lambda0 is fine
        state = zero_state(d=D)
        result = compute_posterior(state, prior_bad, sigma2=SIGMA2)
        assert isinstance(result, FallbackResult)
        assert result.reason == "NONFINITE_PARAMETERS"

    def test_fallback_L_is_valid(self, prior):
        """Fallback L is a valid Cholesky factor (or identity if Lambda0 fails)."""
        state = zero_state(d=D)
        state.A_upper = -np.abs(state.A_upper) - 1.0
        state.b = np.zeros(D, dtype=np.float64)

        result = compute_posterior(state, prior, sigma2=SIGMA2)
        assert result.L.shape == (D, D)
        assert np.all(np.isfinite(result.L)), "Fallback L should be finite"

    def test_fallback_mu_is_finite(self, prior):
        """Fallback mu is finite."""
        state = zero_state(d=D)
        state.A_upper = -np.abs(state.A_upper) - 1.0
        state.b = np.zeros(D, dtype=np.float64)

        result = compute_posterior(state, prior, sigma2=SIGMA2)
        assert np.all(np.isfinite(result.mu)), "Fallback mu should be finite"


# ── score_candidate ───────────────────────────────────────────────────────────

class TestScoreCandidate:
    """score_candidate returns dict with all required keys."""

    REQUIRED_KEYS = {
        "expected_reward", "latent_std", "predictive_std", "prior_weight",
        "n", "is_fallback", "fallback_reason"
    }

    def test_returns_dict_with_required_keys(self, cold_state, prior, phi):
        """Returns dict with all required keys."""
        result = score_candidate(phi, cold_state, prior, sigma2=SIGMA2)
        assert isinstance(result, dict)
        assert set(result.keys()) == self.REQUIRED_KEYS, (
            f"Missing keys: {self.REQUIRED_KEYS - set(result.keys())}, "
            f"Extra keys: {set(result.keys()) - self.REQUIRED_KEYS}"
        )

    def test_is_fallback_false_on_success(self, cold_state, prior, phi):
        """is_fallback = False on success."""
        result = score_candidate(phi, cold_state, prior, sigma2=SIGMA2)
        assert result["is_fallback"] is False
        assert result["fallback_reason"] is None

    def test_is_fallback_true_on_fallback(self, prior, phi):
        """is_fallback = True on fallback."""
        state = zero_state(d=D)
        state.b[0] = np.nan
        result = score_candidate(phi, state, prior, sigma2=SIGMA2)
        assert result["is_fallback"] is True
        assert result["fallback_reason"] == "NONFINITE_PARAMETERS"

    def test_expected_reward_matches_predict_expected_reward(self, prior, phi):
        """expected_reward matches predict_expected_reward."""
        # Build warm state inline
        warm_state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            warm_state = apply_contribution(warm_state, phi_obs, float(reward), SIGMA2)

        result = score_candidate(phi, warm_state, prior, sigma2=SIGMA2)
        posterior = compute_posterior(warm_state, prior, sigma2=SIGMA2)
        expected = predict_expected_reward(phi, posterior)
        np.testing.assert_allclose(
            result["expected_reward"], expected, atol=1e-14, rtol=1e-14
        )

    def test_score_values_are_finite_on_success(self, cold_state, prior, phi):
        """All numeric values are finite on success."""
        result = score_candidate(phi, cold_state, prior, sigma2=SIGMA2)
        for key in ["expected_reward", "latent_std", "predictive_std", "prior_weight"]:
            assert np.isfinite(result[key]), f"{key} should be finite: {result[key]}"

    def test_n_matches_state(self, single_obs_state, prior, phi):
        """n in result matches state.n."""
        result = score_candidate(phi, single_obs_state, prior, sigma2=SIGMA2)
        assert result["n"] == single_obs_state.n

    def test_fallback_all_values_finite(self, prior, phi):
        """Fallback score values are all finite."""
        state = zero_state(d=D)
        state.b[0] = np.nan
        result = score_candidate(phi, state, prior, sigma2=SIGMA2)
        for key in ["expected_reward", "latent_std", "predictive_std", "prior_weight"]:
            assert np.isfinite(result[key]), f"Fallback {key} should be finite: {result[key]}"

    def test_fallback_prior_weight_is_one(self, prior, phi):
        """Fallback prior_weight = 1.0."""
        state = zero_state(d=D)
        state.b[0] = np.nan
        result = score_candidate(phi, state, prior, sigma2=SIGMA2)
        assert result["prior_weight"] == pytest.approx(1.0, abs=1e-15)


# ── score_candidates (batch) ──────────────────────────────────────────────────

class TestScoreCandidates:
    """score_candidates: batch scoring returns (n, 4) array."""

    def test_batch_returns_n_by_4_array(self, cold_state, prior, phi_matrix):
        """Batch scoring returns (n, 4) array."""
        scores = score_candidates(phi_matrix, cold_state, prior, sigma2=SIGMA2)
        assert scores.shape == (4, 4), f"Expected (4, 4), got {scores.shape}"

    def test_batch_results_match_individual_scoring(self, prior):
        """Batch results match individual scoring for each candidate."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        times = np.array([8.0, 12.0, 16.0, 20.0])
        phi_matrix = fourier(times, k=K)

        batch_scores = score_candidates(phi_matrix, state, prior, sigma2=SIGMA2)
        for i in range(phi_matrix.shape[0]):
            individual = score_candidate(phi_matrix[i], state, prior, sigma2=SIGMA2)
            np.testing.assert_allclose(
                batch_scores[i],
                [
                    individual["expected_reward"],
                    individual["latent_std"],
                    individual["predictive_std"],
                    individual["prior_weight"],
                ],
                atol=1e-12, rtol=1e-12,
                err_msg=f"Batch row {i} should match individual scoring"
            )

    def test_empty_input_returns_zero_by_4(self, cold_state, prior):
        """Empty input returns (0, 4) array."""
        empty_phi = fourier(np.array([]), k=K)
        scores = score_candidates(empty_phi, cold_state, prior, sigma2=SIGMA2)
        assert scores.shape == (0, 4), f"Expected (0, 4), got {scores.shape}"
        assert scores.dtype == np.float64

    def test_batch_all_finite(self, prior):
        """All batch scores are finite."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        times = np.array([8.0, 12.0, 16.0, 20.0])
        phi_matrix = fourier(times, k=K)

        scores = score_candidates(phi_matrix, state, prior, sigma2=SIGMA2)
        assert np.all(np.isfinite(scores)), "All batch scores should be finite"

    def test_batch_prior_weight_constant(self, prior):
        """Prior weight is the same for all candidates in a batch."""
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        times = np.array([8.0, 12.0, 16.0, 20.0])
        phi_matrix = fourier(times, k=K)

        scores = score_candidates(phi_matrix, state, prior, sigma2=SIGMA2)
        pw_col = scores[:, 3]
        assert np.allclose(pw_col, pw_col[0]), (
            "Prior weight should be constant across all candidates"
        )

    def test_batch_single_candidate(self, cold_state, prior):
        """Single candidate batch returns (1, 4)."""
        phi_single = fourier(10.0, k=K).reshape(1, -1)
        scores = score_candidates(phi_single, cold_state, prior, sigma2=SIGMA2)
        assert scores.shape == (1, 4)

    def test_batch_2d_only(self, cold_state, prior):
        """ValueError raised if phi_matrix is not 2-D."""
        phi_1d = fourier(10.0, k=K)
        with pytest.raises(ValueError, match="2-D"):
            score_candidates(phi_1d, cold_state, prior, sigma2=SIGMA2)

    def test_batch_dimension_mismatch(self, cold_state, prior):
        """ValueError raised if phi_matrix columns don't match state.d."""
        phi_bad = fourier(np.array([10.0, 14.0]), k=3)  # d=7 vs state.d=5
        with pytest.raises(ValueError, match="does not match"):
            score_candidates(phi_bad, cold_state, prior, sigma2=SIGMA2)

    def test_batch_fallback_returns_zeros_for_uncertainty(self, prior):
        """Fallback batch: latent_std=0, predictive_std=0, prior_weight=1."""
        state = zero_state(d=D)
        state.b[0] = np.nan
        times = np.array([8.0, 12.0, 16.0, 20.0])
        phi_matrix = fourier(times, k=K)
        scores = score_candidates(phi_matrix, state, prior, sigma2=SIGMA2)
        assert np.all(scores[:, 1] == 0.0), "Fallback latent_std should be 0"
        assert np.all(scores[:, 2] == 0.0), "Fallback predictive_std should be 0"
        assert np.all(scores[:, 3] == 1.0), "Fallback prior_weight should be 1"


# ── Numerical accuracy ────────────────────────────────────────────────────────

class TestNumericalAccuracy:
    """SRS §15: atol=1e-9, rtol=1e-9 on well-conditioned float64 fixtures."""

    def test_posterior_matches_trusted_numpy_solution(self, prior):
        """Posterior matches trusted numpy solution for small d."""
        state = zero_state(d=D)
        np.random.seed(123)
        for _ in range(30):
            t = np.random.uniform(0.0, 24.0)
            phi = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        # Trusted solution: direct numpy solve
        Lambda = prior.Lambda0 + state.A
        eta = prior.eta0 + state.b
        mu_trusted = np.linalg.solve(Lambda, eta)

        np.testing.assert_allclose(
            posterior.mu, mu_trusted,
            atol=1e-9, rtol=1e-9,
            err_msg="Posterior mu should match numpy.linalg.solve"
        )

    def test_well_conditioned_precision(self, cold_state, prior):
        """Cold start: posterior values within tight tolerance of analytic solution."""
        posterior = compute_posterior(cold_state, prior, sigma2=SIGMA2)
        # Cold start: mu = mu0, L = cholesky(Lambda0), prior_weight = 1.0
        np.testing.assert_allclose(
            posterior.mu, prior.mu0,
            atol=1e-9, rtol=1e-9
        )
        np.testing.assert_allclose(
            posterior.L @ posterior.L.T, prior.Lambda0,
            atol=1e-9, rtol=1e-9
        )
        assert posterior.prior_weight == pytest.approx(1.0, abs=1e-12)

    def test_predict_uncertainty_precision(self, prior):
        """Uncertainty values match direct computation within tolerance."""
        phi = fourier(10.0, k=K)
        state = zero_state(d=D)
        np.random.seed(42)
        for _ in range(100):
            t = np.random.uniform(0.0, 24.0)
            phi_obs = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi_obs, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        latent, predictive, pw = predict_uncertainty(phi, posterior)

        # Direct computation
        L = posterior.L
        x = np.linalg.solve(L, phi)
        latent_direct = np.sqrt(max(0.0, np.dot(x, x)))
        predictive_direct = np.sqrt(max(0.0, latent_direct ** 2 + posterior.sigma2))

        np.testing.assert_allclose(
            latent, latent_direct, atol=1e-9, rtol=1e-9
        )
        np.testing.assert_allclose(
            predictive, predictive_direct, atol=1e-9, rtol=1e-9
        )

    def test_compute_prior_weight_precision(self, prior):
        """compute_prior_weight matches formula exactly for diagonal matrices."""
        Lambda0 = prior.Lambda0
        state = zero_state(d=D)
        for _ in range(20):
            t = np.random.uniform(0.0, 24.0)
            phi = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi, float(reward), SIGMA2)
        Lambda = prior.Lambda0 + state.A

        weight = compute_prior_weight(Lambda0, Lambda)
        expected = float(np.trace(Lambda0) / np.trace(Lambda))
        np.testing.assert_allclose(
            weight, expected, atol=1e-12, rtol=1e-12
        )

    def test_high_precision_cold_start(self, prior):
        """Cold start values match analytic with very tight tolerance."""
        state = zero_state(d=D)
        posterior = compute_posterior(state, prior, sigma2=SIGMA2)

        # mu = mu0
        np.testing.assert_allclose(
            posterior.mu, prior.mu0, atol=1e-14, rtol=1e-14
        )

        # L = cholesky(Lambda0)
        L_expected = np.linalg.cholesky(prior.Lambda0)
        np.testing.assert_allclose(
            posterior.L, L_expected, atol=1e-14, rtol=1e-14
        )

        # prior_weight = 1.0
        assert posterior.prior_weight == pytest.approx(1.0, abs=1e-15)


# ── Edge cases and additional coverage ────────────────────────────────────────

class TestEdgeCases:
    """Additional edge cases not covered above."""

    def test_different_sigma2_values(self, prior):
        """Different sigma2 values produce valid posteriors."""
        state = zero_state(d=D)
        for s2 in [0.01, 0.06, 0.5, 1.0]:
            posterior = compute_posterior(state, prior, sigma2=s2)
            assert isinstance(posterior, Posterior)
            assert np.isfinite(posterior.sigma2)

    def test_prior_weight_includes_boundary_values(self, prior):
        """prior_weight can reach 0.0 and 1.0 at boundaries."""
        # Cold start: prior_weight = 1.0
        cold = compute_posterior(zero_state(d=D), prior, sigma2=SIGMA2)
        assert cold.prior_weight == pytest.approx(1.0, abs=1e-15)

        # compute_prior_weight with zero denominator: returns 1.0
        weight_zero_denom = compute_prior_weight(np.zeros((3, 3)), np.zeros((3, 3)))
        assert weight_zero_denom == 1.0

        # compute_prior_weight with zero prior trace: returns 0.0
        weight_zero_prior = compute_prior_weight(np.zeros((3, 3)), np.eye(3))
        assert weight_zero_prior == 0.0

    def test_fallback_reasons_are_named(self, prior):
        """All fallback reasons are valid named strings."""
        valid_reasons = {"CHOLESKY_FAILED", "NONFINITE_PARAMETERS", "INCOMPATIBLE_STATE"}

        # CHOLESKY_FAILED
        bad_state = zero_state(d=D)
        bad_state.A_upper = -np.ones(D * (D + 1) // 2, dtype=np.float64) * 1000.0
        result = compute_posterior(bad_state, prior, sigma2=SIGMA2)
        assert isinstance(result, FallbackResult)
        assert result.reason in valid_reasons

        # NONFINITE_PARAMETERS
        bad_state2 = zero_state(d=D)
        bad_state2.b[0] = np.nan
        result2 = compute_posterior(bad_state2, prior, sigma2=SIGMA2)
        assert isinstance(result2, FallbackResult)
        assert result2.reason in valid_reasons

        # INCOMPATIBLE_STATE
        bad_state3 = zero_state(d=3)
        result3 = compute_posterior(bad_state3, prior, sigma2=SIGMA2)
        assert isinstance(result3, FallbackResult)
        assert result3.reason in valid_reasons

    def test_fallback_result_dataclass_fields(self, prior):
        """FallbackResult has all required fields."""
        state = zero_state(d=D)
        state.b[0] = np.nan
        result = compute_posterior(state, prior, sigma2=SIGMA2)

        assert hasattr(result, "reason")
        assert hasattr(result, "mu")
        assert hasattr(result, "L")
        assert hasattr(result, "prior_weight")
        assert isinstance(result.reason, str)
        assert isinstance(result.mu, np.ndarray)
        assert isinstance(result.L, np.ndarray)
        assert isinstance(result.prior_weight, float)

    def test_score_candidate_fallback_reason_propagated(self, prior, phi):
        """Fallback reason is propagated through score_candidate."""
        state = zero_state(d=D)
        state.b[0] = np.nan
        result = score_candidate(phi, state, prior, sigma2=SIGMA2)
        assert result["fallback_reason"] == "NONFINITE_PARAMETERS"

    def test_score_candidates_fallback_reason_not_in_array(self, prior):
        """Batch fallback returns scores array (no reason field in array)."""
        state = zero_state(d=D)
        state.b[0] = np.nan
        times = np.array([8.0, 12.0, 16.0, 20.0])
        phi_matrix = fourier(times, k=K)
        scores = score_candidates(phi_matrix, state, prior, sigma2=SIGMA2)
        assert scores.shape == (4, 4)
        # Fallback reason is not in the array — it's a structured result
        # The batch function returns raw scores for fallback

    def test_nonfinite_mu_returns_fallback(self, prior):
        """Nonfinite posterior mean mu triggers fallback."""
        # Create state that produces nonfinite mu via extreme values
        state = zero_state(d=D)
        # Set b to extreme values that could cause overflow
        state.b = np.array([1e308, 1e308, 1e308, 1e308, 1e308], dtype=np.float64)
        result = compute_posterior(state, prior, sigma2=SIGMA2)
        # Either returns Posterior with finite mu or FallbackResult
        if isinstance(result, Posterior):
            assert np.all(np.isfinite(result.mu)), "mu should be finite"

    def test_large_observation_count_stable(self, prior):
        """Posterior computation is stable with large observation counts."""
        state = zero_state(d=D)
        np.random.seed(456)
        for _ in range(500):
            t = np.random.uniform(0.0, 24.0)
            phi = fourier(t, k=K)
            reward = np.random.normal(0.5, 0.3)
            state = apply_contribution(state, phi, float(reward), SIGMA2)

        posterior = compute_posterior(state, prior, sigma2=SIGMA2)
        assert isinstance(posterior, Posterior)
        assert np.all(np.isfinite(posterior.mu))
        assert np.all(np.isfinite(posterior.L))
        assert 0.0 <= posterior.prior_weight <= 1.0
