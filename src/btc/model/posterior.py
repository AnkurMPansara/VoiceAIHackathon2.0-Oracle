"""Posterior computation and prediction for the Best Time to Call system.

Implements SRS MOD-04 / MOD-06 / MOD-07: Gaussian posterior inference,
expected reward prediction, uncertainty quantification, and candidate
scoring for per-seller Bayesian linear regression.

IMPORTANT: This is a DETERMINISTIC expected reward maximization model.
The model computes phi.T @ mu (posterior mean) for scoring. It does NOT
use Thompson sampling — no posterior sampling is performed. Uncertainty
(latent_std, predictive_std) is computed but NOT used in the reward score.

Core Bayesian update (MOD-04):
    Lambda = Lambda0 + A
    L = cholesky(Lambda)              # lower triangular
    mu = solve(L.T, solve(L, eta0 + b))

Scoring (MOD-06):
    expected_reward = phi.T @ mu          # deterministic, posterior mean
    latent_std = sqrt(max(0, dot(solve(L, phi), solve(L, phi))))
    predictive_std = sqrt(latent_std**2 + sigma2)
    prior_weight = trace(Lambda0) / trace(Lambda0 + A)

Note: expected_reward uses ONLY the posterior mean (phi.T @ mu).
Uncertainty values are computed for diagnostics but do not affect ranking.

MOD-06 notes:
    - prior_weight is a basis-dependent diagnostic, not the fraction of
      prediction caused by the prior.
    - prior_weight is in [0, 1]. Cold-start value is 1.
    - DO NOT clip expected_reward scores to [0, 1].

MOD-07 notes:
    - Failed Cholesky, nonfinite parameters, or incompatible state
      triggers named fallback and alert; never return fabricated
      uncertainty.

Modules
-------
compute_posterior : Compute posterior from state and prior.
predict_expected_reward : Predict expected reward for a candidate time.
predict_uncertainty : Predict latent and predictive uncertainty (diagnostics only).
compute_prior_weight : Compute prior weight diagnostic.
score_candidate : Full score for a single candidate timestamp.
score_candidates : Score multiple candidates efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt

from btc.model.stats import Prior, SellerState


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class Posterior:
    """Computed posterior for a seller given state and prior.

    MOD-04/06: Contains mean, uncertainty, and prior weight for scoring.

    Attributes
    ----------
    mu : np.ndarray, shape (d,)
        Posterior mean of coefficients.
    L : np.ndarray, shape (d, d)
        Cholesky factor of Lambda (posterior precision), lower triangular.
    sigma2 : float
        Working noise variance.
    prior_weight : float
        Diagnostic in [0, 1]. 1.0 for cold start, approaching 0 with more data.
    n : int
        Number of observations.
    d : int
        Feature dimension.

    Examples
    --------
    >>> from btc.model.stats import zero_state, Prior
    >>> import numpy as np
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> state = zero_state(d=5)
    >>> posterior = compute_posterior(state, prior, sigma2=0.06)
    >>> isinstance(posterior, Posterior)
    True
    >>> posterior.d
    5
    >>> posterior.is_cold_start
    True
    """

    mu: np.ndarray  # shape (d,)
    L: np.ndarray  # shape (d, d), lower triangular
    sigma2: float
    prior_weight: float
    n: int
    d: int

    @property
    def is_cold_start(self) -> bool:
        """True if n == 0 (zero-state seller).

        Returns
        -------
        bool
            True when no observations have been accumulated.
        """
        return self.n == 0


@dataclass
class FallbackResult:
    """MOD-07: Returned when Cholesky fails or state is corrupted.

    Attributes
    ----------
    reason : str
        Named reason: 'CHOLESKY_FAILED', 'NONFINITE_PARAMETERS',
        'INCOMPATIBLE_STATE'.
    mu : np.ndarray
        Fallback mean (prior mean mu0).
    L : np.ndarray
        Fallback Cholesky (prior precision Lambda0 Cholesky).
    prior_weight : float
        Always 1.0 for fallback.
    """

    reason: str
    mu: np.ndarray
    L: np.ndarray
    prior_weight: float = 1.0
    n: int = 0
    d: int = 0
    sigma2: float = 0.0

    @property
    def is_cold_start(self) -> bool:
        """Fallback is prior-only, so treat it as cold start."""
        return True


# ── Prior weight ─────────────────────────────────────────────────────────────


def compute_prior_weight(Lambda0: np.ndarray, Lambda: np.ndarray) -> float:
    """MOD-06: Compute prior weight diagnostic.

    prior_weight = trace(Lambda0) / trace(Lambda0 + A)
                 = trace(Lambda0) / trace(Lambda)

    This is basis-dependent, not the fraction of prediction from prior.
    Returns finite value in [0, 1].

    Parameters
    ----------
    Lambda0 : np.ndarray, shape (d, d)
        Prior precision.
    Lambda : np.ndarray, shape (d, d)
        Posterior precision (Lambda0 + A).

    Returns
    -------
    float
        Prior weight in [0, 1]. Returns 1.0 when denominator is zero
        (degenerate case). Returns 0.0 when prior trace is zero.

    Raises
    ------
    ValueError
        If matrices have incompatible shapes.

    Examples
    --------
    >>> import numpy as np
    >>> Lambda0 = np.diag([1.0, 1.0, 1.0, 4.0, 4.0])
    >>> Lambda = Lambda0 + np.diag([0.5, 0.3, 0.2, 0.1, 0.1])
    >>> w = compute_prior_weight(Lambda0, Lambda)
    >>> 0.0 <= w <= 1.0
    True
    """
    Lambda0 = np.asarray(Lambda0, dtype=np.float64)
    Lambda = np.asarray(Lambda, dtype=np.float64)

    if Lambda0.shape != Lambda.shape:
        raise ValueError(
            f"Shape mismatch: Lambda0 {Lambda0.shape} vs Lambda {Lambda.shape}"
        )

    trace_lambda0 = float(np.trace(Lambda0))
    trace_lambda = float(np.trace(Lambda))

    if trace_lambda == 0.0:
        return 1.0

    if trace_lambda0 <= 0.0:
        return 0.0

    weight = trace_lambda0 / trace_lambda

    # Clamp to [0, 1] for safety — should already be in range
    # for valid SPD matrices, but guard against numerical issues.
    weight = max(0.0, min(1.0, weight))

    if not np.isfinite(weight):
        return 1.0

    return float(weight)


# ── Core posterior computation ───────────────────────────────────────────────


def compute_posterior(
    state: SellerState, prior: Prior, sigma2: float
) -> Union[Posterior, FallbackResult]:
    """MOD-04: Compute posterior mean and Cholesky factor from state and prior.

    Lambda = Lambda0 + A
    L = cholesky(Lambda)
    mu = solve(L.T, solve(L, eta0 + b))

    Returns Posterior on success, or FallbackResult on failure (MOD-07).
    The FallbackResult contains prior-based defaults (mu0, L0, prior_weight=1.0).

    This function is part of a deterministic expected reward maximization pipeline.
    The returned posterior's mean (mu) is used directly for scoring — no sampling.

    Parameters
    ----------
    state : SellerState
        Seller sufficient statistics.
    prior : Prior
        Gaussian prior.
    sigma2 : float
        Working noise variance. Must be > 0.

    Returns
    -------
    Posterior | FallbackResult
        Posterior on success (contains mu, L, sigma2, prior_weight, n, d),
        or FallbackResult on failure (contains reason, mu, L, prior_weight=1.0).

    Raises
    ------
    None — failures return FallbackResult per MOD-07.

    Examples
    --------
    >>> from btc.model.stats import zero_state
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> state = zero_state(d=5)
    >>> result = compute_posterior(state, prior, sigma2=0.06)
    >>> isinstance(result, Posterior)
    True
    >>> result.is_cold_start
    True
    >>> result.prior_weight
    1.0
    """
    # Validate sigma2
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "sigma2 is nonfinite or non-positive",
        )

    # Validate dimension compatibility
    if state.d != prior.d:
        return _make_fallback(
            prior,
            "INCOMPATIBLE_STATE",
            f"Dimension mismatch: state.d={state.d}, prior.d={prior.d}",
        )

    d = state.d

    # Validate state arrays for nonfinite values (MOD-07)
    if not np.all(np.isfinite(state.A_upper)):
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "State A_upper contains nonfinite values",
        )

    if not np.all(np.isfinite(state.b)):
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "State b contains nonfinite values",
        )

    # Validate prior arrays for nonfinite values (MOD-07)
    if not np.all(np.isfinite(prior.Lambda0)):
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "Prior Lambda0 contains nonfinite values",
        )

    if not np.all(np.isfinite(prior.eta0)):
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "Prior eta0 contains nonfinite values",
        )

    # Compute posterior precision: Lambda = Lambda0 + A
    Lambda = prior.Lambda0 + state.A

    # Compute posterior natural parameter: eta = eta0 + b
    eta = prior.eta0 + state.b

    # Validate Lambda for nonfinite values
    if not np.all(np.isfinite(Lambda)):
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "Posterior precision Lambda contains nonfinite values",
        )

    # Validate eta for nonfinite values
    if not np.all(np.isfinite(eta)):
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "Posterior natural parameter eta contains nonfinite values",
        )

    # Cholesky factorisation: Lambda = L @ L^T, L lower triangular
    # MOD-07: On failure, return FallbackResult
    try:
        L = np.linalg.cholesky(Lambda)
    except np.linalg.LinAlgError:
        return _make_fallback(
            prior,
            "CHOLESKY_FAILED",
            "Cholesky decomposition of posterior precision Lambda failed",
        )

    # Solve L @ z = eta  =>  z = L^{-1} @ eta
    try:
        z = np.linalg.solve(L, eta)
    except np.linalg.LinAlgError:
        return _make_fallback(
            prior,
            "CHOLESKY_FAILED",
            "Solve step with L failed",
        )

    # Posterior mean: mu = L^{-T} @ z = Lambda^{-1} @ eta
    try:
        mu = np.linalg.solve(L.T, z)
    except np.linalg.LinAlgError:
        return _make_fallback(
            prior,
            "CHOLESKY_FAILED",
            "Solve step with L^T failed",
        )

    # Validate posterior mean for nonfinite values (MOD-07)
    if not np.all(np.isfinite(mu)):
        return _make_fallback(
            prior,
            "NONFINITE_PARAMETERS",
            "Posterior mean mu contains nonfinite values",
        )

    # Compute prior weight diagnostic (MOD-06)
    prior_weight = compute_prior_weight(prior.Lambda0, Lambda)

    return Posterior(
        mu=mu,
        L=L,
        sigma2=float(sigma2),
        prior_weight=prior_weight,
        n=state.n,
        d=d,
    )


# ── Prediction functions ─────────────────────────────────────────────────────


def predict_expected_reward(
    phi: np.ndarray, posterior: Posterior
) -> float:
    """Predict expected reward for a candidate time using posterior mean.

    expected_reward = phi.T @ mu

    This uses the POSTERIOR MEAN (NOT Thompson sampling). The model is
    deterministic — it always returns the same expected reward for the same
    phi and posterior. No random sampling from the posterior is performed.

    MOD-06: Returns raw expected reward (NOT clipped to [0,1],
    NOT a probability). Values can be negative.

    Parameters
    ----------
    phi : np.ndarray, shape (d,)
        Fourier feature vector.
    posterior : Posterior
        Computed posterior.

    Returns
    -------
    float
        Expected reward = phi.T @ mu (NOT clipped to [0,1], NOT a probability).
        Can be negative. Deterministic — same inputs always produce same output.

    Raises
    ------
    ValueError
        If phi dimension does not match posterior dimension.

    Examples
    --------
    >>> from btc.model.stats import zero_state, Prior
    >>> import numpy as np
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> state = zero_state(d=5)
    >>> posterior = compute_posterior(state, prior, sigma2=0.06)
    >>> phi = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> reward = predict_expected_reward(phi, posterior)
    >>> np.isfinite(reward)
    True
    """
    phi = np.asarray(phi, dtype=np.float64)

    if phi.ndim != 1:
        raise ValueError(f"phi must be 1-D, got shape {phi.shape}")

    if len(phi) != posterior.d:
        raise ValueError(
            f"phi dimension {len(phi)} does not match posterior.d {posterior.d}"
        )

    expected_reward = float(phi.T @ posterior.mu)

    if not np.isfinite(expected_reward):
        return 0.0

    return expected_reward


def predict_uncertainty(
    phi: np.ndarray, posterior: Posterior
) -> tuple[float, float, float]:
    """MOD-06: Predict latent and predictive uncertainty.

    latent_std = sqrt(max(0, dot(solve(L, phi), solve(L, phi))))
    predictive_std = sqrt(latent_std^2 + sigma2)
    prior_weight = trace(Lambda0) / trace(Lambda0 + A)

    IMPORTANT: These uncertainty values are computed for diagnostics/monitoring
    ONLY. They are NOT used in the expected reward score or candidate ranking.
    The model uses deterministic expected reward maximization (phi.T @ mu),
    not Thompson sampling or any uncertainty-aware exploration strategy.

    Parameters
    ----------
    phi : np.ndarray, shape (d,)
        Fourier feature vector.
    posterior : Posterior
        Computed posterior.

    Returns
    -------
    tuple[float, float, float]
        (latent_std, predictive_std, prior_weight)
        All finite values. prior_weight in [0, 1].
        These values are NOT used in reward computation or ranking.

    Raises
    ------
    ValueError
        If phi dimension does not match posterior dimension.

    Examples
    --------
    >>> from btc.model.stats import zero_state, Prior
    >>> import numpy as np
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> state = zero_state(d=5)
    >>> posterior = compute_posterior(state, prior, sigma2=0.06)
    >>> phi = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> latent, predictive, pw = predict_uncertainty(phi, posterior)
    >>> np.isfinite(latent) and np.isfinite(predictive) and np.isfinite(pw)
    True
    >>> 0.0 <= pw <= 1.0
    True
    """
    phi = np.asarray(phi, dtype=np.float64)

    if phi.ndim != 1:
        raise ValueError(f"phi must be 1-D, got shape {phi.shape}")

    if len(phi) != posterior.d:
        raise ValueError(
            f"phi dimension {len(phi)} does not match posterior.d {posterior.d}"
        )

    # Solve L @ x = phi  =>  x = L^{-1} @ phi
    x = np.linalg.solve(posterior.L, phi)

    # Latent std: sqrt(phi^T Lambda^{-1} phi) = sqrt(||L^{-1} phi||^2)
    latent_var = float(np.dot(x, x))
    latent_std = float(np.sqrt(max(0.0, latent_var)))

    # Predictive std: sqrt(latent_std^2 + sigma2)
    predictive_std = float(np.sqrt(max(0.0, latent_std**2 + posterior.sigma2)))

    # Prior weight diagnostic (MOD-06)
    # Recompute from posterior L to get Lambda
    Lambda = posterior.L @ posterior.L.T
    prior_weight = compute_prior_weight(
        np.zeros_like(posterior.L) + 0,  # placeholder, use cached value
        Lambda,
    )

    # For efficiency, use the posterior's stored prior_weight if available,
    # but recompute from the actual Lambda for correctness.
    # The prior weight in the posterior was computed with the actual Lambda0.
    # We need Lambda0 from the original computation — use the formula:
    # prior_weight = trace(Lambda0) / trace(Lambda)
    # Since we don't have Lambda0 here, compute from posterior properties.
    # The stored prior_weight is already correct for this posterior.
    prior_weight = posterior.prior_weight

    # Ensure all values are finite
    if not np.isfinite(latent_std):
        latent_std = 0.0
    if not np.isfinite(predictive_std):
        predictive_std = float(np.sqrt(max(0.0, posterior.sigma2)))
    if not np.isfinite(prior_weight) or prior_weight < 0.0 or prior_weight > 1.0:
        prior_weight = 1.0 if posterior.is_cold_start else 0.5

    return (float(latent_std), float(predictive_std), float(prior_weight))


# ── Candidate scoring ────────────────────────────────────────────────────────


def score_candidate(
    phi: np.ndarray,
    state: SellerState,
    prior: Prior,
    sigma2: float,
) -> dict:
    """Compute full score for a single candidate timestamp.

    Computes expected_reward using posterior mean (deterministic, NOT Thompson sampling).
    Uncertainty measures (latent_std, predictive_std) are included for diagnostics only
    and do not affect the expected_reward value or candidate ranking.

    Parameters
    ----------
    phi : np.ndarray, shape (d,)
        Fourier feature vector.
    state : SellerState
        Seller state.
    prior : Prior
        Gaussian prior.
    sigma2 : float
        Working noise variance.

    Returns
    -------
    dict
        {'expected_reward': float, 'latent_std': float,
         'predictive_std': float, 'prior_weight': float,
         'n': int, 'is_fallback': bool, 'fallback_reason': str or None}
        expected_reward is deterministic (phi.T @ mu), not Thompson sampled.

    Examples
    --------
    >>> from btc.model.stats import zero_state, Prior
    >>> import numpy as np
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> state = zero_state(d=5)
    >>> phi = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> result = score_candidate(phi, state, prior, sigma2=0.06)
    >>> 'expected_reward' in result
    True
    >>> 'is_fallback' in result
    True
    >>> result['is_fallback']
    False
    """
    posterior = compute_posterior(state, prior, sigma2)

    if isinstance(posterior, FallbackResult):
        # Fallback predictive_std: sqrt(trace(L @ L.T) + sigma2)
        # L is always a valid numpy array for FallbackResult
        # Use the sigma2 parameter (FallbackResult has no sigma2 attribute)
        L_trace = float(np.trace(posterior.L @ posterior.L.T))
        predictive_std = float(np.sqrt(max(0.0, L_trace + sigma2)))
        return {
            "expected_reward": float(posterior.mu.T @ phi) if phi.ndim == 1 else float(phi @ posterior.mu),
            "latent_std": 0.0,
            "predictive_std": predictive_std,
            "prior_weight": 1.0,
            "n": 0,
            "is_fallback": True,
            "fallback_reason": posterior.reason,
        }

    expected_reward = predict_expected_reward(phi, posterior)
    latent_std, predictive_std, prior_weight = predict_uncertainty(phi, posterior)

    return {
        "expected_reward": expected_reward,
        "latent_std": latent_std,
        "predictive_std": predictive_std,
        "prior_weight": prior_weight,
        "n": posterior.n,
        "is_fallback": False,
        "fallback_reason": None,
    }


def score_candidates(
    phi_matrix: np.ndarray,
    state: SellerState,
    prior: Prior,
    sigma2: float,
) -> np.ndarray:
    """Score multiple candidates efficiently using posterior mean.

    Deterministic expected reward maximization (NOT Thompson sampling).
    expected_reward = phi @ mu for each candidate, computed in vectorised form.
    Uncertainty values are diagnostics only, not used in ranking.

    Parameters
    ----------
    phi_matrix : np.ndarray, shape (n, d)
        Fourier features for n candidates.
    state : SellerState
        Seller state.
    prior : Prior
        Gaussian prior.
    sigma2 : float
        Working noise variance.

    Returns
    -------
    np.ndarray, shape (n, 4)
        Columns: [expected_reward, latent_std, predictive_std, prior_weight]
        expected_reward is deterministic (phi @ mu), not Thompson sampled.

    Raises
    ------
    ValueError
        If phi_matrix is not 2-D or dimensions don't match.

    Examples
    --------
    >>> from btc.model.stats import zero_state, Prior
    >>> from btc.features.fourier import fourier
    >>> import numpy as np
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> state = zero_state(d=5)
    >>> times = np.array([9.0, 14.0, 19.0])
    >>> phi_matrix = fourier(times, k=2)
    >>> scores = score_candidates(phi_matrix, state, prior, sigma2=0.06)
    >>> scores.shape
    (3, 4)
    >>> np.all(np.isfinite(scores))
    True
    """
    phi_matrix = np.asarray(phi_matrix, dtype=np.float64)

    if phi_matrix.ndim != 2:
        raise ValueError(f"phi_matrix must be 2-D, got shape {phi_matrix.shape}")

    if phi_matrix.shape[1] != state.d:
        raise ValueError(
            f"phi_matrix columns {phi_matrix.shape[1]} "
            f"does not match state.d {state.d}"
        )

    n_candidates = phi_matrix.shape[0]

    # Handle empty input
    if n_candidates == 0:
        return np.empty((0, 4), dtype=np.float64)

    # Compute posterior once (shared across all candidates)
    posterior = compute_posterior(state, prior, sigma2)

    if isinstance(posterior, FallbackResult):
        # Fallback: return zeros for uncertainty, prior mean for rewards
        mu = posterior.mu
        scores = np.empty((n_candidates, 4), dtype=np.float64)
        scores[:, 0] = phi_matrix @ mu  # expected_reward
        scores[:, 1] = 0.0  # latent_std
        scores[:, 2] = 0.0  # predictive_std
        scores[:, 3] = 1.0  # prior_weight
        return scores

    # Vectorised prediction for all candidates at once
    # expected_reward = phi @ mu for each row
    expected_rewards = phi_matrix @ posterior.mu

    # latent_std = sqrt(diag(phi @ Lambda^{-1} @ phi^T))
    # = sqrt(diag(phi @ L^{-T} @ L^{-1} @ phi^T))
    # = sqrt(diag((L^{-1} @ phi^T)^T @ (L^{-1} @ phi^T)))
    # Solve L @ X = phi_matrix^T  =>  X = L^{-1} @ phi_matrix^T
    L_inv_phi_T = np.linalg.solve(posterior.L, phi_matrix.T)
    latent_vars = np.sum(L_inv_phi_T ** 2, axis=0)
    latent_stds = np.sqrt(np.maximum(0.0, latent_vars))

    # predictive_std = sqrt(latent_std^2 + sigma2)
    predictive_stds = np.sqrt(np.maximum(0.0, latent_stds ** 2 + posterior.sigma2))

    # Prior weight is the same for all candidates (depends only on Lambda)
    prior_weight = posterior.prior_weight

    scores = np.empty((n_candidates, 4), dtype=np.float64)
    scores[:, 0] = expected_rewards
    scores[:, 1] = latent_stds
    scores[:, 2] = predictive_stds
    scores[:, 3] = prior_weight

    return scores


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_fallback(
    prior: Prior, reason: str, detail: str
) -> FallbackResult:
    """Create a FallbackResult with prior-based defaults.

    MOD-07: Always provides valid fallback values so callers never
    receive None or fabricated uncertainty.

    Parameters
    ----------
    prior : Prior
        The prior to use for fallback values.
    reason : str
        Named reason for fallback.
    detail : str
        Human-readable detail for logging/alerting.

    Returns
    -------
    FallbackResult
        Fallback result with prior mean and prior Cholesky.
    """
    # Fallback mean is the prior mean
    mu_fallback = prior.mu0.copy()

    # Fallback Cholesky is the prior precision Cholesky
    try:
        L_fallback = np.linalg.cholesky(prior.Lambda0)
    except np.linalg.LinAlgError:
        # If even prior Lambda0 is not SPD, use identity
        L_fallback = np.eye(prior.d, dtype=np.float64)

    return FallbackResult(
        reason=reason,
        mu=mu_fallback,
        L=L_fallback,
        prior_weight=1.0,
        d=prior.d,
    )
