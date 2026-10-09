"""Seller sufficient statistics and Bayesian update functions.

Implements MOD-04 / MOD-05: Bayesian linear regression sufficient statistics
for per-seller posterior inference in the Best Time to Call system.

The model maintains, for each seller, the sufficient statistics
    A = sum_t outer(phi_t, phi_t) / sigma2
    b = sum_t y_t * phi_t / sigma2
    n = number of observations

Given a Gaussian prior w ~ N(mu0, Sigma0) with precision Lambda0 = Sigma0^-1
and natural parameter eta0 = Lambda0 @ mu0, the posterior is

    Lambda = Lambda0 + A
    mu = Lambda^{-1} @ (eta0 + b)

which is computed via Cholesky factorisation for numerical stability.

All functions are PURE (no side effects, no global state).
gamma = 1 is enforced — no discounting, order-independent additions.

Modules
-------
zero_state : Create a cold-start zero state for a new seller.
compute_contribution : A, b contribution for a single observation.
apply_contribution : Update seller state with a new observation.
revoke_contribution : Remove a contribution (revision handling).
update_revision : Subtract old, add new (atomic revision).
extract_upper_triangle : Extract upper triangle from symmetric matrix.
reconstruct_symmetric : Reconstruct full symmetric matrix from upper triangle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import numpy.typing as npt


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class SellerState:
    """Sufficient statistics for a single seller's Bayesian posterior.

    MOD-04 / MOD-05: Stores A (precision accumulator), b (reward-feature
    accumulator), and n (observation count). Lambda and mu are derived
    from prior + state via :func:`posterior_params`.

    The A matrix is stored as its upper triangle only (including diagonal)
    for memory efficiency (STATE-07 codec compatibility). For dimension d,
    the upper triangle storage has size d*(d+1)//2.

    Attributes
    ----------
    A_upper : np.ndarray, shape (d*(d+1)//2,)
        Upper triangle of A, stored as a 1-D vector.
    b : np.ndarray, shape (d,)
        Reward-feature accumulator: sum of y*phi/sigma2 over observations.
    n : int
        Number of observations contributing to this state.
    state_version : int
        Monotonic version counter for cache invalidation (signed 64-bit).
        Incremented on apply, decremented on revoke.
    d : int
        Feature dimension (2*K + 1). Must match prior dimensions.

    Examples
    --------
    >>> state = zero_state(d=5)
    >>> state.n
    0
    >>> state.d
    5
    """

    A_upper: np.ndarray  # Upper triangle of A, shape (d*(d+1)//2,)
    b: np.ndarray  # shape (d,)
    n: int  # observation count
    state_version: int  # monotonic version counter
    d: int  # feature dimension

    def __post_init__(self) -> None:
        """Validate dimensions and array shapes."""
        expected_upper_size = self.d * (self.d + 1) // 2
        if self.A_upper.shape != (expected_upper_size,):
            raise ValueError(
                f"A_upper shape must be ({expected_upper_size},), "
                f"got {self.A_upper.shape}"
            )
        if self.b.shape != (self.d,):
            raise ValueError(
                f"b shape must be ({self.d},), got {self.b.shape}"
            )
        if self.A_upper.dtype != np.float64:
            self.A_upper = self.A_upper.astype(np.float64)
        if self.b.dtype != np.float64:
            self.b = self.b.astype(np.float64)

    @property
    def A(self) -> np.ndarray:
        """Full symmetric A matrix from upper triangle storage.

        Returns
        -------
        np.ndarray, shape (d, d)
            Symmetric matrix reconstructed from the upper triangle.

        Examples
        --------
        >>> state = zero_state(d=3)
        >>> state.A.shape
        (3, 3)
        >>> np.allclose(state.A, state.A.T)
        True
        """
        return reconstruct_symmetric(self.A_upper, self.d)

    def copy(self) -> "SellerState":
        """Deep copy of state.

        Returns
        -------
        SellerState
            Independent copy of this state with all arrays deep-copied.

        Examples
        --------
        >>> state = zero_state(d=3)
        >>> other = state.copy()
        >>> other is state
        False
        >>> np.shares_memory(other.A_upper, state.A_upper)
        False
        """
        return SellerState(
            A_upper=self.A_upper.copy(),
            b=self.b.copy(),
            n=self.n,
            state_version=self.state_version,
            d=self.d,
        )


@dataclass
class Prior:
    """Gaussian prior for seller coefficients.

    MOD-03 / MOD-04: w_s ~ N(mu0, Sigma0), precision Lambda0 = Sigma0^-1,
    eta0 = Lambda0 @ mu0.

    The prior encodes beliefs about seller coefficients before observing
    any data. The posterior is obtained by adding sufficient statistics
    from observations to the prior precision and natural parameter.

    Attributes
    ----------
    mu0 : np.ndarray, shape (d,)
        Prior mean vector.
    Sigma0 : np.ndarray, shape (d, d)
        Prior covariance matrix (symmetric positive definite).
    Lambda0 : np.ndarray, shape (d, d)
        Prior precision matrix (inverse covariance = Sigma0^-1).
    eta0 : np.ndarray, shape (d,)
        Prior natural parameter: Lambda0 @ mu0.
    d : int
        Feature dimension.

    Examples
    --------
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> prior.d
    5
    >>> prior.mu0.shape
    (5,)
    >>> prior.Lambda0.shape
    (5, 5)
    """

    mu0: np.ndarray  # shape (d,)
    Sigma0: np.ndarray  # shape (d, d), SPD
    Lambda0: np.ndarray  # shape (d, d), SPD
    eta0: np.ndarray  # shape (d,)
    d: int  # feature dimension

    def __post_init__(self) -> None:
        """Validate prior dimensions and SPD properties."""
        if self.mu0.shape != (self.d,):
            raise ValueError(f"mu0 shape must be ({self.d},), got {self.mu0.shape}")
        if self.Sigma0.shape != (self.d, self.d):
            raise ValueError(
                f"Sigma0 shape must be ({self.d}, {self.d}), got {self.Sigma0.shape}"
            )
        if self.Lambda0.shape != (self.d, self.d):
            raise ValueError(
                f"Lambda0 shape must be ({self.d}, {self.d}), "
                f"got {self.Lambda0.shape}"
            )
        if self.eta0.shape != (self.d,):
            raise ValueError(f"eta0 shape must be ({self.d},), got {self.eta0.shape}")

        # Verify eta0 = Lambda0 @ mu0
        if not np.allclose(self.eta0, self.Lambda0 @ self.mu0):
            raise ValueError("eta0 must equal Lambda0 @ mu0")

        # Verify Lambda0 = Sigma0^-1
        if not np.allclose(self.Lambda0, np.linalg.inv(self.Sigma0)):
            raise ValueError("Lambda0 must equal Sigma0^-1")

    @classmethod
    def diagonal_prior(cls, d: int, alpha: float = 0.1) -> "Prior":
        """Regularised diagonal prior.

        TRAIN-03: Constructs a diagonal covariance prior that expresses
        shrinkage, not empirical heterogeneity.

        Sigma0 = alpha * diag(1, 1, 1, 1/4, 1/4, ..., 1/K^2, 1/K^2)

        The first three diagonal elements are all 1 (bias and fundamental
        frequency share the same scale). Higher harmonics receive
        progressively smaller variance (larger precision), implementing
        smoothness-inducing shrinkage.

        Parameters
        ----------
        d : int
            Feature dimension. Must be odd (d = 2*K + 1).
        alpha : float
            Overall scale factor for the covariance diagonal. Default 0.1.

        Returns
        -------
        Prior
            Prior with diagonal Sigma0, inverse diagonal Lambda0, zero mean,
            and computed eta0.

        Raises
        ------
        ValueError
            If d is not odd, or alpha <= 0.

        Examples
        --------
        >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
        >>> prior.d
        5
        >>> prior.Sigma0.shape
        (5, 5)
        >>> np.allclose(prior.Sigma0, np.diag(np.diag(prior.Sigma0)))
        True
        """
        if d % 2 == 0:
            raise ValueError(f"d must be odd (d = 2*K + 1), got {d}")

        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        k = (d - 1) // 2

        # Build diagonal: [1, 1, 1, 1/4, 1/4, 1/9, 1/9, ...]
        diag_values = np.empty(d, dtype=np.float64)
        diag_values[0] = 1.0  # bias term
        for j in range(1, k + 1):
            scale = 1.0 / (j * j)
            diag_values[2 * j - 1] = scale  # sin component
            diag_values[2 * j] = scale  # cos component

        Sigma0 = alpha * np.diag(diag_values)
        Lambda0 = np.diag(1.0 / np.diag(Sigma0))
        mu0 = np.zeros(d, dtype=np.float64)
        eta0 = Lambda0 @ mu0

        return cls(
            mu0=mu0,
            Sigma0=Sigma0,
            Lambda0=Lambda0,
            eta0=eta0,
            d=d,
        )


# ── Utility functions ────────────────────────────────────────────────────────


def extract_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle (including diagonal) from symmetric matrix.

    Parameters
    ----------
    matrix : np.ndarray, shape (d, d)
        Symmetric matrix.

    Returns
    -------
    np.ndarray, shape (d*(d+1)//2,)
        Upper triangle stored as a 1-D vector in row-major order:
        [matrix[0,0], matrix[0,1], ..., matrix[0,d-1],
         matrix[1,1], matrix[1,2], ..., matrix[1,d-1],
         ...,
         matrix[d-1,d-1]]

    Examples
    --------
    >>> m = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=np.float64)
    >>> extract_upper_triangle(m)
    array([1., 2., 3., 4., 5., 6.])
    """
    d = matrix.shape[0]
    upper_size = d * (d + 1) // 2
    result = np.empty(upper_size, dtype=np.float64)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            result[idx] = matrix[i, j]
            idx += 1
    return result


def reconstruct_symmetric(upper: np.ndarray, d: int) -> np.ndarray:
    """Reconstruct full symmetric matrix from upper triangle.

    MOD-04 / STATE-07: Inverse of :func:`extract_upper_triangle`.

    Parameters
    ----------
    upper : np.ndarray, shape (d*(d+1)//2,)
        Upper triangle stored as a 1-D vector (row-major).
    d : int
        Matrix dimension.

    Returns
    -------
    np.ndarray, shape (d, d)
        Full symmetric matrix.

    Raises
    ------
    ValueError
        If upper array length does not match expected size for d.

    Examples
    --------
    >>> upper = np.array([1., 2., 3., 4., 5., 6.], dtype=np.float64)
    >>> m = reconstruct_symmetric(upper, 3)
    >>> m
    array([[1., 2., 3.],
           [2., 4., 5.],
           [3., 5., 6.]])
    """
    expected_size = d * (d + 1) // 2
    if len(upper) != expected_size:
        raise ValueError(
            f"upper length must be {expected_size}, got {len(upper)}"
        )

    matrix = np.zeros((d, d), dtype=np.float64)
    idx = 0
    for i in range(d):
        for j in range(i, d):
            val = upper[idx]
            matrix[i, j] = val
            matrix[j, i] = val
            idx += 1
    return matrix


# ── State management ─────────────────────────────────────────────────────────


def zero_state(d: int, state_version: int = 0) -> SellerState:
    """Create a cold-start zero state for a new seller.

    Parameters
    ----------
    d : int
        Feature dimension (2*K + 1).
    state_version : int
        Initial version number. Default 0.

    Returns
    -------
    SellerState
        Zero A (upper triangle), zero b, n=0, given version.

    Raises
    ------
    ValueError
        If d <= 0.

    Examples
    --------
    >>> state = zero_state(d=5)
    >>> state.n
    0
    >>> state.d
    5
    >>> state.A_upper.shape
    (15,)
    >>> state.b.shape
    (5,)
    """
    if d <= 0:
        raise ValueError(f"d must be > 0, got {d}")

    upper_size = d * (d + 1) // 2
    return SellerState(
        A_upper=np.zeros(upper_size, dtype=np.float64),
        b=np.zeros(d, dtype=np.float64),
        n=0,
        state_version=state_version,
        d=d,
    )


# ── Contribution functions ───────────────────────────────────────────────────


def compute_contribution(
    phi: np.ndarray, reward: float, sigma2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the A and b contribution for a single observation.

    MOD-04:
        contribution_A = outer(phi, phi) / sigma2
        contribution_b = y * phi / sigma2

    Parameters
    ----------
    phi : np.ndarray, shape (d,)
        Fourier feature vector for this observation. Must be 1-D float64.
    reward : float
        Reward value y for this observation.
    sigma2 : float
        Working noise variance. Must be > 0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (contribution_A_upper, contribution_b)
        contribution_A_upper is the upper triangle of outer(phi, phi)/sigma2.
        contribution_b is y * phi / sigma2.

    Raises
    ------
    ValueError
        If phi is not 1-D, or sigma2 <= 0.

    Examples
    --------
    >>> phi = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> A_upper, b = compute_contribution(phi, reward=0.5, sigma2=0.06)
    >>> A_upper.shape
    (15,)
    >>> b.shape
    (5,)
    >>> np.allclose(b, phi * 0.5 / 0.06)
    True
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")

    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 1:
        raise ValueError(f"phi must be 1-D, got shape {phi.shape}")

    inv_sigma2 = 1.0 / sigma2
    d = len(phi)

    # Compute outer(phi, phi) / sigma2
    outer_phi = np.outer(phi, phi) * inv_sigma2

    # Extract upper triangle
    A_upper = extract_upper_triangle(outer_phi)

    # Compute y * phi / sigma2
    b = phi * (reward * inv_sigma2)

    return (A_upper, b)


def apply_contribution(
    state: SellerState,
    phi: np.ndarray,
    reward: float,
    sigma2: float,
) -> SellerState:
    """Update seller state with a new observation.

    MOD-04:
        A += outer(phi, phi) / sigma2
        b += y * phi / sigma2
        n += 1

    This is the forward update for accumulating sufficient statistics.
    gamma = 1 is enforced — all observations have equal weight.

    Parameters
    ----------
    state : SellerState
        Current seller state.
    phi : np.ndarray, shape (d,)
        Fourier feature vector. Must match state.d.
    reward : float
        Reward value.
    sigma2 : float
        Working noise variance. Must be > 0.

    Returns
    -------
    SellerState
        Updated state with incremented state_version.

    Raises
    ------
    ValueError
        If feature dimension mismatch or sigma2 <= 0.

    Examples
    --------
    >>> state = zero_state(d=5)
    >>> phi = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> new_state = apply_contribution(state, phi, reward=0.5, sigma2=0.06)
    >>> new_state.n
    1
    >>> new_state.state_version
    1
    >>> new_state.n == state.n + 1
    True
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 1:
        raise ValueError(f"phi must be 1-D, got shape {phi.shape}")
    if len(phi) != state.d:
        raise ValueError(
            f"phi dimension {len(phi)} does not match state.d {state.d}"
        )
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")

    A_upper, b_contrib = compute_contribution(phi, reward, sigma2)

    new_A_upper = state.A_upper + A_upper
    new_b = state.b + b_contrib

    return SellerState(
        A_upper=new_A_upper,
        b=new_b,
        n=state.n + 1,
        state_version=state.state_version + 1,
        d=state.d,
    )


def revoke_contribution(
    state: SellerState,
    phi: np.ndarray,
    reward: float,
    sigma2: float,
) -> SellerState:
    """Remove a contribution (for revision handling).

    MOD-05:
        A -= outer(phi, phi) / sigma2
        b -= y * phi / sigma2
        n stays unchanged

    This reverses the effect of a previously applied contribution,
    e.g., when an outcome is corrected or removed.

    Parameters
    ----------
    state : SellerState
        Current seller state.
    phi : np.ndarray, shape (d,)
        Fourier feature vector of the contribution to remove.
    reward : float
        Reward value of the contribution to remove.
    sigma2 : float
        Working noise variance. Must be > 0.

    Returns
    -------
    SellerState
        Updated state with decremented state_version.

    Raises
    ------
    ValueError
        If feature dimension mismatch or sigma2 <= 0.

    Examples
    --------
    >>> state = zero_state(d=5)
    >>> phi = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> applied = apply_contribution(state, phi, reward=0.5, sigma2=0.06)
    >>> revoked = revoke_contribution(applied, phi, reward=0.5, sigma2=0.06)
    >>> revoked.n == applied.n
    True
    >>> np.allclose(revoked.A_upper, state.A_upper)
    True
    >>> np.allclose(revoked.b, state.b)
    True
    >>> revoked.state_version == applied.state_version - 1
    True
    """
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 1:
        raise ValueError(f"phi must be 1-D, got shape {phi.shape}")
    if len(phi) != state.d:
        raise ValueError(
            f"phi dimension {len(phi)} does not match state.d {state.d}"
        )
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")

    A_upper, b_contrib = compute_contribution(phi, reward, sigma2)

    new_A_upper = state.A_upper - A_upper
    new_b = state.b - b_contrib

    return SellerState(
        A_upper=new_A_upper,
        b=new_b,
        n=state.n,  # n stays unchanged per MOD-05
        state_version=state.state_version - 1,
        d=state.d,
    )


def update_revision(
    state: SellerState,
    old_phi: np.ndarray,
    old_reward: float,
    new_phi: np.ndarray,
    new_reward: float,
    sigma2: float,
) -> SellerState:
    """Apply a revision — subtract old contribution, add new.

    MOD-05: Atomic operation for outcome corrections.

    n stays unchanged. This is equivalent to:
        state = revoke_contribution(state, old_phi, old_reward, sigma2)
        state = apply_contribution(state, new_phi, new_reward, sigma2)
    but performed as a single atomic update to avoid intermediate
    state inconsistency.

    Parameters
    ----------
    state : SellerState
        Current seller state.
    old_phi : np.ndarray, shape (d,)
        Fourier features of the old outcome being replaced.
    old_reward : float
        Reward of the old outcome.
    new_phi : np.ndarray, shape (d,)
        Fourier features of the new outcome.
    new_reward : float
        Reward of the new outcome.
    sigma2 : float
        Working noise variance. Must be > 0.

    Returns
    -------
    SellerState
        Updated state with incremented state_version.

    Raises
    ------
    ValueError
        If feature dimension mismatch or sigma2 <= 0.

    Examples
    --------
    >>> state = zero_state(d=5)
    >>> phi_old = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> phi_new = np.array([1.0, 0.5, 0.5, 0.0, 0.5], dtype=np.float64)
    >>> revised = update_revision(
    ...     state, phi_old, old_reward=0.5,
    ...     new_phi=phi_new, new_reward=1.0, sigma2=0.06
    ... )
    >>> revised.n == state.n
    True
    >>> revised.state_version == 1
    True
    """
    old_phi = np.asarray(old_phi, dtype=np.float64)
    new_phi = np.asarray(new_phi, dtype=np.float64)

    if old_phi.ndim != 1:
        raise ValueError(f"old_phi must be 1-D, got shape {old_phi.shape}")
    if new_phi.ndim != 1:
        raise ValueError(f"new_phi must be 1-D, got shape {new_phi.shape}")
    if len(old_phi) != state.d:
        raise ValueError(
            f"old_phi dimension {len(old_phi)} does not match state.d {state.d}"
        )
    if len(new_phi) != state.d:
        raise ValueError(
            f"new_phi dimension {len(new_phi)} does not match state.d {state.d}"
        )
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")

    # Subtract old contribution
    old_A_upper, old_b = compute_contribution(old_phi, old_reward, sigma2)

    # Add new contribution
    new_A_upper, new_b = compute_contribution(new_phi, new_reward, sigma2)

    # Atomic update
    delta_A = new_A_upper - old_A_upper
    delta_b = new_b - old_b

    return SellerState(
        A_upper=state.A_upper + delta_A,
        b=state.b + delta_b,
        n=state.n,  # n stays unchanged per MOD-05
        state_version=state.state_version + 1,
        d=state.d,
    )


# ── Posterior computation ────────────────────────────────────────────────────


def posterior_params(
    state: SellerState, prior: Prior
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior mean and covariance from state and prior.

    MOD-04: Bayesian linear regression posterior for Gaussian likelihood
    with Gaussian prior.

    Given:
        w ~ N(mu0, Sigma0)          (prior)
        y_t ~ N(phi_t^T w, sigma2)  (likelihood)

    The posterior is:
        Lambda = Lambda0 + A          (posterior precision)
        eta = eta0 + b                (posterior natural parameter)
        mu = Lambda^{-1} @ eta        (posterior mean)
        Sigma = Lambda^{-1}           (posterior covariance)

    Computed via Cholesky factorisation for numerical stability:
        L = cholesky(Lambda)          (lower triangular)
        mu = L^{-T} @ L^{-1} @ eta

    Parameters
    ----------
    state : SellerState
        Seller sufficient statistics.
    prior : Prior
        Gaussian prior.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mu, Sigma)
        mu: posterior mean, shape (d,)
        Sigma: posterior covariance, shape (d, d)

    Raises
    ------
    ValueError
        If dimension mismatch between state and prior.

    Examples
    --------
    >>> prior = Prior.diagonal_prior(d=5, alpha=0.1)
    >>> state = zero_state(d=5)
    >>> phi = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    >>> state = apply_contribution(state, phi, reward=0.5, sigma2=0.06)
    >>> mu, Sigma = posterior_params(state, prior)
    >>> mu.shape
    (5,)
    >>> Sigma.shape
    (5, 5)
    """
    if state.d != prior.d:
        raise ValueError(
            f"Dimension mismatch: state.d={state.d}, prior.d={prior.d}"
        )

    # Posterior precision: Lambda = Lambda0 + A
    Lambda = prior.Lambda0 + state.A

    # Posterior natural parameter: eta = eta0 + b
    eta = prior.eta0 + state.b

    # Cholesky factorisation: Lambda = L @ L^T, L lower triangular
    L = np.linalg.cholesky(Lambda)

    # Solve L @ z = eta  =>  z = L^{-1} @ eta
    z = np.linalg.solve(L, eta)

    # Posterior mean: mu = L^{-T} @ z = Lambda^{-1} @ eta
    mu = np.linalg.solve(L.T, z)

    # Posterior covariance: Sigma = Lambda^{-1} = L^{-T} @ L^{-1}
    L_inv = np.linalg.solve(L, np.eye(state.d, dtype=np.float64))
    Sigma = L_inv.T @ L_inv

    return (mu, Sigma)
