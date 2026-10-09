"""Hierarchical prior fitting for the Best Time to Call prediction system.

Implements TRAIN-02: Hierarchical Bayesian priors with shrinkage.

The prior fitting process computes segment-level Gaussian priors for the
Fourier coefficients that model seller call-time preference patterns.
Priors are fit hierarchically with shrinkage toward parent segments:

    mu_segment = (n_segment * mu_empirical + lambda_parent * mu_parent)
                 / (n_segment + lambda_parent)

Smoothing is applied to the covariance via:

    Sigma_smooth = (1 - lambda_smooth) * Sigma_empirical
                   + lambda_smooth * trace(Sigma_empirical)/d * I

This module is designed to be called from :mod:`btc.model.trainer` and
provides pure functions with no side effects.

Modules
-------
fit_segment_prior : Fit a Gaussian prior for a single segment.
fit_hierarchical_priors : Fit hierarchical priors across all segments.
compute_segment_empirical : Compute empirical sufficient statistics per segment.
shrink_to_parent : Apply hierarchical shrinkage toward parent segment.
smooth_covariance : Apply smoothness penalty to covariance matrices.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from btc.config import ModelConfig, RewardConfig
from btc.features.fourier import feature_dim
from btc.model.stats import Prior, extract_upper_triangle, reconstruct_symmetric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SegmentPrior:
    """Gaussian prior for a single segment.

    Attributes
    ----------
    segment_key : str
        Canonical segment key (JSON array, TRAIN-09).
    mu0 : np.ndarray, shape (d,)
        Prior mean vector.
    Sigma0 : np.ndarray, shape (d, d)
        Prior covariance matrix (symmetric positive definite).
    Lambda0 : np.ndarray, shape (d, d)
        Prior precision matrix.
    eta0 : np.ndarray, shape (d,)
        Prior natural parameter: Lambda0 @ mu0.
    n_observations : int
        Number of observations used to fit this prior.
    n_sellers : int
        Number of distinct sellers in this segment.
    is_shrunk : bool
        True if this prior was shrunk toward a parent segment.
    parent_key : str | None
        Parent segment key, or None for global/root segment.
    d : int
        Feature dimension.
    """

    segment_key: str
    mu0: np.ndarray
    Sigma0: np.ndarray
    Lambda0: np.ndarray
    eta0: np.ndarray
    n_observations: int = 0
    n_sellers: int = 0
    is_shrunk: bool = False
    parent_key: Optional[str] = None
    d: int = 0

    def __post_init__(self) -> None:
        """Validate dimensions and SPD properties."""
        if self.mu0.shape != (self.d,):
            raise ValueError(f"mu0 shape must be ({self.d},), got {self.mu0.shape}")
        if self.Sigma0.shape != (self.d, self.d):
            raise ValueError(f"Sigma0 shape must be ({self.d}, {self.d}), got {self.Sigma0.shape}")
        if self.Lambda0.shape != (self.d, self.d):
            raise ValueError(f"Lambda0 shape must be ({self.d}, {self.d}), got {self.Lambda0.shape}")
        if self.eta0.shape != (self.d,):
            raise ValueError(f"eta0 shape must be ({self.d},), got {self.eta0.shape}")

        if not np.allclose(self.eta0, self.Lambda0 @ self.mu0):
            raise ValueError("eta0 must equal Lambda0 @ mu0")

        if not np.allclose(self.Lambda0, np.linalg.inv(self.Sigma0)):
            raise ValueError("Lambda0 must equal Sigma0^-1")

        # Ensure SPD
        if not _is_spd(self.Sigma0):
            raise ValueError("Sigma0 must be symmetric positive definite")


@dataclass
class HierarchicalPriorResult:
    """Result of hierarchical prior fitting.

    Attributes
    ----------
    segment_priors : dict[str, SegmentPrior]
        Segment key → SegmentPrior mapping.
    global_prior : SegmentPrior
        Global (root) segment prior.
    segment_hierarchy : dict[str, Optional[str]]
        Segment key → parent segment key mapping.
    d : int
        Feature dimension.
    k : int
        Fourier basis order.
    """

    segment_priors: dict[str, SegmentPrior] = field(default_factory=dict)
    global_prior: SegmentPrior = None  # type: ignore[assignment]
    segment_hierarchy: dict[str, Optional[str]] = field(default_factory=dict)
    d: int = 0
    k: int = 0

    def __post_init__(self) -> None:
        """Validate that global_prior is set."""
        if self.global_prior is None:
            raise ValueError("global_prior must be set")


# ---------------------------------------------------------------------------
# Helper: SPD check
# ---------------------------------------------------------------------------


def _is_spd(matrix: np.ndarray) -> bool:
    """Check if a matrix is symmetric positive definite.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to check.

    Returns
    -------
    bool
        True if symmetric positive definite.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if not np.allclose(matrix, matrix.T):
        return False
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


# ---------------------------------------------------------------------------
# compute_segment_empirical
# ---------------------------------------------------------------------------


def compute_segment_empirical(
    data: list[dict],
    reward_config: RewardConfig,
    k: int,
) -> dict[str, dict[str, Any]]:
    """Compute empirical sufficient statistics per segment.

    For each segment, computes:
    - n_observations: count of records
    - n_sellers: distinct seller count
    - sum_phi_reward: sum of phi * reward (sufficient statistic for mean)
    - sum_phi_phi: sum of outer(phi, phi), the plain Gram matrix (for the ridge fit)

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'segment', 'seller_id', 'call_start_time',
        'answered', 'meeting_fixed', 'disposition' fields.
    reward_config : RewardConfig
        Reward configuration for computing reward values.
    k : int
        Fourier basis order.

    Returns
    -------
    dict[str, dict]
        Segment key → empirical stats dict with keys:
        n_observations, n_sellers, sum_phi_reward, sum_phi_phi_reward
    """
    d = feature_dim(k)
    segment_data: dict[str, dict[str, Any]] = {}

    for record in data:
        segment = record.get("segment", "UNKNOWN")
        seller_id = record.get("seller_id")
        call_start = record.get("call_start_time")

        if call_start is None:
            continue

        # Compute reward
        try:
            from btc.model.reward import compute_reward
            reward = compute_reward(
                answered=record.get("answered", False),
                meeting_fixed=record.get("meeting_fixed", False),
                disposition=record.get("disposition", "UNKNOWN"),
                reward_config=reward_config,
            )
        except ValueError:
            # Skip records with invalid outcome consistency
            continue

        # Compute Fourier features
        from btc.features.fourier import fourier, time_to_hours
        hours = time_to_hours(call_start)
        phi = fourier(hours, k=k)

        if segment not in segment_data:
            segment_data[segment] = {
                "n_observations": 0,
                "sellers": set(),
                "sum_phi_reward": np.zeros(d, dtype=np.float64),
                "sum_phi_phi": np.zeros((d, d), dtype=np.float64),  
            }

        seg = segment_data[segment]
        seg["n_observations"] += 1
        if seller_id is not None:
            seg["sellers"].add(seller_id)
        seg["sum_phi_reward"] += phi * reward
        seg["sum_phi_phi"] += np.outer(phi, phi)  

    # Convert seller sets to counts
    result: dict[str, dict[str, Any]] = {}
    for segment, stats in segment_data.items():
        result[segment] = {
            "n_observations": stats["n_observations"],
            "n_sellers": len(stats["sellers"]),
            "sum_phi_reward": stats["sum_phi_reward"],
            "sum_phi_phi": stats["sum_phi_phi"],  
        }

    return result


# ---------------------------------------------------------------------------
# fit_segment_prior
# ---------------------------------------------------------------------------


def fit_segment_prior(
    empirical: dict[str, Any],
    d: int,
    sigma2: float,
    alpha: float = 0.1,
) -> SegmentPrior:
    """Fit a Gaussian prior for a single segment from empirical statistics.

    The prior mean is computed as:
        mu0 = (1/sigma2) * A^{-1} * b
    where A = sum_phi_phi_reward / sigma2 and b = sum_phi_reward / sigma2.

    If the empirical covariance is not SPD, a regularised diagonal prior
    is used instead (TRAIN-03).

    Parameters
    ----------
    empirical : dict
        Empirical statistics from compute_segment_empirical.
    d : int
        Feature dimension.
    sigma2 : float
        Working noise variance. Must be > 0.
    alpha : float
        Regularisation scale for diagonal fallback (TRAIN-03).

    Returns
    -------
    SegmentPrior
        Fitted prior for this segment.

    Raises
    ------
    ValueError
        If sigma2 <= 0.
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")

    n_obs = empirical["n_observations"]
    n_sellers = empirical["n_sellers"]
    sum_phi_reward = empirical["sum_phi_reward"]

    # Compute empirical covariance
    if n_obs > 1:
        # Covariance = E[phi phi^T * y] - E[phi * y] E[phi * y]^T / n
        # For Gaussian working approximation:
        # Sigma_empirical = (1/n) * sum_phi_phi_reward - mu_hat @ mu_hat^T
        # But we use the working approximation directly:
        # A = sum_phi_phi_reward / sigma2
        # b = sum_phi_reward / sigma2
        # Sigma = sigma2 * A^{-1}
        # mu = Sigma @ b = A^{-1} @ b

        base = Prior.diagonal_prior(d, alpha=alpha)
        A = empirical["sum_phi_phi"] / sigma2 + base.Lambda0
        mu0 = np.linalg.solve(A, empirical["sum_phi_reward"] / sigma2)
        Sigma0 = base.Sigma0
    else:
        # Not enough data — use diagonal prior
        mu0 = np.zeros(d, dtype=np.float64)
        Sigma0 = Prior.diagonal_prior(d, alpha=alpha).Sigma0

    # Compute precision and natural parameter
    Lambda0 = np.linalg.inv(Sigma0)
    eta0 = Lambda0 @ mu0

    return SegmentPrior(
        segment_key="UNKNOWN",  # Set by caller
        mu0=mu0,
        Sigma0=Sigma0,
        Lambda0=Lambda0,
        eta0=eta0,
        n_observations=n_obs,
        n_sellers=n_sellers,
        is_shrunk=False,
        parent_key=None,
        d=d,
    )


# ---------------------------------------------------------------------------
# smooth_covariance
# ---------------------------------------------------------------------------


def smooth_covariance(
    Sigma: np.ndarray,
    lambda_smooth: float,
    d: int,
) -> np.ndarray:
    """Apply smoothness penalty to covariance matrix.

    TRAIN-02: Smooths the covariance to encourage smooth Fourier
    coefficient patterns:

        Sigma_smooth = (1 - lambda_smooth) * Sigma
                       + lambda_smooth * (trace(Sigma)/d) * I

    Parameters
    ----------
    Sigma : np.ndarray, shape (d, d)
        Covariance matrix.
    lambda_smooth : float
        Smoothness penalty weight in [0, 1].
    d : int
        Feature dimension.

    Returns
    -------
    np.ndarray, shape (d, d)
        Smoothed covariance matrix.
    """
    if lambda_smooth <= 0.0:
        return Sigma.copy()

    trace_sigma = np.trace(Sigma)
    identity = np.eye(d, dtype=np.float64)
    Sigma_smooth = (1.0 - lambda_smooth) * Sigma + lambda_smooth * (trace_sigma / d) * identity

    # Ensure SPD after smoothing
    if not _is_spd(Sigma_smooth):
        # Fallback to diagonal
        diag_values = np.diag(Sigma_smooth)
        diag_values = np.maximum(diag_values, 1e-6)
        Sigma_smooth = np.diag(diag_values)

    return Sigma_smooth


# ---------------------------------------------------------------------------
# shrink_to_parent
# ---------------------------------------------------------------------------


def shrink_to_parent(
    child_prior: SegmentPrior,
    parent_prior: SegmentPrior,
    lambda_parent: float,
) -> SegmentPrior:
    """Apply hierarchical shrinkage toward parent segment prior.

    TRAIN-02: Shrinks the child segment prior toward the parent:

        mu_shrunk = (n_child * mu_child + lambda_parent * mu_parent)
                    / (n_child + lambda_parent)

        Sigma_shrunk = Sigma_child  (covariance unchanged)

    Parameters
    ----------
    child_prior : SegmentPrior
        Prior for the child segment.
    parent_prior : SegmentPrior
        Prior for the parent segment.
    lambda_parent : float
        Shrinkage weight toward parent.

    Returns
    -------
    SegmentPrior
        Shrunk prior with updated mu0, Lambda0, eta0.
    """
    n_child = child_prior.n_observations

    # Weighted combination of means
    total_weight = n_child + lambda_parent
    mu_shrunk = (n_child * child_prior.mu0 + lambda_parent * parent_prior.mu0) / total_weight

    # Covariance stays the same
    Sigma_shrunk = child_prior.Sigma0.copy()
    Lambda_shrunk = np.linalg.inv(Sigma_shrunk)
    eta_shrunk = Lambda_shrunk @ mu_shrunk

    return SegmentPrior(
        segment_key=child_prior.segment_key,
        mu0=mu_shrunk,
        Sigma0=Sigma_shrunk,
        Lambda0=Lambda_shrunk,
        eta0=eta_shrunk,
        n_observations=child_prior.n_observations,
        n_sellers=child_prior.n_sellers,
        is_shrunk=True,
        parent_key=child_prior.parent_key,
        d=child_prior.d,
    )


# ---------------------------------------------------------------------------
# fit_hierarchical_priors
# ---------------------------------------------------------------------------


def fit_hierarchical_priors(
    data: list[dict],
    config: ModelConfig,
) -> HierarchicalPriorResult:
    """Fit hierarchical priors across all segments.

    TRAIN-02: Implements the full hierarchical prior fitting pipeline:

    1. Compute empirical statistics per segment
    2. Fit raw priors for each segment
    3. Smooth covariances with lambda_smooth penalty
    4. Build segment hierarchy from segment keys
    5. Apply hierarchical shrinkage toward parent segments
    6. Return global prior (pooled across all segments)

    Segment hierarchy is derived from the canonical JSON array segment keys:
    - ["global"] → root (no parent)
    - ["category_group"] → parent is "global"
    - ["category_group", "turnover_band"] → parent is ["category_group"]

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'segment', 'seller_id', 'call_start_time',
        'answered', 'meeting_fixed', 'disposition' fields.
    config : ModelConfig
        Model configuration with lambda_smooth, lambda_parent, alpha.

    Returns
    -------
    HierarchicalPriorResult
        Fitted priors for all segments plus global prior.

    Raises
    ------
    ValueError
        If reward sigma2 is invalid or feature dimension is inconsistent.
    """
    k = config.k
    d = feature_dim(k)
    sigma2 = config.reward_config.sigma2
    lambda_smooth = config.lambda_smooth
    lambda_parent = config.lambda_parent
    alpha = config.alpha

    if sigma2 <= 0:
        raise ValueError(f"reward_config.sigma2 must be > 0, got {sigma2}")

    # Step 1: Compute empirical statistics per segment
    empirical = compute_segment_empirical(data, config.reward_config, k)

    if not empirical:
        raise ValueError("No valid data for prior fitting")

    # Step 2: Build segment hierarchy
    global_key = json.dumps(["global"])
    segment_hierarchy: dict[str, Optional[str]] = {}
    for seg_key in empirical:
        try:
            path = json.loads(seg_key)
        except (json.JSONDecodeError, TypeError):
            path = ["UNKNOWN"]

        if not isinstance(path, list):
            path = ["UNKNOWN"]

        if seg_key == global_key:
            # Global is the root: no parent
            segment_hierarchy[seg_key] = None
        elif len(path) <= 1:
            # One-level segments hang off the global root
            segment_hierarchy[seg_key] = global_key
        else:
            # Parent is the prefix of length len-1
            segment_hierarchy[seg_key] = json.dumps(path[:-1])

    # Step 3: Fit raw priors for each segment
    raw_priors: dict[str, SegmentPrior] = {}
    for seg_key, stats in empirical.items():
        prior = fit_segment_prior(stats, d, sigma2, alpha)
        prior.segment_key = seg_key
        prior.parent_key = segment_hierarchy.get(seg_key)
        raw_priors[seg_key] = prior

    # Step 4: Smooth covariances
    for seg_key, prior in raw_priors.items():
        prior.Sigma0 = smooth_covariance(prior.Sigma0, lambda_smooth, d)
        prior.Lambda0 = np.linalg.inv(prior.Sigma0)
        prior.eta0 = prior.Lambda0 @ prior.mu0

    # Step 5: Compute global prior (pooled across all segments)
    total_n = sum(s["n_observations"] for s in empirical.values())

    pooled = {
        "n_observations": total_n,
        "n_sellers": len(set(r.get("seller_id") for r in data if r.get("seller_id"))),
        "sum_phi_reward": sum(s["sum_phi_reward"] for s in empirical.values()),
        "sum_phi_phi": sum(s["sum_phi_phi"] for s in empirical.values()),
    }
    _global_fit = fit_segment_prior(pooled, d, sigma2, alpha)
    global_mu = _global_fit.mu0
    global_Sigma = _global_fit.Sigma0

    global_Lambda = np.linalg.inv(global_Sigma)
    global_eta = global_Lambda @ global_mu

    global_prior = SegmentPrior(
        segment_key=global_key,
        mu0=global_mu,
        Sigma0=global_Sigma,
        Lambda0=global_Lambda,
        eta0=global_eta,
        n_observations=total_n,
        n_sellers=len(set(
            r.get("seller_id") for r in data if r.get("seller_id")
        )),
        is_shrunk=False,
        parent_key=None,
        d=d,
    )

    # Step 6: Apply hierarchical shrinkage
    shrunk_priors: dict[str, SegmentPrior] = {}
    for seg_key, prior in raw_priors.items():
        # The global segment has no parent; never shrink it toward itself
        if seg_key == global_key:
            shrunk_priors[seg_key] = prior
            continue

        # Walk up to the nearest ancestor that actually has a raw prior
        anc = segment_hierarchy.get(seg_key)
        while anc is not None and anc not in raw_priors:
            p = json.loads(anc)
            anc = json.dumps(p[:-1]) if len(p) > 1 else None

        parent_prior = raw_priors[anc] if anc is not None else global_prior
        shrunk_priors[seg_key] = shrink_to_parent(prior, parent_prior, lambda_parent)

    # Update global prior to include all data
    global_prior.n_observations = total_n
    global_prior.n_sellers = len(set(
        r.get("seller_id") for r in data if r.get("seller_id")
    ))

    return HierarchicalPriorResult(
        segment_priors=shrunk_priors,
        global_prior=global_prior,
        segment_hierarchy=segment_hierarchy,
        d=d,
        k=k,
    )
