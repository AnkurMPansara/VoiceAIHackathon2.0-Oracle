"""Predictive backtest for the Best Time to Call prediction system.

Implements SRS EVAL-02: Chronological predictive backtest with frozen priors
and hyperparameters. Processes each recommendation at decision time using only
labels finalized by that time.

Evaluates predictions at logged actual call time with metrics:
- Reward MSE by history count groups (0, 1-4, 5-9, >=10)
- Gaussian predictive NLL by history count groups
- Expected vs observed by supported time bin

Modules
-------
PriorBundle : Frozen prior bundle for replay.
chronological_replay : Chronological replay engine.
run_backtest : Full backtest entry point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from btc.config import ModelConfig, RewardConfig
from btc.features.fourier import fourier, time_to_hours
from btc.model.posterior import (
    Posterior,
    compute_posterior,
    predict_expected_reward,
    predict_uncertainty,
)
from btc.model.priors import SegmentPrior, fit_hierarchical_priors
from btc.model.reward import compute_reward
from btc.model.stats import Prior, SellerState, compute_contribution, posterior_params, zero_state

logger = logging.getLogger(__name__)


# ── PriorBundle ───────────────────────────────────────────────────────────────


@dataclass
class PriorBundle:
    """Frozen prior bundle for replay.

    Contains the global prior and segment priors used for predictions.
    These are frozen at a fixed point in time (e.g., September test)
    to ensure no data leakage during replay.

    Attributes
    ----------
    global_prior : Prior
        Global (root) Gaussian prior.
    segment_priors : dict[str, SegmentPrior]
        Segment key → SegmentPrior mapping.
    k : int
        Fourier basis order.
    d : int
        Feature dimension.
    bundle_id : str
        Identifier for this bundle version.
    bundle_date : str
        Date the bundle was created (ISO format).
    """

    global_prior: Prior
    segment_priors: dict[str, SegmentPrior]
    k: int
    d: int
    bundle_id: str = "frozen"
    bundle_date: str = "unknown"

    def get_segment_prior(self, segment_key: str) -> Prior:
        """Get the Prior for a segment key.

        Falls back to global_prior if segment not found.

        Parameters
        ----------
        segment_key : str
            Canonical segment key (JSON array string).

        Returns
        -------
        Prior
            Prior for the segment, or global prior as fallback.
        """
        seg = self.segment_priors.get(segment_key)
        if seg is not None:
            return Prior(
                mu0=seg.mu0,
                Sigma0=seg.Sigma0,
                Lambda0=seg.Lambda0,
                eta0=seg.eta0,
                d=self.d,
            )
        return self.global_prior


# ── Seller state builder ──────────────────────────────────────────────────────


def _resolve_prior(
    bundle: PriorBundle,
    segment_key: str,
) -> Prior:
    """Resolve the Prior for a segment from the bundle.

    Parameters
    ----------
    bundle : PriorBundle
        Frozen prior bundle.
    segment_key : str
        Canonical segment key.

    Returns
    -------
    Prior
        Segment prior or global prior fallback.
    """
    return bundle.get_segment_prior(segment_key)


# ── Gaussian NLL computation ─────────────────────────────────────────────────


def gaussian_nll(
    mean: float,
    std: float,
    observed: float,
) -> float:
    """Compute Gaussian negative log-likelihood for a single observation.

    NLL = 0.5 * log(2*pi*sigma^2) + 0.5 * (y - mu)^2 / sigma^2

    Parameters
    ----------
    mean : float
        Predicted mean (mu).
    std : float
        Predicted standard deviation (sigma). Must be > 0.
    observed : float
        Observed value (y).

    Returns
    -------
    float
        Negative log-likelihood. Returns a large value if std <= 0.
    """
    if std <= 0 or not np.isfinite(std):
        return 1e6  # Penalty for degenerate predictions

    sigma2 = std * std
    nll = 0.5 * np.log(2.0 * np.pi * sigma2) + 0.5 * (observed - mean) ** 2 / sigma2

    if not np.isfinite(nll):
        return 1e6

    return float(nll)


# ── History count groups ──────────────────────────────────────────────────────


def _history_count_group(n: int) -> str:
    """Map observation count to history count group.

    Parameters
    ----------
    n : int
        Number of prior observations.

    Returns
    -------
    str
        Group label: '0', '1-4', '5-9', '>=10'.
    """
    if n == 0:
        return "0"
    elif n <= 4:
        return "1-4"
    elif n <= 9:
        return "5-9"
    else:
        return ">=10"


# ── chronological_replay ─────────────────────────────────────────────────────


def chronological_replay(
    data: list[dict],
    bundle: PriorBundle,
    config: ModelConfig,
    reward_config: RewardConfig,
) -> list[dict]:
    """Process data chronologically, building seller state incrementally.

    At each step, computes prediction using only prior state (labels
    finalized before the current record's call time). The seller's
    sufficient statistics are updated after the prediction.

    This implements a strict point-in-time replay: no future data
    leaks into the prediction.

    Parameters
    ----------
    data : list[dict]
        Normalized outcome records. Must be sorted by call_start_time
        ascending. If not sorted, they will be sorted internally.
    bundle : PriorBundle
        Frozen prior bundle with global and segment priors.
    config : ModelConfig
        Model configuration.
    reward_config : RewardConfig
        Reward configuration for computing reward values.

    Returns
    -------
    list[dict]
        Replay results, each entry contains:
        - seller_id, lead_id, attempt_id
        - call_start_time, reward (observed)
        - predicted_reward, predicted_std, latent_std
        - history_count, history_group
        - gaussian_nll, reward_mse
        - is_fallback, fallback_reason
    """
    sigma2 = reward_config.sigma2
    k = config.k

    # Sort by call_start_time for chronological processing
    sorted_data = sorted(
        data,
        key=lambda r: r.get("call_start_time") or datetime.min.replace(tzinfo=timezone.utc),
    )

    # Per-seller state: seller_id -> (SellerState, segment_key)
    seller_states: dict[str, tuple[SellerState, str]] = {}

    results: list[dict] = []

    for record in sorted_data:
        seller_id = record.get("seller_id", "")
        lead_id = record.get("lead_id", "")
        attempt_id = record.get("attempt_id", "")
        call_start = record.get("call_start_time")
        segment = record.get("segment", "UNKNOWN")

        if call_start is None:
            continue

        # Compute observed reward
        try:
            observed_reward = compute_reward(
                answered=record.get("answered", False),
                meeting_fixed=record.get("meeting_fixed", False),
                disposition=record.get("disposition", "UNKNOWN"),
                reward_config=reward_config,
            )
        except ValueError:
            continue

        # Get or create seller state
        if seller_id not in seller_states:
            state = zero_state(d=bundle.d)
            seller_states[seller_id] = (state, segment)

        state, seg_key = seller_states[seller_id]

        # Resolve prior for this seller's segment
        prior = _resolve_prior(bundle, seg_key)

        # Compute posterior from current state
        posterior_result = compute_posterior(state, prior, sigma2)

        # Compute Fourier features for call time
        hours = time_to_hours(call_start)
        phi = fourier(hours, k=k)

        # Prediction
        is_fallback = False
        fallback_reason = None
        predicted_reward = 0.0
        predicted_std = 0.0
        latent_std = 0.0

        if isinstance(posterior_result, Posterior):
            predicted_reward = predict_expected_reward(phi, posterior_result)
            latent_std, predicted_std, _ = predict_uncertainty(phi, posterior_result)
        else:
            # Fallback: use prior mean
            is_fallback = True
            fallback_reason = posterior_result.reason
            predicted_reward = float(phi @ posterior_result.mu)
            # Fallback predictive_std: sqrt(trace(L @ L.T) + sigma2)
            L_trace = float(np.trace(posterior_result.L @ posterior_result.L.T))
            predicted_std = float(np.sqrt(max(0.0, L_trace + sigma2)))
            latent_std = 0.0

        # Metrics
        n = state.n
        history_group = _history_count_group(n)
        nll = gaussian_nll(predicted_reward, predicted_std, observed_reward)
        mse = (predicted_reward - observed_reward) ** 2

        results.append({
            "seller_id": seller_id,
            "lead_id": lead_id,
            "attempt_id": attempt_id,
            "call_start_time": call_start,
            "segment": seg_key,
            "reward": observed_reward,
            "predicted_reward": predicted_reward,
            "predicted_std": predicted_std,
            "latent_std": latent_std,
            "history_count": n,
            "history_group": history_group,
            "gaussian_nll": nll,
            "reward_mse": mse,
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
        })

        # Update seller state with this observation (after prediction)
        phi_array = np.asarray(phi, dtype=np.float64)
        new_state = apply_contribution(state, phi_array, observed_reward, sigma2)
        seller_states[seller_id] = (new_state, seg_key)

    return results


# ── Alias for compatibility ──────────────────────────────────────────────────


def apply_contribution(
    state: SellerState,
    phi: np.ndarray,
    reward: float,
    sigma2: float,
) -> SellerState:
    """Update seller state with a new observation.

    Convenience wrapper around btc.model.stats.apply_contribution.

    Parameters
    ----------
    state : SellerState
        Current seller state.
    phi : np.ndarray, shape (d,)
        Fourier feature vector.
    reward : float
        Reward value.
    sigma2 : float
        Working noise variance.

    Returns
    -------
    SellerState
        Updated state.
    """
    return btc.model.stats.apply_contribution(state, phi, reward, sigma2)


# ── Backtest aggregation ─────────────────────────────────────────────────────


def _aggregate_by_group(
    results: list[dict],
    group_key: str = "history_group",
) -> dict[str, dict[str, float]]:
    """Aggregate metrics by group (e.g., history count groups).

    Parameters
    ----------
    results : list[dict]
        Replay results.
    group_key : str
        Field to group by.

    Returns
    -------
    dict[str, dict[str, float]]
        Group label -> {metric: value} mapping.
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        g = r.get(group_key, "unknown")
        if g not in groups:
            groups[g] = []
        groups[g].append(r)

    aggregated: dict[str, dict[str, float]] = {}
    for g, records in groups.items():
        n = len(records)
        aggregated[g] = {
            "n": n,
            "reward_mse": float(np.mean([r["reward_mse"] for r in records])) if n > 0 else 0.0,
            "gaussian_nll": float(np.mean([r["gaussian_nll"] for r in records])) if n > 0 else 0.0,
            "predicted_reward_mean": float(np.mean([r["predicted_reward"] for r in records])) if n > 0 else 0.0,
            "observed_reward_mean": float(np.mean([r["reward"] for r in records])) if n > 0 else 0.0,
            "fallback_fraction": sum(1 for r in records if r.get("is_fallback")) / n if n > 0 else 0.0,
        }

    return aggregated


# ── Expected vs observed by time bin ─────────────────────────────────────────


def _expected_vs_observed_by_bin(
    results: list[dict],
) -> dict[str, dict[str, float]]:
    """Expected vs observed reward by supported time bin.

    Parameters
    ----------
    results : list[dict]
        Replay results with 'call_start_time' field.

    Returns
    -------
    dict[str, dict[str, float]]
        Time bin label -> {predicted_mean, observed_mean, n}.
    """
    from btc.evaluation.phase0 import _hour_bin_index, _BIN_LABELS

    bins: dict[int, list[dict]] = {i: [] for i in range(96)}

    for r in results:
        call_start = r.get("call_start_time")
        if call_start is None:
            continue
        idx = _hour_bin_index(call_start)
        if 0 <= idx < 96:
            bins[idx].append(r)

    result: dict[str, dict[str, float]] = {}
    for i in range(96):
        records = bins[i]
        if not records:
            continue
        n = len(records)
        result[_BIN_LABELS[i]] = {
            "predicted_mean": float(np.mean([r["predicted_reward"] for r in records])),
            "observed_mean": float(np.mean([r["reward"] for r in records])),
            "n": n,
        }

    return result


# ── run_backtest ──────────────────────────────────────────────────────────────


def run_backtest(
    normalized_data: list[dict],
    bundle: PriorBundle,
    config: ModelConfig,
    reward_config: RewardConfig,
    splits: Optional[dict] = None,
) -> dict:
    """EVAL-02: Chronological predictive backtest.

    Processes each recommendation at decision time using only labels
    finalized by that time. Evaluates at logged actual call time.

    Uses frozen priors and hyperparameters for the test period.

    Parameters
    ----------
    normalized_data : list[dict]
        Normalized outcome records.
    bundle : PriorBundle
        Frozen prior bundle for the test period.
    config : ModelConfig
        Model configuration.
    reward_config : RewardConfig
        Reward configuration.
    splits : dict | None
        Optional chronological splits dict. If provided, uses the
        'test' split for evaluation; otherwise uses all data.

    Returns
    -------
    dict
        Backtest report with keys:
        - overall: aggregate metrics across all predictions
        - by_history_group: metrics grouped by history count (0, 1-4, 5-9, >=10)
        - by_time_bin: expected vs observed by 15-minute time bin
        - summary: total predictions, fallback count, time range
    """
    # Select data: use test split if available, else all data
    if splits is not None and "test" in splits:
        test_data = splits["test"]
    else:
        test_data = normalized_data

    # Run chronological replay
    replay_results = chronological_replay(
        data=test_data,
        bundle=bundle,
        config=config,
        reward_config=reward_config,
    )

    if not replay_results:
        return {
            "overall": {
                "n": 0,
                "reward_mse": 0.0,
                "gaussian_nll": 0.0,
                "predicted_reward_mean": 0.0,
                "observed_reward_mean": 0.0,
            },
            "by_history_group": {},
            "by_time_bin": {},
            "summary": {
                "total_predictions": 0,
                "fallback_count": 0,
                "time_range": {"start": None, "end": None},
            },
        }

    # Overall metrics
    n = len(replay_results)
    overall = {
        "n": n,
        "reward_mse": float(np.mean([r["reward_mse"] for r in replay_results])),
        "gaussian_nll": float(np.mean([r["gaussian_nll"] for r in replay_results])),
        "predicted_reward_mean": float(np.mean([r["predicted_reward"] for r in replay_results])),
        "observed_reward_mean": float(np.mean([r["reward"] for r in replay_results])),
        "fallback_fraction": sum(1 for r in replay_results if r["is_fallback"]) / n,
    }

    # By history count group
    by_history_group = _aggregate_by_group(replay_results, "history_group")

    # Expected vs observed by time bin
    by_time_bin = _expected_vs_observed_by_bin(replay_results)

    # Time range
    call_times = [r["call_start_time"] for r in replay_results if r.get("call_start_time")]
    time_range = {
        "start": min(call_times).isoformat() if call_times else None,
        "end": max(call_times).isoformat() if call_times else None,
    }

    return {
        "overall": overall,
        "by_history_group": by_history_group,
        "by_time_bin": by_time_bin,
        "summary": {
            "total_predictions": n,
            "fallback_count": sum(1 for r in replay_results if r["is_fallback"]),
            "time_range": time_range,
        },
    }


# ── Convenience: build bundle from data ──────────────────────────────────────


def build_bundle(
    fit_data: list[dict],
    config: ModelConfig,
    bundle_id: str = "frozen",
    bundle_date: str = "unknown",
) -> PriorBundle:
    """Build a PriorBundle from fit data.

    Fits hierarchical priors from the fit data and returns a frozen
    bundle for replay.

    Parameters
    ----------
    fit_data : list[dict]
        Data used for prior fitting (e.g., prior_fit + warmup + validation splits).
    config : ModelConfig
        Model configuration.
    bundle_id : str
        Bundle identifier.
    bundle_date : str
        Bundle creation date.

    Returns
    -------
    PriorBundle
        Frozen prior bundle.
    """
    from btc.model.stats import Prior as StatsPrior

    # Fit hierarchical priors
    hierarchical = fit_hierarchical_priors(fit_data, config)

    # Convert SegmentPrior to Prior
    global_prior = Prior(
        mu0=hierarchical.global_prior.mu0,
        Sigma0=hierarchical.global_prior.Sigma0,
        Lambda0=hierarchical.global_prior.Lambda0,
        eta0=hierarchical.global_prior.eta0,
        d=hierarchical.d,
    )

    segment_priors: dict[str, Any] = {}
    for seg_key, seg_prior in hierarchical.segment_priors.items():
        segment_priors[seg_key] = seg_prior

    return PriorBundle(
        global_prior=global_prior,
        segment_priors=segment_priors,
        k=config.k,
        d=hierarchical.d,
        bundle_id=bundle_id,
        bundle_date=bundle_date,
    )
