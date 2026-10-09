"""Experiment metrics for the Best Time to Call prediction system.

Implements SRS EVAL-03/04 (one-step OPE), EVAL-05 (online experiment metrics),
and EVAL-06 (seller-cluster confidence intervals):

- IPS, SNIPS, DR estimators with weight clipping, 95% CI from bootstrap
- Online metrics: meeting rate, answer rate, call yield, retry conversion,
  delay percentiles, call pressure
- Seller-cluster confidence intervals via 1000-iteration bootstrap

Modules
-------
compute_ope : One-step policy evaluation (IPS, SNIPS, DR).
compute_online_metrics : Online experiment metrics.
bootstrap_ci : 95% CI from bootstrap samples.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


# ── Weight clipping constant ──────────────────────────────────────────────────

_WEIGHT_CLIP_MAX = 10.0
"""Maximum weight for IPS/SNIPS clipping (EVAL-03/04)."""


# ── bootstrap_ci ──────────────────────────────────────────────────────────────


def bootstrap_ci(
    samples: list,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute 95% CI from bootstrap samples.

    Resamples the input with replacement n_bootstrap times and
    computes the 2.5th and 97.5th percentiles of the bootstrap
    distribution of the mean.

    Parameters
    ----------
    samples : list
        List of scalar values to bootstrap.
    n_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        (lower_95, upper_95) confidence interval bounds.
        Returns (0.0, 0.0) if no samples.
    """
    if not samples:
        return (0.0, 0.0)

    samples_array = np.array(samples, dtype=np.float64)

    if len(samples_array) == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    n = len(samples_array)
    bootstrap_means: list[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_sample = samples_array[indices]
        boot_mean = float(np.mean(boot_sample))
        bootstrap_means.append(boot_mean)

    bootstrap_array = np.array(bootstrap_means, dtype=np.float64)
    lower = float(np.percentile(bootstrap_array, 2.5))
    upper = float(np.percentile(bootstrap_array, 97.5))

    return (lower, upper)


# ── compute_ope ───────────────────────────────────────────────────────────────


def compute_ope(
    logged_decisions: list[dict],
    config,
) -> dict:
    """EVAL-03/04: One-step policy evaluation.

    Computes IPS, SNIPS, and DR estimators for off-policy evaluation
    using logged decisions and outcomes.

    IPS (Inverse Propensity Scoring):
        IPS = (1/N) * sum( (y_i * pi_new(a_i|x_i)) / pi_old(a_i|x_i) )

    SNIPS (Self-Normalized IPS):
        SNIPS = sum( w_i * y_i ) / sum( w_i )
        where w_i = pi_new(a_i|x_i) / pi_old(a_i|x_i)

    DR (Doubly Robust):
        DR = IPS + (1/N) * sum( y_i - y_hat_i ) * (pi_new / pi_old - 1)

    Weight clipping is applied at max_weight (default 10).
    95% CI is computed from 1000 seller-cluster bootstrap iterations.

    Parameters
    ----------
    logged_decisions : list[dict]
        Logged decision records. Each dict must contain:
        - seller_id: str
        - reward: float (observed outcome)
        - action_probability: float (pi_old, probability under behavior policy)
        - assignment_probability: float | None (assignment arm probability)
        - predicted_reward: float | None (baseline prediction for DR)
        - policy: str | None (policy used)
    config : Any
        Configuration object (used for clip threshold if available).

    Returns
    -------
    dict
        OPE report with keys:
        - ips: float (point estimate)
        - snips: float (point estimate)
        - dr: float | None (point estimate, None if predicted_reward missing)
        - ips_ci: tuple[float, float] (95% CI)
        - snips_ci: tuple[float, float] (95% CI)
        - dr_ci: tuple[float, float] | None (95% CI)
        - ess: float (effective sample size)
        - clipping_fraction: float (fraction of weights clipped)
        - n: int (total decisions)
        - overlap: float (fraction with non-zero behavior prob)
    """
    clip_max = _WEIGHT_CLIP_MAX

    # Extract fields
    seller_ids: list[str] = []
    rewards: list[float] = []
    action_probs: list[float] = []
    assignment_probs: list[float] = []
    predicted_rewards: list[float] = []
    has_predicted: list[bool] = []

    for record in logged_decisions:
        seller_id = record.get("seller_id", "")
        reward = record.get("reward", 0.0)
        action_prob = record.get("action_probability")
        assignment_prob = record.get("assignment_probability")
        predicted_reward = record.get("predicted_reward")

        seller_ids.append(seller_id)
        rewards.append(float(reward))

        if action_prob is not None and action_prob > 0:
            action_probs.append(float(action_prob))
        else:
            action_probs.append(0.0)

        if assignment_prob is not None:
            assignment_probs.append(float(assignment_prob))
        else:
            assignment_probs.append(1.0)

        if predicted_reward is not None:
            predicted_rewards.append(float(predicted_reward))
            has_predicted.append(True)
        else:
            predicted_rewards.append(0.0)
            has_predicted.append(False)

    n = len(rewards)
    if n == 0:
        return {
            "ips": 0.0,
            "snips": 0.0,
            "dr": None,
            "ips_ci": (0.0, 0.0),
            "snips_ci": (0.0, 0.0),
            "dr_ci": None,
            "ess": 0.0,
            "clipping_fraction": 0.0,
            "n": 0,
            "overlap": 0.0,
        }

    # Compute propensity weights: w_i = 1 / pi_old(a_i|x_i)
    # For IPS, we use the behavior policy probability as the denominator.
    # The "new" policy probability is assumed to be 1 for the chosen action.
    weights = []
    clipped_count = 0
    overlap_count = 0

    for i in range(n):
        p = action_probs[i]
        if p > 0:
            overlap_count += 1
            w = 1.0 / p
            if w > clip_max:
                w = clip_max
                clipped_count += 1
            weights.append(w)
        else:
            weights.append(0.0)

    weights_array = np.array(weights, dtype=np.float64)
    rewards_array = np.array(rewards, dtype=np.float64)

    # IPS: (1/N) * sum(w_i * y_i)
    ips = float(np.mean(weights_array * rewards_array))

    # SNIPS: sum(w_i * y_i) / sum(w_i)
    sum_w = float(np.sum(weights_array))
    if sum_w > 0:
        snips = float(np.sum(weights_array * rewards_array) / sum_w)
    else:
        snips = 0.0

    # ESS: (sum(w_i))^2 / sum(w_i^2)
    sum_w2 = float(np.sum(weights_array ** 2))
    if sum_w2 > 0:
        ess = (sum_w ** 2) / sum_w2
    else:
        ess = 0.0

    # Clipping fraction
    clipping_fraction = clipped_count / n if n > 0 else 0.0

    # Overlap fraction
    overlap = overlap_count / n if n > 0 else 0.0

    # DR (Doubly Robust): only if predicted rewards are available
    dr = None
    dr_ci: Optional[tuple[float, float]] = None
    all_predicted = all(has_predicted)

    if all_predicted:
        predicted_array = np.array(predicted_rewards, dtype=np.float64)
        # DR = IPS + (1/N) * sum( (y_i - y_hat_i) * (w_i - 1) )
        # Note: w_i is already the importance weight 1/pi_old
        dr_values = weights_array * rewards_array  # IPS terms
        dr_correction = (rewards_array - predicted_array) * (weights_array - 1.0)
        dr = float(np.mean(dr_values + dr_correction))

    # Bootstrap CI: cluster by seller_id
    ips_bootstrap = _bootstrap_clustered(
        seller_ids, rewards_array, weights_array,
        estimator="ips", clip_max=clip_max,
        n_bootstrap=1000,
    )
    snips_bootstrap = _bootstrap_clustered(
        seller_ids, rewards_array, weights_array,
        estimator="snips", clip_max=clip_max,
        n_bootstrap=1000,
    )

    ips_ci = bootstrap_ci(ips_bootstrap, n_bootstrap=1000, seed=42)
    snips_ci = bootstrap_ci(snips_bootstrap, n_bootstrap=1000, seed=42)

    if dr is not None:
        dr_bootstrap = _bootstrap_clustered(
            seller_ids, rewards_array, weights_array, predicted_array,
            estimator="dr", clip_max=clip_max,
            n_bootstrap=1000,
        )
        dr_ci = bootstrap_ci(dr_bootstrap, n_bootstrap=1000, seed=42)

    return {
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "ips_ci": ips_ci,
        "snips_ci": snips_ci,
        "dr_ci": dr_ci,
        "ess": ess,
        "clipping_fraction": clipping_fraction,
        "n": n,
        "overlap": overlap,
    }


# ── Clustered bootstrap helper ───────────────────────────────────────────────


def _bootstrap_clustered(
    seller_ids: list[str],
    rewards: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    predicted: Optional[npt.NDArray[np.float64]] = None,
    estimator: str = "ips",
    clip_max: float = _WEIGHT_CLIP_MAX,
    n_bootstrap: int = 1000,
) -> list[float]:
    """Clustered bootstrap for OPE estimators.

    Resamples at the seller level (cluster bootstrap) to account
    for within-seller correlation.

    Parameters
    ----------
    seller_ids : list[str]
        Seller IDs for each observation.
    rewards : np.ndarray
        Observed rewards.
    weights : np.ndarray
        Importance weights.
    predicted : np.ndarray | None
        Predicted rewards for DR estimator.
    estimator : str
        Estimator type: 'ips', 'snips', 'dr'.
    clip_max : float
        Maximum weight clip value.
    n_bootstrap : int
        Number of bootstrap iterations.

    Returns
    -------
    list[float]
        Bootstrap distribution of the estimator.
    """
    rng = np.random.default_rng(42)
    unique_sellers = list(set(seller_ids))

    if not unique_sellers:
        return [0.0] * n_bootstrap

    bootstrap_values: list[float] = []

    for _ in range(n_bootstrap):
        # Resample sellers with replacement
        boot_sellers = rng.choice(unique_sellers, size=len(unique_sellers), replace=True)

        # Build mask for bootstrapped sellers
        seller_set = set(boot_sellers)
        mask = np.array([sid in seller_set for sid in seller_ids], dtype=np.float64)

        # Compute weighted estimator
        boot_rewards = rewards * mask
        boot_weights = weights * mask

        # Clip weights
        boot_weights = np.minimum(boot_weights, clip_max)

        sum_w = float(np.sum(boot_weights))
        sum_wy = float(np.sum(boot_weights * boot_rewards))

        if estimator == "ips":
            n_total = len(rewards)
            value = sum_wy / n_total if n_total > 0 else 0.0
        elif estimator == "snips":
            value = sum_wy / sum_w if sum_w > 0 else 0.0
        elif estimator == "dr" and predicted is not None:
            boot_predicted = predicted * mask
            dr_values = boot_weights * boot_rewards
            dr_correction = (boot_rewards - boot_predicted) * (boot_weights - mask)
            n_total = len(rewards)
            value = float(np.mean(dr_values + dr_correction)) if n_total > 0 else 0.0
        else:
            value = 0.0

        bootstrap_values.append(value)

    return bootstrap_values


# ── compute_online_metrics ───────────────────────────────────────────────────


def compute_online_metrics(
    decisions: list[dict],
    outcomes: list[dict],
    attribution_window_days: int = 7,
) -> dict:
    """EVAL-05: Online experiment metrics.

    Computes key online metrics for experiment monitoring:
    - Meeting rate, answer rate, call yield
    - Retry conversion, not interested rate
    - Delay (p50/p95), call pressure

    Parameters
    ----------
    decisions : list[dict]
        Decision records with fields:
        - seller_id, lead_id, decision_id
        - scheduled_at: datetime (recommended call time)
        - mode: str (policy mode)
        - assignment: str (experiment arm)
    outcomes : list[dict]
        Outcome records with fields:
        - seller_id, lead_id, decision_id
        - call_start_time: datetime
        - answered: bool
        - disposition: str
        - meeting_fixed: bool
    attribution_window_days : int
        Days window for attributing outcomes to decisions.

    Returns
    -------
    dict
        Online metrics with keys:
        - meeting_rate: float
        - answer_rate: float
        - call_yield: float
        - retry_conversion_rate: float
        - not_interested_rate: float
        - delay_p50_hours: float
        - delay_p95_hours: float
        - call_pressure_per_seller: float
        - decisions_by_mode: dict[str, int]
        - decisions_by_assignment: dict[str, int]
        - total_decisions: int
        - matched_outcomes: int
    """
    if not decisions:
        return {
            "meeting_rate": 0.0,
            "answer_rate": 0.0,
            "call_yield": 0.0,
            "retry_conversion_rate": 0.0,
            "not_interested_rate": 0.0,
            "delay_p50_hours": 0.0,
            "delay_p95_hours": 0.0,
            "call_pressure_per_seller": 0.0,
            "decisions_by_mode": {},
            "decisions_by_assignment": {},
            "total_decisions": 0,
            "matched_outcomes": 0,
        }

    # Index outcomes by decision_id
    outcome_by_decision: dict[str, dict] = {}
    for outcome in outcomes:
        dec_id = outcome.get("decision_id")
        if dec_id:
            outcome_by_decision[dec_id] = outcome

    # Match decisions to outcomes
    matched: list[dict] = []
    delays: list[float] = []
    meetings = 0
    answered = 0
    not_interested = 0
    retries = 0
    total_attempts = 0
    sellers_set: set[str] = set()

    mode_counts: Counter = defaultdict(int)
    assignment_counts: Counter = defaultdict(int)

    for decision in decisions:
        dec_id = decision.get("decision_id", "")
        seller_id = decision.get("seller_id", "")
        mode = decision.get("mode", "UNKNOWN")
        assignment = decision.get("assignment", "UNKNOWN")

        mode_counts[mode] += 1
        assignment_counts[assignment] += 1
        sellers_set.add(seller_id)

        outcome = outcome_by_decision.get(dec_id)
        if outcome is None:
            continue

        matched.append({
            "decision": decision,
            "outcome": outcome,
        })

        # Call yield: has an outcome
        if outcome.get("call_start_time"):
            # Delay: scheduled_at -> call_start_time
            scheduled = decision.get("scheduled_at")
            call_start = outcome.get("call_start_time")
            if scheduled and call_start:
                try:
                    if isinstance(scheduled, datetime) and isinstance(call_start, datetime):
                        delta = (call_start - scheduled).total_seconds() / 3600.0
                        if delta >= 0:
                            delays.append(delta)
                except (TypeError, ValueError):
                    pass

            # Metrics from outcome
            if outcome.get("answered"):
                answered += 1
            if outcome.get("meeting_fixed"):
                meetings += 1
            disposition = outcome.get("disposition", "")
            if disposition == "NOT_INTERESTED":
                not_interested += 1
            if disposition == "CALL_LATER_BUSY":
                retries += 1

        total_attempts += 1

    n_decisions = len(decisions)
    n_matched = len(matched)

    # Compute delay percentiles
    if delays:
        delays_array = np.array(delays, dtype=np.float64)
        delay_p50 = float(np.percentile(delays_array, 50))
        delay_p95 = float(np.percentile(delays_array, 95))
    else:
        delay_p50 = 0.0
        delay_p95 = 0.0

    # Call pressure: outcomes per seller
    n_sellers = len(sellers_set) if sellers_set else 1
    call_pressure = total_attempts / n_sellers

    return {
        "meeting_rate": meetings / total_attempts if total_attempts > 0 else 0.0,
        "answer_rate": answered / total_attempts if total_attempts > 0 else 0.0,
        "call_yield": n_matched / n_decisions if n_decisions > 0 else 0.0,
        "retry_conversion_rate": (
            retries / total_attempts if total_attempts > 0 else 0.0
        ),
        "not_interested_rate": (
            not_interested / total_attempts if total_attempts > 0 else 0.0
        ),
        "delay_p50_hours": delay_p50,
        "delay_p95_hours": delay_p95,
        "call_pressure_per_seller": call_pressure,
        "decisions_by_mode": dict(mode_counts),
        "decisions_by_assignment": dict(assignment_counts),
        "total_decisions": n_decisions,
        "matched_outcomes": n_matched,
    }


# ── Seller-cluster CI ────────────────────────────────────────────────────────


def seller_cluster_ci(
    decisions: list[dict],
    outcomes: list[dict],
    metric_key: str = "reward",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """EVAL-06: Seller-cluster confidence intervals.

    Computes confidence intervals for a metric, clustered by seller_id.
    Resamples sellers (not individual observations) to account for
    within-seller correlation.

    Parameters
    ----------
    decisions : list[dict]
        Decision records with 'seller_id' field.
    outcomes : list[dict]
        Outcome records with 'seller_id' and 'decision_id' fields.
    metric_key : str
        Field name in outcome records to compute CI for.
    n_bootstrap : int
        Number of bootstrap iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {
            'metric': str,
            'point_estimate': float,
            'ci_lower': float,
            'ci_upper': float,
            'n_sellers': int,
            'n_observations': int,
        }
    """
    # Index outcomes by decision_id
    outcome_by_dec: dict[str, dict] = {}
    for outcome in outcomes:
        dec_id = outcome.get("decision_id")
        if dec_id:
            outcome_by_dec[dec_id] = outcome

    # Build seller-level data
    seller_data: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []

    for decision in decisions:
        dec_id = decision.get("decision_id", "")
        seller_id = decision.get("seller_id", "")
        outcome = outcome_by_dec.get(dec_id)

        if outcome is None:
            continue

        value = outcome.get(metric_key)
        if value is None:
            continue

        try:
            v = float(value)
        except (ValueError, TypeError):
            continue

        seller_data[seller_id].append(v)
        all_values.append(v)

    if not all_values:
        return {
            "metric": metric_key,
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_sellers": 0,
            "n_observations": 0,
        }

    # Compute per-seller mean
    seller_means: dict[str, float] = {}
    for sid, vals in seller_data.items():
        seller_means[sid] = float(np.mean(vals))

    unique_sellers = list(seller_data.keys())
    n_sellers = len(unique_sellers)

    # Cluster bootstrap
    rng = np.random.default_rng(seed)
    bootstrap_means: list[float] = []

    for _ in range(n_bootstrap):
        boot_sellers = rng.choice(unique_sellers, size=n_sellers, replace=True)
        boot_values = [seller_means[s] for s in boot_sellers]
        boot_mean = float(np.mean(boot_values))
        bootstrap_means.append(boot_mean)

    point_estimate = float(np.mean(all_values))
    ci_lower, ci_upper = bootstrap_ci(bootstrap_means, n_bootstrap=n_bootstrap, seed=seed)

    return {
        "metric": metric_key,
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_sellers": n_sellers,
        "n_observations": len(all_values),
    }


# ── Counter import alias ──────────────────────────────────────────────────────


from collections import Counter
