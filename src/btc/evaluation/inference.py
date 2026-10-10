"""Best Time to Call inference module.

Provides two tiers of inference functionality:

1. **Pure inference** – ``predict_for_time_slots()`` takes pre-computed inputs
   (seller state, segment prior, candidate hours, etc.) and returns scored
   time slots with zero side effects: no file I/O, no printing, no logging.
   This function is intended to be called from root-level scripts (e.g.
   ``inference_for_seller.py``) that handle data loading, state construction,
   and output.

2. **Full pipeline** – ``predict_for_seller()``, ``predict_for_seller_with_training()``,
   and ``generate_inference_report()`` orchestrate the entire workflow: loading
   CSV data, building seller states, scoring hours, and computing accuracy /
   historical comparison metrics.

Modules
-------
predict_for_time_slots : Pure inference – score candidate time slots.
predict_for_seller : Generate predictions for a single seller (no training data).
predict_for_seller_with_training : Generate predictions using training data.
generate_inference_report : Full report generation pipeline (loads CSVs, scores sellers).
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Optional
from datetime import datetime, timezone, timedelta

import numpy as np

from btc.config import RewardConfig
from btc.data.adapters import load_attempts_csv, load_sellers_csv
from btc.data.normalization import (
    normalize_outcome,
    create_chronological_splits,
)
from btc.features.fourier import fourier, time_to_hours
from btc.model.bundle import load_bundle
from btc.model.posterior import compute_posterior, predict_expected_reward, score_candidates
from btc.model.reward import compute_reward
from btc.model.stats import Prior, SellerState, zero_state, apply_contribution

logger = logging.getLogger(__name__)


# ── Circular hour helpers ─────────────────────────────────────────────────────


def circular_distance(h1: float, h2: float) -> float:
    """Compute the circular (wraparound-aware) distance between two hours.

    Hours are on a 24-hour clock, so the distance between 23 and 1 is 2,
    not 22.

    Parameters
    ----------
    h1 : float
        First hour (0-23).
    h2 : float
        Second hour (0-23).

    Returns
    -------
    float
        Circular distance in hours (0-12).
    """
    diff = abs(h1 - h2)
    return min(diff, 24 - diff)


# ── Test data loading ────────────────────────────────────────────────────────


def load_test_data(data_dir: str, splits: dict) -> list[dict]:
    """Load test split data from normalized records.

    Parameters
    ----------
    data_dir : str
        Directory containing CSV data files.
    splits : dict
        Chronological splits dictionary with 'test' key.

    Returns
    -------
    list[dict]
        Normalized test records.
    """
    test_records = splits.get("test", [])
    logger.info("Loaded %d test records", len(test_records))
    return test_records


# ── Model bundle loading ─────────────────────────────────────────────────────


def load_model_bundle(bundle_path: str) -> tuple[dict, dict]:
    """Load model bundle metadata and arrays.

    Parameters
    ----------
    bundle_path : str
        Path to the model bundle directory.

    Returns
    -------
    tuple[dict, dict]
        (metadata, arrays) where arrays maps array names to numpy arrays.

    Raises
    ------
    ValueError
        If bundle validation fails.
    """
    metadata, arrays = load_bundle(bundle_path)
    logger.info("Loaded bundle %s (k=%d, d=%d, segments=%d)",
                metadata.get("bundle_id", ""),
                metadata.get("k", 0),
                metadata.get("d", 0),
                len(metadata.get("segment_keys", [])))
    return metadata, arrays


# ── Raw Prior helper (bypasses __post_init__ validation) ────────────────────


def _make_raw_prior(
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    Lambda0: np.ndarray,
    eta0: np.ndarray,
    d: int,
) -> Prior:
    """Create a Prior object directly, bypassing __post_init__ validation.

    Bundle arrays may have minor numerical inconsistencies that the strict
    Prior.__post_init__ would reject. This helper creates the object
    without validation, then fixes eta0 to be consistent.

    Parameters
    ----------
    mu0 : np.ndarray, shape (d,)
        Prior mean.
    Sigma0 : np.ndarray, shape (d, d)
        Prior covariance.
    Lambda0 : np.ndarray, shape (d, d)
        Prior precision.
    eta0 : np.ndarray, shape (d,)
        Prior natural parameter.
    d : int
        Feature dimension.

    Returns
    -------
    Prior
        A Prior with consistent parameters.
    """
    # Fix eta0 to be consistent with Lambda0 @ mu0
    eta0 = (Lambda0 @ mu0).astype(np.float64)

    # Create via object creation bypassing __post_init__
    prior = Prior.__new__(Prior)
    prior.mu0 = mu0
    prior.Sigma0 = Sigma0
    prior.Lambda0 = Lambda0
    prior.eta0 = eta0
    prior.d = d
    return prior


# ── Reward config from bundle metadata ───────────────────────────────────────


def _reward_config_from_metadata(metadata: dict) -> RewardConfig:
    """Create a RewardConfig from bundle metadata reward_params.

    Parameters
    ----------
    metadata : dict
        Bundle metadata.

    Returns
    -------
    RewardConfig
        Reward configuration.
    """
    rp = metadata.get("reward_params", {})
    return RewardConfig(
        w_meeting=float(rp.get("w_meeting", 1.0)),
        w_answered=float(rp.get("w_answered", 0.1)),
        c_dial=float(rp.get("c_dial", 0.02)),
        w_not_interested=float(rp.get("w_not_interested", 0.0)),
        sigma2=float(metadata.get("sigma2", 0.06)),
    )


# ── Segment prior lookup ─────────────────────────────────────────────────────
GLOBAL_SEGMENT_KEY = json.dumps(["global"])  # '["global"]'


def _parent_key(key: str) -> Optional[str]:
    """Derive the parent segment key from the key itself."""
    try:
        p = json.loads(key)
    except (TypeError, ValueError):
        return GLOBAL_SEGMENT_KEY if key != GLOBAL_SEGMENT_KEY else None
    if not isinstance(p, list) or not p:
        return GLOBAL_SEGMENT_KEY
    if len(p) > 1:
        return json.dumps(p[:-1])
    return None if p == ["global"] else GLOBAL_SEGMENT_KEY


def _make_global_prior(bundle_arrays: dict, d: int) -> Prior:
    """Build the bundle's global prior (replaces the zeros/eye fallback)."""
    try:
        return _make_raw_prior(
            bundle_arrays["global_mu0"],
            bundle_arrays["global_sigma0"],
            bundle_arrays["global_lambda0"],
            bundle_arrays["global_eta0"],
            d,
        )
    except KeyError as exc:
        raise ValueError(f"Bundle is missing global prior array: {exc}") from exc

def _find_segment_prior(
    segment_key: str,
    metadata: dict,
    segment_keys: list[str],
    segment_mu0: np.ndarray,
    segment_sigma0: np.ndarray,
    segment_lambda0: np.ndarray,
    segment_eta0: np.ndarray,
) -> Optional[Prior]:
    """Find the segment prior for a given segment key.

    Walks up the segment hierarchy if the exact segment is not found.

    Parameters
    ----------
    segment_key : str
        Canonical segment key (JSON array string).
    metadata : dict
        Bundle metadata (for hierarchy).
    segment_keys : list[str]
        List of segment keys from bundle.
    segment_mu0 : np.ndarray, shape (n_segments, d)
        Per-segment prior means.
    segment_sigma0 : np.ndarray, shape (n_segments, d, d)
        Per-segment prior covariances.
    segment_lambda0 : np.ndarray, shape (n_segments, d, d)
        Per-segment prior precisions.
    segment_eta0 : np.ndarray, shape (n_segments, d)
        Per-segment prior natural parameters.

    Returns
    -------
    Prior or None
        The segment prior, or None if no prior found.
    """
    d = segment_mu0.shape[1]

    def _try_idx(key: str) -> Optional[int]:
        for i, k in enumerate(segment_keys):
            if k == key:
                return i
        return None

    key: Optional[str] = segment_key
    while key is not None:
        idx = _try_idx(key)
        if idx is not None:
            return _make_raw_prior(
                segment_mu0[idx], segment_sigma0[idx],
                segment_lambda0[idx], segment_eta0[idx], d,
            )
        key = _parent_key(key)
    return None


# ── Seller state builder ─────────────────────────────────────────────────────


def _build_seller_state(
    seller_records: list[dict],
    reward_config: RewardConfig,
    sigma2: float,
    d: int,
) -> tuple[SellerState, list[float], list[np.ndarray]]:
    """Build seller sufficient statistics from training records.

    Parameters
    ----------
    seller_records : list[dict]
        Training records for this seller.
    reward_config : RewardConfig
        Reward configuration for computing reward values.
    sigma2 : float
        Working noise variance.
    d : int
        Feature dimension.

    Returns
    -------
    tuple[SellerState, list[float], list[np.ndarray]]
        (state, rewards, phis) for computing test metrics.
    """
    state = zero_state(d)
    rewards = []
    phis = []
    k = (d - 1) // 2

    for record in seller_records:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue

        try:
            reward = compute_reward(
                answered=record.get("answered", False),
                meeting_fixed=record.get("meeting_fixed", False),
                disposition=record.get("disposition", "UNKNOWN"),
                reward_config=reward_config,
            )
        except ValueError:
            continue

        hours = time_to_hours(call_start)
        try:
            phi = fourier(hours, k=k)
        except ValueError:
            continue

        state = apply_contribution(state, phi, reward, sigma2)
        rewards.append(reward)
        phis.append(phi)

    return state, rewards, phis


def _compute_meeting_rate(
    seller_records: list[dict],
) -> Optional[float]:
    """Compute the meeting rate for a seller's records.

    Parameters
    ----------
    seller_records : list[dict]
        Records for the seller.

    Returns
    -------
    float or None
        Fraction of records with meeting_fixed=True, or None if no records.
    """
    if not seller_records:
        return None
    meetings = sum(1 for r in seller_records if r.get("meeting_fixed", False))
    return meetings / len(seller_records)


# ── Historical slot computation ──────────────────────────────────────────────


def _extract_meeting_hours(training_records: list[dict]) -> list[float]:
    """Extract hours (0-23) when a seller had meetings in training data.

    Filters records where meeting_fixed=True and extracts the hour from
    call_start_time for each meeting.

    Parameters
    ----------
    training_records : list[dict]
        Training split records for a seller.

    Returns
    -------
    list[float]
        List of hour values (0-23) where meetings occurred.
    """
    meeting_hours = []
    for record in training_records:
        if not record.get("meeting_fixed", False):
            continue
        call_start = record.get("call_start_time")
        if call_start is None:
            continue
        _ist_hour = lambda dt: dt.astimezone(timezone(timedelta(hours=5, minutes=30))).hour
        meeting_hours.append(float(_ist_hour(call_start)))
    return meeting_hours


def _compute_historical_metrics(
    training_records: list[dict],
    predicted_best_hour: Optional[float],
) -> dict:
    """Compute historical slot comparison metrics for a seller.

    Analyzes training data (Apr-Sep 2026) to find the seller's historical
    best calling hour, compares it against the model's prediction, and
    computes various statistical metrics.

    Parameters
    ----------
    training_records : list[dict]
        Training split records for this seller.
    predicted_best_hour : float or None
        The model's predicted best hour.

    Returns
    -------
    dict
        Historical comparison metrics with keys:
        - historical_best_hour: hour with highest meeting rate in training
        - historical_best_meeting_rate: meeting rate at historical best hour
        - historical_closest_hour: hour closest to predicted that had meetings
        - historical_closest_meeting_rate: meeting rate at that closest hour
        - hours_from_historical_best: circular distance from prediction to historical best
        - hours_from_historical_closest: circular distance from prediction to closest historical hour
        - std_dev_from_historical_best: std of meeting hours around historical best
        - std_dev_from_predicted: std of meeting hours around predicted hour
        - historical_meeting_hours: list of all hours with meetings
        - model_agrees_with_history: True if predicted within 2h of historical best
    """
    result = {
        "historical_best_hour": None,
        "historical_best_meeting_rate": None,
        "historical_closest_hour": None,
        "historical_closest_meeting_rate": None,
        "hours_from_historical_best": None,
        "hours_from_historical_closest": None,
        "std_dev_from_historical_best": None,
        "std_dev_from_predicted": None,
        "historical_meeting_hours": [],
        "model_agrees_with_history": False,
    }

    # Extract meeting hours from training data
    meeting_hours = _extract_meeting_hours(training_records)

    if not meeting_hours:
        return result

    result["historical_meeting_hours"] = meeting_hours

    # Compute hourly attempts, meetings, and meeting rates
    hour_attempts: dict[int, int] = {}
    hour_counts: dict[int, int] = {}

    for record in training_records:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue

        hour = call_start.astimezone(
            timezone(timedelta(hours=5, minutes=30))
        ).hour

        hour_attempts[hour] = hour_attempts.get(hour, 0) + 1

        if record.get("meeting_fixed", False):
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

    hour_rates = {
        hour: hour_counts.get(hour, 0) / attempts
        for hour, attempts in hour_attempts.items()
        if attempts > 0
    }

    # Find historical best hour (highest observed meeting rate)
    best_hour = max(hour_rates, key=hour_rates.get)
    result["historical_best_hour"] = float(best_hour)
    result["historical_best_meeting_rate"] = _ensure_finite_float(
        hour_rates[best_hour]
    )

    # Find historical closest hour to predicted (if prediction exists)
    if predicted_best_hour is not None:
        # Find the hour in meeting_hours that is closest to predicted
        closest_hour = min(
            hour_counts.keys(),
            key=lambda h: circular_distance(float(h), predicted_best_hour)
        )
        closest_count = hour_counts[closest_hour]
        result["historical_closest_hour"] = float(closest_hour)
        result["historical_closest_meeting_rate"] = _ensure_finite_float(
            hour_rates[int(closest_hour)]
        ) if int(closest_hour) in hour_rates else None

        # Hours from historical best
        result["hours_from_historical_best"] = _ensure_finite_float(
            circular_distance(float(best_hour), predicted_best_hour)
        )

        # Hours from historical closest
        result["hours_from_historical_closest"] = _ensure_finite_float(
            circular_distance(float(closest_hour), predicted_best_hour)
        )

        # Model agrees with history if within 2 hours
        result["model_agrees_with_history"] = (
            circular_distance(float(best_hour), predicted_best_hour) <= 2.0
        )

    # Compute std dev of meeting hours around historical best
    meeting_hours_arr = np.array(meeting_hours, dtype=np.float64)

    # std_dev_from_historical_best: std of all meeting hours around the historical best hour
    # Use circular-aware std: map hours to unit circle, compute std of angular positions
    rms = lambda c: float(np.sqrt(np.mean([
        circular_distance(h, c) ** 2 for h in meeting_hours
    ])))

    result["std_dev_from_historical_best"] = rms(float(best_hour))
    result["std_dev_from_predicted"] = (
        rms(predicted_best_hour)
        if predicted_best_hour is not None
        else None
    )

    return result


# ── Accuracy metric helpers ──────────────────────────────────────────────────


def _ensure_finite_float(value: Optional[float], default: float = 0.0) -> float:
    """Ensure a float value is finite, replacing NaN/Inf with default."""
    if value is None:
        return default
    if not math.isfinite(value):
        return default
    return float(value)


def _ensure_finite_int(value: Optional[int], default: int = 0) -> int:
    """Ensure an int value is finite."""
    if value is None:
        return default
    return int(value)


def _compute_actual_best_hour(
    test_records: list[dict],
    min_attempts_per_hour: int = 2,
) -> tuple[Optional[float], dict[int, dict]]:
    """Compute the actual best hour from test data.

    Groups test attempts by hour (0-23), computes meeting rate per hour,
    and returns the hour with the highest meeting rate.

    Parameters
    ----------
    test_records : list[dict]
        Test split records for this seller.
    min_attempts_per_hour : int
        Minimum attempts per hour to be considered (default 2).

    Returns
    -------
    tuple[Optional[float], dict[int, dict]]
        (actual_best_hour, hour_stats) where hour_stats maps hour ->
        {'attempts': int, 'meetings': int, 'meeting_rate': float}.
    """
    hour_stats: dict[int, dict] = {}
    for h in range(24):
        hour_stats[h] = {"attempts": 0, "meetings": 0, "meeting_rate": 0.0}

    for record in test_records:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue
        _ist_hour = lambda dt: dt.astimezone(timezone(timedelta(hours=5, minutes=30))).hour
        hour = _ist_hour(call_start)
        hour_stats[hour]["attempts"] += 1
        if record.get("meeting_fixed", False):
            hour_stats[hour]["meetings"] += 1

    # Compute meeting rates
    for h in range(24):
        stats = hour_stats[h]
        if stats["attempts"] > 0:
            stats["meeting_rate"] = stats["meetings"] / stats["attempts"]

    # Find best hour among those with >= min_attempts_per_hour
    best_hour = None
    best_rate = -1.0
    for h in range(24):
        stats = hour_stats[h]
        if stats["attempts"] >= min_attempts_per_hour:
            if stats["meeting_rate"] > best_rate:
                best_rate = stats["meeting_rate"]
                best_hour = float(h)

    return best_hour, hour_stats


# ── Prediction ───────────────────────────────────────────────────────────────


def predict_for_seller(
    seller_id: str,
    test_records: list[dict],
    bundle_metadata: dict,
    bundle_arrays: dict,
) -> dict:
    """Generate predictions for a single seller (cold start / no training data).

    Builds a zero state, selects the segment prior from the bundle, computes
    the posterior, and scores all 24 hours to find the best and secondary
    predicted hours.

    Parameters
    ----------
    seller_id : str
        Seller identifier.
    test_records : list[dict]
        Test split records for this seller.
    bundle_metadata : dict
        Bundle metadata (includes k, d, sigma2, segment_keys, reward_params).
    bundle_arrays : dict
        Bundle arrays (segment priors, global prior, etc.).

    Returns
    -------
    dict
        Prediction result with keys:
        seller_id, segment_key, n_seller_attempts, n_test_attempts,
        predicted_best_hour, predicted_secondary_hour, expected_reward,
        prior_weight, test_meeting_rate, test_answer_rate, test_meeting_count,
        test_total_attempts, predicted_hour_in_support, actual_best_hour,
        hour_prediction_error, meeting_rate_at_predicted_hour,
        meeting_rate_at_best_hour.
    """
    k = int(bundle_metadata.get("k", 4))
    d = int(bundle_metadata.get("d", 2 * k + 1))
    sigma2 = float(bundle_metadata.get("sigma2", 0.06))
    reward_config = _reward_config_from_metadata(bundle_metadata)

    segment_keys = bundle_metadata.get("segment_keys", [])
    segment_mu0 = bundle_arrays.get("segment_mu0")
    segment_sigma0 = bundle_arrays.get("segment_sigma0")
    segment_lambda0 = bundle_arrays.get("segment_lambda0")
    segment_eta0 = bundle_arrays.get("segment_eta0")

    # Determine seller's segment from test records
    segment_key = GLOBAL_SEGMENT_KEY
    for record in test_records:
        seg = record.get("segment", GLOBAL_SEGMENT_KEY)
        if seg != GLOBAL_SEGMENT_KEY:
            segment_key = seg
            break

    n_test_attempts = len(test_records)

    if n_test_attempts == 0:
        return {
            "seller_id": seller_id,
            "segment_key": segment_key,
            "n_seller_attempts": 0,
            "n_test_attempts": 0,
            "predicted_best_hour": None,
            "predicted_secondary_hour": None,
            "expected_reward": None,
            "prior_weight": 1.0,
            "test_meeting_rate": None,
            "test_answer_rate": 0.0,
            "test_meeting_count": 0,
            "test_total_attempts": 0,
            "predicted_hour_in_support": False,
            "actual_best_hour": None,
            "hour_prediction_error": None,
            "meeting_rate_at_predicted_hour": None,
            "meeting_rate_at_best_hour": None,
        }

    # Compute test answer rate and meeting count
    test_answered = sum(1 for r in test_records if r.get("answered", False))
    test_meeting_count = sum(1 for r in test_records if r.get("meeting_fixed", False))
    test_answer_rate = test_answered / n_test_attempts
    test_meeting_rate = test_meeting_count / n_test_attempts

    # Compute actual best hour from test data
    actual_best_hour, hour_stats = _compute_actual_best_hour(test_records)

    # Use segment prior (cold start since no training data provided here)
    segment_prior = _find_segment_prior(
        segment_key, bundle_metadata, segment_keys,
        segment_mu0, segment_sigma0, segment_lambda0, segment_eta0,
    )

    if segment_prior is None:
        segment_prior = _make_global_prior(bundle_arrays, d)

    # Zero state (cold start)
    state = zero_state(d)

    # Compute posterior
    posterior = compute_posterior(state, segment_prior, sigma2)

    if hasattr(posterior, "is_cold_start") and posterior.is_cold_start:
        prior_weight = 1.0
    else:
        prior_weight = float(posterior.prior_weight)

    # Score all 24 hours to find best and secondary
    hours = np.array([float(h) for h in range(24)], dtype=np.float64)
    phi_matrix = fourier(hours, k=k)
    expected_rewards = phi_matrix @ posterior.mu

    best_idx = int(np.argmax(expected_rewards))
    predicted_best_hour = float(hours[best_idx])

    secondary_idx = -1
    max_secondary = float("-inf")
    for i in range(len(hours)):
        if i != best_idx and expected_rewards[i] > max_secondary:
            max_secondary = float(expected_rewards[i])
            secondary_idx = i

    predicted_secondary_hour = float(hours[secondary_idx]) if secondary_idx >= 0 else None
    expected_reward = float(expected_rewards[best_idx])

    # Accuracy metrics
    predicted_hour_in_support = (
        8 <= predicted_best_hour <= 18
    )

    meeting_rate_at_predicted_hour = None
    meeting_rate_at_best_hour = None
    hour_prediction_error = None

    if actual_best_hour is not None:
        meeting_rate_at_best_hour = _ensure_finite_float(
            hour_stats[int(actual_best_hour)]["meeting_rate"]
        )
        hour_prediction_error = _ensure_finite_float(
            abs(predicted_best_hour - actual_best_hour)
        )

    if predicted_best_hour is not None:
        pred_hour_int = int(round(predicted_best_hour))
        if 0 <= pred_hour_int <= 23:
            meeting_rate_at_predicted_hour = _ensure_finite_float(
                hour_stats[pred_hour_int]["meeting_rate"]
            )

    return {
        "seller_id": seller_id,
        "segment_key": segment_key,
        "n_seller_attempts": 0,
        "n_test_attempts": n_test_attempts,
        "predicted_best_hour": predicted_best_hour,
        "predicted_secondary_hour": predicted_secondary_hour,
        "expected_reward": expected_reward,
        "prior_weight": prior_weight,
        "test_meeting_rate": test_meeting_rate,
        "test_answer_rate": _ensure_finite_float(test_answer_rate),
        "test_meeting_count": _ensure_finite_int(test_meeting_count),
        "test_total_attempts": _ensure_finite_int(n_test_attempts),
        "predicted_hour_in_support": predicted_hour_in_support,
        "actual_best_hour": actual_best_hour,
        "hour_prediction_error": hour_prediction_error,
        "meeting_rate_at_predicted_hour": meeting_rate_at_predicted_hour,
        "meeting_rate_at_best_hour": meeting_rate_at_best_hour,
    }


def predict_for_seller_with_training(
    seller_id: str,
    test_records: list[dict],
    training_records: list[dict],
    bundle_metadata: dict,
    bundle_arrays: dict,
) -> dict:
    """Generate predictions for a single seller using training data.

    Builds the seller state from training records, selects the segment prior,
    computes the posterior, scores all 24 hours, and computes historical
    slot comparison metrics.

    Parameters
    ----------
    seller_id : str
        Seller identifier.
    test_records : list[dict]
        Test split records for this seller.
    training_records : list[dict]
        Training split records (prior_fit + warmup + validation).
    bundle_metadata : dict
        Bundle metadata (includes k, d, sigma2, segment_keys, reward_params).
    bundle_arrays : dict
        Bundle arrays (segment priors, global prior, etc.).

    Returns
    -------
    dict
        Prediction result with full metrics including accuracy, historical
        slot comparison (historical_best_hour, model_agrees_with_history, etc.).
    """
    k = int(bundle_metadata.get("k", 4))
    d = int(bundle_metadata.get("d", 2 * k + 1))
    sigma2 = float(bundle_metadata.get("sigma2", 0.06))
    reward_config = _reward_config_from_metadata(bundle_metadata)

    segment_keys = bundle_metadata.get("segment_keys", [])
    segment_mu0 = bundle_arrays.get("segment_mu0")
    segment_sigma0 = bundle_arrays.get("segment_sigma0")
    segment_lambda0 = bundle_arrays.get("segment_lambda0")
    segment_eta0 = bundle_arrays.get("segment_eta0")

    # Determine seller's segment
    segment_key = GLOBAL_SEGMENT_KEY
    for record in test_records + training_records:
        seg = record.get("segment", GLOBAL_SEGMENT_KEY)
        if seg != GLOBAL_SEGMENT_KEY:
            segment_key = seg
            break

    n_test_attempts = len(test_records)

    # Compute test answer rate and meeting count
    test_answered = sum(1 for r in test_records if r.get("answered", False))
    test_meeting_count = sum(1 for r in test_records if r.get("meeting_fixed", False))
    test_answer_rate = test_answered / n_test_attempts if n_test_attempts > 0 else 0.0
    test_meeting_rate = test_meeting_count / n_test_attempts if n_test_attempts > 0 else 0.0

    # Compute actual best hour from test data
    actual_best_hour, hour_stats = _compute_actual_best_hour(test_records)

    # Build seller state from training records
    state, rewards, phis = _build_seller_state(
        training_records, reward_config, sigma2, d,
    )
    n_seller_attempts = state.n

    # Find segment prior
    segment_prior = _find_segment_prior(
        segment_key, bundle_metadata, segment_keys,
        segment_mu0, segment_sigma0, segment_lambda0, segment_eta0,
    )

    if segment_prior is None:
        segment_prior = _make_global_prior(bundle_arrays, d)

    # Compute posterior
    posterior = compute_posterior(state, segment_prior, sigma2)

    if hasattr(posterior, "is_cold_start") and posterior.is_cold_start:
        prior_weight = 1.0
    else:
        prior_weight = float(posterior.prior_weight)

    # Score all 24 hours
    hours = np.array([float(h) for h in range(24)], dtype=np.float64)
    phi_matrix = fourier(hours, k=k)
    expected_rewards = phi_matrix @ posterior.mu

    best_idx = int(np.argmax(expected_rewards))
    predicted_best_hour = float(hours[best_idx])

    secondary_idx = -1
    max_secondary = float("-inf")
    for i in range(len(hours)):
        if i != best_idx and expected_rewards[i] > max_secondary:
            max_secondary = float(expected_rewards[i])
            secondary_idx = i

    predicted_secondary_hour = float(hours[secondary_idx]) if secondary_idx >= 0 else None
    expected_reward = float(expected_rewards[best_idx])

    # Accuracy metrics
    predicted_hour_in_support = (
        8 <= predicted_best_hour <= 18
    )

    meeting_rate_at_predicted_hour = None
    meeting_rate_at_best_hour = None
    hour_prediction_error = None

    if actual_best_hour is not None:
        meeting_rate_at_best_hour = _ensure_finite_float(
            hour_stats[int(actual_best_hour)]["meeting_rate"]
        )
        hour_prediction_error = _ensure_finite_float(
            abs(predicted_best_hour - actual_best_hour)
        )

    if predicted_best_hour is not None:
        pred_hour_int = int(round(predicted_best_hour))
        if 0 <= pred_hour_int <= 23:
            meeting_rate_at_predicted_hour = _ensure_finite_float(
                hour_stats[pred_hour_int]["meeting_rate"]
            )

    # Historical slot comparison metrics
    historical_metrics = _compute_historical_metrics(
        training_records, predicted_best_hour,
    )

    return {
        "seller_id": seller_id,
        "segment_key": segment_key,
        "n_seller_attempts": n_seller_attempts,
        "n_test_attempts": n_test_attempts,
        "predicted_best_hour": predicted_best_hour,
        "predicted_secondary_hour": predicted_secondary_hour,
        "expected_reward": expected_reward,
        "prior_weight": prior_weight,
        "test_meeting_rate": test_meeting_rate,
        "test_answer_rate": _ensure_finite_float(test_answer_rate),
        "test_meeting_count": _ensure_finite_int(test_meeting_count),
        "test_total_attempts": _ensure_finite_int(n_test_attempts),
        "predicted_hour_in_support": predicted_hour_in_support,
        "actual_best_hour": actual_best_hour,
        "hour_prediction_error": hour_prediction_error,
        "meeting_rate_at_predicted_hour": meeting_rate_at_predicted_hour,
        "meeting_rate_at_best_hour": meeting_rate_at_best_hour,
        # Historical slot comparison
        "historical_best_hour": historical_metrics["historical_best_hour"],
        "historical_best_meeting_rate": historical_metrics["historical_best_meeting_rate"],
        "historical_closest_hour": historical_metrics["historical_closest_hour"],
        "historical_closest_meeting_rate": historical_metrics["historical_closest_meeting_rate"],
        "hours_from_historical_best": historical_metrics["hours_from_historical_best"],
        "hours_from_historical_closest": historical_metrics["hours_from_historical_closest"],
        "std_dev_from_historical_best": historical_metrics["std_dev_from_historical_best"],
        "std_dev_from_predicted": historical_metrics["std_dev_from_predicted"],
        "historical_meeting_hours": historical_metrics["historical_meeting_hours"],
        "model_agrees_with_history": historical_metrics["model_agrees_with_history"],
    }


# ── Pure inference for time slots ────────────────────────────────────────────


def predict_for_time_slots(
    candidate_hours: list[float],
    state: SellerState,
    segment_prior: Prior,
    sigma2: float,
    k: int,
    support_bins: Optional[dict] = None,
    segment_key: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """Pure inference: score candidate time slots and return the best one.

    This function has **no side effects** — it does not load files, access
    the filesystem, print, or log. It expects all inputs to be pre-prepared
    by the caller (typically a root-level script such as ``inference_for_seller.py``
    that handles CSV loading, state construction, and model bundle loading).

    Parameters
    ----------
    candidate_hours : list[float]
        List of hours (0-23) to score.
    state : SellerState
        Seller sufficient statistics (built from training data or zero for cold start).
    segment_prior : Prior
        Segment prior for the seller (from bundle or global fallback).
    sigma2 : float
        Noise variance (from bundle metadata).
    k : int
        Fourier feature dimension parameter (from bundle metadata).
    support_bins : dict or None
        Support bins from bundle metadata (optional). Used to mark which
        hours fall within the seller's supported time range.
    segment_key : str or None
        Segment key for support bin lookup (optional).

    Returns
    -------
    tuple[list[dict], dict]
        ``(candidate_slots, best_slot)`` where:
        - ``candidate_slots``: list of dicts, each with keys
          ``hour``, ``expected_reward``, ``latent_std``, ``predictive_std``,
          ``is_supported``.
        - ``best_slot``: the dict with the highest expected reward, or ``None``
          if ``candidate_hours`` is empty.
    """
    if not candidate_hours:
        return [], None

    hours_arr = np.array(candidate_hours, dtype=np.float64)
    phi_matrix = fourier(hours_arr, k=k)
    scores = score_candidates(phi_matrix, state, segment_prior, sigma2)

    candidate_slots = []
    for i, hour in enumerate(candidate_hours):
        expected_reward = float(scores[i, 0])
        latent_std = float(scores[i, 1])
        predictive_std = float(scores[i, 2])

        is_supported = False
        if support_bins and segment_key:
            is_supported = bool(
                support_bins.get(segment_key, {}).get(str(hour), False)
                or support_bins.get("__all__", {}).get(str(hour), False)
            )

        candidate_slots.append({
            "hour": round(hour, 2),
            "expected_reward": round(expected_reward, 6),
            "latent_std": round(latent_std, 6),
            "predictive_std": round(predictive_std, 6),
            "is_supported": is_supported,
        })

    best_idx = int(np.argmax(scores[:, 0]))
    best_slot = candidate_slots[best_idx] if candidate_slots else None

    return candidate_slots, best_slot


def _compute_hour_distribution_buckets(
    hours_from_historical_best: list[float],
) -> dict:
    """Compute the distribution of hour differences vs historical best.

    Parameters
    ----------
    hours_from_historical_best : list[float]
        List of circular distances from predicted to historical best.

    Returns
    -------
    dict
        Distribution buckets with percentages.
    """
    n = len(hours_from_historical_best)
    if n == 0:
        return {
            "within_1h": 0.0,
            "within_2h": 0.0,
            "within_4h": 0.0,
            "within_6h": 0.0,
            "beyond_6h": 0.0,
        }

    within_1h = sum(1 for h in hours_from_historical_best if h <= 1.0)
    within_2h = sum(1 for h in hours_from_historical_best if h <= 2.0)
    within_4h = sum(1 for h in hours_from_historical_best if h <= 4.0)
    within_6h = sum(1 for h in hours_from_historical_best if h <= 6.0)
    beyond_6h = sum(1 for h in hours_from_historical_best if h > 6.0)

    return {
        "within_1h": _ensure_finite_float(within_1h / n),
        "within_2h": _ensure_finite_float(within_2h / n),
        "within_4h": _ensure_finite_float(within_4h / n),
        "within_6h": _ensure_finite_float(within_6h / n),
        "beyond_6h": _ensure_finite_float(beyond_6h / n),
    }


def generate_inference_report(
    data_dir: str,
    bundle_path: str,
    output_path: str,
    n_sellers: int = 100,
) -> dict:
    """Generate inference report: full pipeline from CSV loading to JSON output.

    This function orchestrates the entire workflow — loading CSV files,
    normalizing records, creating chronological splits, loading the model
    bundle, generating per-seller predictions, and saving the report.

    Steps:
    1. Load and normalize CSV data (attempts + sellers)
    2. Create chronological splits (prior_fit, warmup, validation, test)
    3. Extract test split (Sep-Oct 2026) and training data
    4. Load model bundle (metadata + arrays)
    5. Group records by seller and sort by priority
    6. For each seller (up to n_sellers):
       - Build seller state from training data
       - Generate predictions with accuracy metrics
       - Compute historical slot comparison metrics
    7. Compute summary statistics (accuracy, meeting rate, historical comparison)
    8. Save report JSON to output_path

    Parameters
    ----------
    data_dir : str
        Directory containing CSV data files (attempts and sellers).
    bundle_path : str
        Path to the model bundle directory.
    output_path : str
        Path to save the inference report JSON.
    n_sellers : int
        Maximum number of sellers to generate predictions for.

    Returns
    -------
    dict
        Inference report with per-seller predictions and summary statistics.
    """
    logger.info("Starting inference report generation")
    logger.info("  data_dir=%s, bundle_path=%s, output_path=%s",
                data_dir, bundle_path, output_path)

    # Step 1: Load and normalize data
    logger.info("Step 1: Loading and normalizing data")
    attempts_path, sellers_path = None, None

    for filename in sorted(os.listdir(data_dir)):
        if filename.lower().endswith(".csv"):
            if "seller" in filename.lower():
                sellers_path = os.path.join(data_dir, filename)
            elif "attempt" in filename.lower() or "call" in filename.lower():
                attempts_path = os.path.join(data_dir, filename)

    if not attempts_path or not sellers_path:
        raise FileNotFoundError(
            f"Could not find attempts and sellers CSV files in {data_dir}"
        )

    raw_attempts, attempts_report = load_attempts_csv(attempts_path)
    sellers = load_sellers_csv(sellers_path)
    logger.info("Loaded %d attempts, %d sellers",
                len(raw_attempts), len(sellers))

    normalized_data = []
    for raw in raw_attempts:
        try:
            record = normalize_outcome(raw, sellers)
            normalized_data.append(record)
        except ValueError:
            continue

    logger.info("Normalized %d records", len(normalized_data))

    # Step 2: Create chronological splits
    logger.info("Step 2: Creating chronological splits")
    splits = create_chronological_splits(normalized_data)
    logger.info("Split sizes: %s",
                {k: len(v) for k, v in splits.items()})

    # Step 3: Extract splits
    test_records = splits.get("test", [])
    training_records = (
        splits.get("warmup", []) +
        splits.get("validation", [])
    )
    logger.info("Test: %d, Training: %d", len(test_records), len(training_records))

    # Step 4: Load model bundle
    logger.info("Step 3: Loading model bundle")
    bundle_metadata, bundle_arrays = load_model_bundle(bundle_path)

    # Step 5: Group records by seller
    logger.info("Step 4: Generating predictions for up to %d sellers", n_sellers)
    seller_test_records: dict[str, list[dict]] = {}
    for record in test_records:
        sid = record.get("seller_id", "")
        if sid:
            if sid not in seller_test_records:
                seller_test_records[sid] = []
            seller_test_records[sid].append(record)

    seller_training_records: dict[str, list[dict]] = {}
    for record in training_records:
        sid = record.get("seller_id", "")
        if sid:
            if sid not in seller_training_records:
                seller_training_records[sid] = []
            seller_training_records[sid].append(record)

    # Identify sellers with historical meetings for prioritization
    sellers_with_historical_meetings = set()
    for sid, train_recs in seller_training_records.items():
        if any(r.get("meeting_fixed", False) for r in train_recs):
            sellers_with_historical_meetings.add(sid)

    # Sort by test record count, prioritizing sellers with historical meetings
    def seller_sort_key(sid):
        has_history = 1 if sid in sellers_with_historical_meetings else 0
        return (has_history, len(seller_test_records[sid]))

    sorted_sellers = sorted(
        seller_test_records.keys(),
        key=seller_sort_key,
        reverse=True,
    )

    # Step 6: Generate predictions
    predictions = []
    errors = 0

    for i, seller_id in enumerate(sorted_sellers[:n_sellers]):
        try:
            test_recs = seller_test_records[seller_id]
            train_recs = seller_training_records.get(seller_id, [])

            result = predict_for_seller_with_training(
                seller_id=seller_id,
                test_records=test_recs,
                training_records=train_recs,
                bundle_metadata=bundle_metadata,
                bundle_arrays=bundle_arrays,
            )
            predictions.append(result)

            if i % 20 == 0:
                logger.info("Processed %d/%d sellers", i + 1, min(n_sellers, len(sorted_sellers)))

        except Exception as exc:
            errors += 1
            logger.warning("Error processing seller %s: %s", seller_id, exc)
            continue

    # Step 7: Compute summary statistics
    logger.info("Step 5: Computing summary statistics")
    all_best_hours = [
        p["predicted_best_hour"] for p in predictions
        if p["predicted_best_hour"] is not None
    ]
    all_rewards = [
        p["expected_reward"] for p in predictions
        if p["expected_reward"] is not None
    ]
    all_prior_weights = [
        p["prior_weight"] for p in predictions
        if p["prior_weight"] is not None
    ]
    sellers_with_secondary = sum(
        1 for p in predictions if p["predicted_secondary_hour"] is not None
    )

    segment_dist: dict[str, int] = {}
    for p in predictions:
        seg = p.get("segment_key", "unknown")
        segment_dist[seg] = segment_dist.get(seg, 0) + 1

    # Accuracy summary metrics
    all_test_answer_rates = [
        p["test_answer_rate"] for p in predictions
        if p["test_answer_rate"] is not None
    ]
    all_test_meeting_rates = [
        p["test_meeting_rate"] for p in predictions
        if p["test_meeting_rate"] is not None
    ]
    sellers_with_meetings = sum(
        1 for p in predictions
        if p.get("test_meeting_count", 0) is not None and p["test_meeting_count"] > 0
    )

    # Hour prediction accuracy
    hour_errors = [
        p["hour_prediction_error"] for p in predictions
        if p["hour_prediction_error"] is not None
    ]
    mean_error = float(np.mean(hour_errors)) if hour_errors else 0.0
    median_error = float(np.median(hour_errors)) if hour_errors else 0.0
    error_within_2h = (
        sum(1 for e in hour_errors if e <= 2.0) / len(hour_errors)
        if hour_errors else 0.0
    )
    error_within_4h = (
        sum(1 for e in hour_errors if e <= 4.0) / len(hour_errors)
        if hour_errors else 0.0
    )

    # Meeting rate comparison
    rate_comparisons = [
        (p["meeting_rate_at_predicted_hour"], p["meeting_rate_at_best_hour"])
        for p in predictions
        if (p["meeting_rate_at_predicted_hour"] is not None
            and p["meeting_rate_at_best_hour"] is not None)
    ]
    avg_meeting_rate_at_predicted = (
        sum(r[0] for r in rate_comparisons) / len(rate_comparisons)
        if rate_comparisons else 0.0
    )
    avg_meeting_rate_at_best = (
        sum(r[1] for r in rate_comparisons) / len(rate_comparisons)
        if rate_comparisons else 0.0
    )
    predicted_is_better = sum(
        1 for r in rate_comparisons if r[0] > r[1]
    )
    predicted_is_worse = sum(
        1 for r in rate_comparisons if r[0] < r[1]
    )

    # ── Historical comparison summary metrics ──────────────────────────────

    sellers_with_historical = [
        p for p in predictions
        if p.get("historical_best_hour") is not None
    ]
    sellers_with_historical_count = len(sellers_with_historical)

    hours_from_hist_best = [
        p["hours_from_historical_best"]
        for p in sellers_with_historical
        if p.get("hours_from_historical_best") is not None
    ]
    hours_from_hist_closest = [
        p["hours_from_historical_closest"]
        for p in sellers_with_historical
        if p.get("hours_from_historical_closest") is not None
    ]
    std_dev_from_hist_best = [
        p["std_dev_from_historical_best"]
        for p in sellers_with_historical
        if p.get("std_dev_from_historical_best") is not None
    ]
    std_dev_from_predicted = [
        p["std_dev_from_predicted"]
        for p in sellers_with_historical
        if p.get("std_dev_from_predicted") is not None
    ]

    model_agrees_count = sum(
        1 for p in predictions if p.get("model_agrees_with_history", False)
    )
    model_agreement_rate = (
        model_agrees_count / len(predictions) if predictions else 0.0
    )

    hour_distribution = _compute_hour_distribution_buckets(hours_from_hist_best)

    summary = {
        "avg_predicted_best_hour": float(np.mean(all_best_hours)) if all_best_hours else 0.0,
        "avg_expected_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "avg_prior_weight": float(np.mean(all_prior_weights)) if all_prior_weights else 0.0,
        "sellers_with_secondary": sellers_with_secondary,
        "segment_distribution": segment_dist,
        # Accuracy summary
        "avg_test_answer_rate": _ensure_finite_float(
            float(np.mean(all_test_answer_rates)) if all_test_answer_rates else 0.0
        ),
        "avg_test_meeting_rate": _ensure_finite_float(
            float(np.mean(all_test_meeting_rates)) if all_test_meeting_rates else 0.0
        ),
        "sellers_with_meetings": sellers_with_meetings,
        "hour_prediction_accuracy": {
            "mean_error_hours": _ensure_finite_float(mean_error),
            "median_error_hours": _ensure_finite_float(median_error),
            "error_within_2h": _ensure_finite_float(error_within_2h),
            "error_within_4h": _ensure_finite_float(error_within_4h),
        },
        "meeting_rate_comparison": {
            "avg_meeting_rate_at_predicted": _ensure_finite_float(avg_meeting_rate_at_predicted),
            "avg_meeting_rate_at_best": _ensure_finite_float(avg_meeting_rate_at_best),
            "predicted_is_better": predicted_is_better,
            "predicted_is_worse": predicted_is_worse,
        },
        # Historical comparison summary
        "historical_comparison": {
            "sellers_with_historical_meetings": sellers_with_historical_count,
            "avg_hours_from_historical_best": _ensure_finite_float(
                float(np.mean(hours_from_hist_best)) if hours_from_hist_best else 0.0
            ),
            "avg_hours_from_historical_closest": _ensure_finite_float(
                float(np.mean(hours_from_hist_closest)) if hours_from_hist_closest else 0.0
            ),
            "avg_std_dev_from_historical_best": _ensure_finite_float(
                float(np.mean(std_dev_from_hist_best)) if std_dev_from_hist_best else 0.0
            ),
            "avg_std_dev_from_predicted": _ensure_finite_float(
                float(np.mean(std_dev_from_predicted)) if std_dev_from_predicted else 0.0
            ),
            "model_agrees_with_history_count": model_agrees_count,
            "model_agreement_rate": _ensure_finite_float(model_agreement_rate),
            "hour_distribution_vs_history": hour_distribution,
        },
    }

    # Build report
    report = {
        "n_sellers_tested": len(predictions),
        "n_test_records": len(test_records),
        "bundle_id": bundle_metadata.get("bundle_id", ""),
        "predictions": predictions,
        "summary": summary,
    }

    # Step 8: Save report
    logger.info("Step 6: Saving report to %s", output_path)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Report saved: %d sellers, %d test records, errors=%d",
                len(predictions), len(test_records), errors)

    return report
