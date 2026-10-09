"""Model training pipeline and bundle management for the Best Time to Call system.

Implements WP2 (Data/training agent) model training workflow:
- TRAIN-06: Chronological data splits via btc.data.normalization
- TRAIN-07: Bundle format (JSON metadata + non-pickled numeric arrays)
- TRAIN-08: Model compatibility and state namespace management
- TRAIN-09: Segment keys as canonical JSON arrays

This module provides the full training pipeline (run_training_pipeline),
bundle I/O (save_bundle, load_bundle), compatibility checking
(check_compatibility), and support bin computation (compute_segment_support).

The training workflow wraps :mod:`btc.model.priors` for hierarchical prior
fitting and uses :mod:`btc.data.adapters` and :mod:`btc.data.normalization`
for data loading, normalization, and splitting.

Modules
-------
run_training_pipeline : Full training pipeline entry point.
save_bundle : TRAIN-07 bundle serialization.
load_bundle : TRAIN-07 bundle deserialization with validation.
check_compatibility : TRAIN-08 compatibility checking.
compute_segment_support : TRAIN-05 support bin computation.
generate_bundle_id : Unique bundle ID generation.
generate_compatibility_id : TRAIN-08 compatibility ID generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from btc.config import Config, ModelConfig, RewardConfig, load_config
from btc.data.adapters import load_attempts_csv, load_sellers_csv, join_attempts_sellers
from btc.data.normalization import (
    normalize_batch,
    create_chronological_splits,
    compute_segment_statistics,
    compute_support_bins,
    validate_split_integrity,
)
from btc.features.fourier import feature_dim
from btc.model.priors import (
    _is_spd,
    fit_hierarchical_priors,
    HierarchicalPriorResult,
    SegmentPrior,
)
from btc.model.stats import Prior

logger = logging.getLogger(__name__)

# Canonical key for the global (root) prior. It has no entry in segment_stats,
# so it must be exempt from the eligibility filter. Adjust if yours differs.
GLOBAL_SEGMENT_KEY = "[]"


# ---------------------------------------------------------------------------
# PriorBundle data structure (TRAIN-07)
# ---------------------------------------------------------------------------


@dataclass
class PriorBundle:
    """TRAIN-07: Bundle containing fitted priors and metadata.

    Stores the fitted hierarchical priors as non-pickled numeric arrays
    alongside JSON-serializable metadata. The bundle is saved as:

        output_dir/
            metadata.json    # Bundle metadata (JSON)
            arrays.npz       # Numeric arrays (NumPy NPZ, non-pickled)
            checksums.txt    # SHA-256 checksums of each file

    Attributes
    ----------
    bundle_id : str
        Unique bundle identifier (UUID v4).
    compatibility_id : str
        Compatibility ID for state namespace matching (TRAIN-08).
    version : str
        Bundle version string.
    created_at : str
        ISO 8601 creation timestamp (UTC).
    k : int
        Fourier basis order.
    d : int
        Feature dimension (2*k + 1).
    sigma2 : float
        Working noise variance.
    reward_params : dict
        Reward configuration parameters.
    state_history_start : str
        Inclusive start of state-history interval (ISO date).
    state_history_end : str
        Exclusive end of state-history interval (ISO date).
    normalization : dict
        Normalization parameters (mean/std per feature).
    segment_keys : list[str]
        Canonical segment keys as JSON arrays (TRAIN-09).
    global_mu0 : np.ndarray, shape (d,)
        Global prior mean vector.
    global_sigma0 : np.ndarray, shape (d, d)
        Global prior covariance matrix.
    global_lambda0 : np.ndarray, shape (d, d)
        Global prior precision matrix.
    global_eta0 : np.ndarray, shape (d,)
        Global prior natural parameter.
    segment_mu0 : np.ndarray, shape (n_segments, d)
        Per-segment prior mean vectors.
    segment_sigma0 : np.ndarray, shape (n_segments, d, d)
        Per-segment prior covariance matrices.
    segment_lambda0 : np.ndarray, shape (n_segments, d, d)
        Per-segment prior precision matrices.
    segment_eta0 : np.ndarray, shape (n_segments, d,)
        Per-segment prior natural parameters.
    segment_n_obs : np.ndarray, shape (n_segments,)
        Observation counts per segment.
    segment_n_sellers : np.ndarray, shape (n_segments,)
        Seller counts per segment.
    segment_is_shrunk : np.ndarray, shape (n_segments,)
        Shrinkage flags per segment (0 or 1).
    segment_parent_keys : list[str | None]
        Parent segment keys per segment.
    segment_hierarchy : dict[str, str | None]
        Segment key → parent key mapping.
    data_checksum : str
        SHA-256 of input data for reproducibility.
    split_integrity : dict
        Validation report from split integrity check.
    segment_statistics : dict
        Segment eligibility statistics.
    support_bins : dict
        Support bin mask per segment.
    hyperparameter_report : dict
        Grid search results per segment.
    """

    bundle_id: str = ""
    compatibility_id: str = ""
    version: str = "1.0.0"
    format_version: int = 1
    created_at: str = ""
    k: int = 4
    d: int = 0
    sigma2: float = 0.06
    reward_params: dict = field(default_factory=dict)
    state_history_start: str = ""
    state_history_end: str = ""
    normalization: dict = field(default_factory=dict)
    feature_ordering: list[str] = field(default_factory=lambda: ["intercept"] + [f"sin_{i}" for i in range(1, 5)] + [f"cos_{i}" for i in range(1, 5)])
    segment_keys: list[str] = field(default_factory=list)
    global_mu0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    global_mean: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    prior_alpha: float = 0.1
    global_sigma0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    global_lambda0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    global_eta0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_mu0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_sigma0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_lambda0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_eta0: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_n_obs: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_n_sellers: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_is_shrunk: npt.NDArray[np.float64] = None  # type: ignore[assignment]
    segment_parent_keys: list[Optional[str]] = field(default_factory=list)
    segment_hierarchy: dict[str, Optional[str]] = field(default_factory=dict)
    data_checksum: str = ""
    split_integrity: dict = field(default_factory=dict)
    segment_statistics: dict = field(default_factory=dict)
    support_bins: dict = field(default_factory=dict)
    hyperparameter_report: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate bundle consistency."""
        if self.d == 0 and self.k > 0:
            self.d = feature_dim(self.k)
        elif self.d > 0 and self.k == 0:
            self.k = (self.d - 1) // 2

        if self.created_at == "":
            self.created_at = datetime.now(timezone.utc).isoformat()

        if self.bundle_id == "":
            self.bundle_id = generate_bundle_id()

        if self.compatibility_id == "":
            self.compatibility_id = generate_compatibility_id(
                k=self.k,
                sigma2=self.sigma2,
                reward_params=self.reward_params,
                state_history_start=self.state_history_start,
            )

    def to_metadata_dict(self) -> dict:
        """Convert bundle to a serializable metadata dictionary.

        Returns
        -------
        dict
            Metadata dict with all non-array fields.
        """
        metadata = {
            "bundle_id": self.bundle_id,
            "compatibility_id": self.compatibility_id,
            "version": self.version,
            "format_version": self.format_version,
            "created_at": self.created_at,
            "k": self.k,
            "d": self.d,
            "sigma2": self.sigma2,
            "reward_params": self.reward_params,
            "state_history_start": self.state_history_start,
            "state_history_end": self.state_history_end,
            "normalization": self.normalization,
            "feature_ordering": self.feature_ordering,
            "global_mean": self.global_mean.tolist() if self.global_mean is not None else [],
            "prior_alpha": self.prior_alpha,
            "segment_keys": self.segment_keys,
            "segment_parent_keys": self.segment_parent_keys,
            "segment_hierarchy": self.segment_hierarchy,
            "data_checksum": self.data_checksum,
            "split_integrity": self.split_integrity,
            "segment_statistics": self.segment_statistics,
            "support_bins": self.support_bins,
            "hyperparameter_report": self.hyperparameter_report,
        }
        return metadata

    @classmethod
    def from_metadata_dict(cls, metadata: dict, arrays: dict[str, npt.NDArray[np.float64]]) -> PriorBundle:
        """Reconstruct a PriorBundle from metadata dict and array dict.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary (output of to_metadata_dict).
        arrays : dict[str, np.ndarray]
            Array dictionary mapping array names to numpy arrays.

        Returns
        -------
        PriorBundle
            Reconstructed bundle.
        """
        bundle = cls(
            bundle_id=metadata.get("bundle_id", ""),
            compatibility_id=metadata.get("compatibility_id", ""),
            version=metadata.get("version", "1.0.0"),
            format_version=metadata.get("format_version", 1),
            created_at=metadata.get("created_at", ""),
            k=metadata.get("k", 4),
            d=metadata.get("d", 0),
            sigma2=metadata.get("sigma2", 0.06),
            reward_params=metadata.get("reward_params", {}),
            state_history_start=metadata.get("state_history_start", ""),
            state_history_end=metadata.get("state_history_end", ""),
            normalization=metadata.get("normalization", {}),
            feature_ordering=metadata.get("feature_ordering", ["intercept"] + [f"sin_{i}" for i in range(1, 5)] + [f"cos_{i}" for i in range(1, 5)]),
            global_mean=np.array(metadata.get("global_mean", []), dtype=np.float64) if metadata.get("global_mean") else None,
            prior_alpha=metadata.get("prior_alpha", 0.1),
            segment_keys=metadata.get("segment_keys", []),
            segment_parent_keys=metadata.get("segment_parent_keys", []),
            segment_hierarchy=metadata.get("segment_hierarchy", {}),
            data_checksum=metadata.get("data_checksum", ""),
            split_integrity=metadata.get("split_integrity", {}),
            segment_statistics=metadata.get("segment_statistics", {}),
            support_bins=metadata.get("support_bins", {}),
            hyperparameter_report=metadata.get("hyperparameter_report", {}),
        )

        # Load arrays
        bundle.global_mu0 = arrays.get("global_mu0")
        bundle.global_sigma0 = arrays.get("global_sigma0")
        bundle.global_lambda0 = arrays.get("global_lambda0")
        bundle.global_eta0 = arrays.get("global_eta0")
        bundle.segment_mu0 = arrays.get("segment_mu0")
        bundle.segment_sigma0 = arrays.get("segment_sigma0")
        bundle.segment_lambda0 = arrays.get("segment_lambda0")
        bundle.segment_eta0 = arrays.get("segment_eta0")
        bundle.segment_n_obs = arrays.get("segment_n_obs")
        bundle.segment_n_sellers = arrays.get("segment_n_sellers")
        bundle.segment_is_shrunk = arrays.get("segment_is_shrunk")

        return bundle


# ---------------------------------------------------------------------------
# generate_bundle_id
# ---------------------------------------------------------------------------


def generate_bundle_id() -> str:
    """Generate a unique bundle ID (UUID v4).

    Returns
    -------
    str
        UUID v4 string.

    Examples
    --------
    >>> bid = generate_bundle_id()
    >>> len(bid) == 36
    True
    >>> bid.count("-") == 4
    True
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# generate_compatibility_id
# ---------------------------------------------------------------------------


def generate_compatibility_id(
    k: int,
    sigma2: float,
    reward_params: dict,
    state_history_start: str,
) -> str:
    """TRAIN-08: Generate compatibility ID from model definition.

    Changing any of these parameters requires a new state namespace:
    - k: Fourier basis order (feature dimension)
    - sigma2: Working noise variance
    - reward_params: Reward configuration (weights, costs)
    - state_history_start: Start of state-history interval

    The ID is a SHA-256 hex digest of a canonical JSON representation
    of these parameters.

    Parameters
    ----------
    k : int
        Fourier basis order.
    sigma2 : float
        Working noise variance.
    reward_params : dict
        Reward configuration parameters.
    state_history_start : str
        Inclusive start of state-history interval (ISO date).

    Returns
    -------
    str
        SHA-256 hex digest (compatibility ID).

    Examples
    --------
    >>> cid = generate_compatibility_id(
    ...     k=4, sigma2=0.06,
    ...     reward_params={"w_meeting": 1.0, "w_answered": 0.1,
    ...                    "c_dial": 0.02, "w_not_interested": 0.0},
    ...     state_history_start="2026-04-01"
    ... )
    >>> len(cid) == 64
    True
    """
    canonical = {
        "k": k,
        "sigma2": sigma2,
        "reward_params": dict(sorted(reward_params.items())),
        "state_history_start": state_history_start,
    }
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# compute_segment_support
# ---------------------------------------------------------------------------


def compute_segment_support(
    data: list[dict],
    min_attempts: int = 50,
    min_sellers: int = 30,
) -> dict:
    """TRAIN-05: Compute supported time bins per segment.

    Returns a mapping from segment keys to sets of supported bin keys.
    A bin is supported if it has >= min_attempts attempts and
    >= min_sellers distinct sellers.

    Bin keys are in the format "YYYY-MM-DDTHH:MM" representing the
    start of 15-minute half-open intervals [start, start+15min).

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'segment', 'seller_id', 'call_start_time' fields.
    min_attempts : int
        Minimum attempts per bin (TRAIN-05).
    min_sellers : int
        Minimum distinct sellers per bin (TRAIN-05).

    Returns
    -------
    dict
        Mapping of segment_key → set of supported bin key strings.

    Examples
    --------
    >>> # With sufficient data:
    >>> support = compute_segment_support(data, min_attempts=50, min_sellers=30)
    >>> isinstance(support, dict)
    True
    """
    # Collect per-segment, per-bin statistics
    bin_data: dict[str, dict[str, dict[str, Any]]] = {}

    for record in data:
        segment = record.get("segment", "UNKNOWN")
        call_start = record.get("call_start_time")
        seller_id = record.get("seller_id")

        if call_start is None:
            continue

        # Compute 15-minute bin key
        minute = call_start.minute
        floored_minute = (minute // 15) * 15
        bin_key = call_start.strftime(f"%Y-%m-%dT%H:{floored_minute:02d}")

        if segment not in bin_data:
            bin_data[segment] = {}

        if bin_key not in bin_data[segment]:
            bin_data[segment][bin_key] = {
                "sellers": set(),
                "n_attempts": 0,
            }

        bin_data[segment][bin_key]["n_attempts"] += 1
        if seller_id is not None:
            bin_data[segment][bin_key]["sellers"].add(seller_id)

    # Build support mask: segment_key → set of supported bin keys
    result: dict[str, set[str]] = {}
    for segment, bins in bin_data.items():
        supported_bins: set[str] = set()
        for bin_key, stats in bins.items():
            n_sellers = len(stats["sellers"])
            n_attempts = stats["n_attempts"]
            if n_attempts >= min_attempts and n_sellers >= min_sellers:
                supported_bins.add(bin_key)
        result[segment] = supported_bins

    return result


# ---------------------------------------------------------------------------
# Grid search hyperparameters (TRAIN-04)
# ---------------------------------------------------------------------------


def _grid_search_hyperparameters(
    data: list[dict],
    config: ModelConfig,
    splits: dict,
) -> dict:
    """TRAIN-04: Grid search over hyperparameter combinations.

    Evaluates different (lambda_smooth, lambda_parent, alpha) combinations
    on the validation split and returns the best configuration.

    Parameters
    ----------
    data : list[dict]
        Normalized data.
    config : ModelConfig
        Base model configuration.
    splits : dict
        Chronological splits from create_chronological_splits.

    Returns
    -------
    dict
        Hyperparameter report with grid results and best config.
    """
    # Define grid search space
    lambda_smooth_values = [0.0, 0.1, 0.3, 0.5]
    lambda_parent_values = [0.0, 1.0, 5.0, 10.0, 50.0]
    alpha_values = [0.01, 0.05, 0.1, 0.5]

    validation_data = splits.get("validation", [])
    if not validation_data:
        logger.warning("No validation data for hyperparameter grid search")
        return {
            "grid_results": [],
            "best_config": {
                "lambda_smooth": config.lambda_smooth,
                "lambda_parent": config.lambda_parent,
                "alpha": config.alpha,
            },
            "best_score": 0.0,
        }

    grid_results: list[dict] = []
    best_score = float("-inf")
    best_config = {
        "lambda_smooth": config.lambda_smooth,
        "lambda_parent": config.lambda_parent,
        "alpha": config.alpha,
    }

    for lambda_smooth in lambda_smooth_values:
        for lambda_parent in lambda_parent_values:
            for alpha in alpha_values:
                # Temporarily modify config
                original_smooth = config.lambda_smooth
                original_parent = config.lambda_parent
                original_alpha = config.alpha

                config.lambda_smooth = lambda_smooth
                config.lambda_parent = lambda_parent
                config.alpha = alpha

                try:
                    # Fit priors on prior_fit data
                    prior_fit_data = splits.get("prior_fit", [])
                    if not prior_fit_data:
                        continue

                    result = fit_hierarchical_priors(prior_fit_data, config)

                    # Score on validation data using the fitted priors
                    score = _evaluate_priors_on_data(
                        validation_data, result, config
                    )

                    grid_results.append({
                        "lambda_smooth": lambda_smooth,
                        "lambda_parent": lambda_parent,
                        "alpha": alpha,
                        "score": float(score),
                        "n_segments": len(result.segment_priors),
                    })

                    if score > best_score:
                        best_score = float(score)
                        best_config = {
                            "lambda_smooth": lambda_smooth,
                            "lambda_parent": lambda_parent,
                            "alpha": alpha,
                        }

                except Exception as exc:
                    logger.debug(
                        "Grid search config failed (smooth=%.2f, parent=%.1f, alpha=%.3f): %s",
                        lambda_smooth, lambda_parent, alpha, exc,
                    )
                    continue
                finally:
                    # Restore original config
                    config.lambda_smooth = original_smooth
                    config.lambda_parent = original_parent
                    config.alpha = original_alpha

    return {
        "grid_results": grid_results,
        "best_config": best_config,
        "best_score": float(best_score),
    }


def _evaluate_priors_on_data(
    data: list[dict],
    prior_result: HierarchicalPriorResult,
    config: ModelConfig,
) -> float:
    """Evaluate fitted priors on data using mean Gaussian predictive log-likelihood.

    Each record is scored under its own segment prior (falling back to the
    global prior if the segment has no fitted prior). The predictive variance
    is sigma2 + phi' Sigma0 phi, so both mu0 and Sigma0 affect the score.
    """
    if not data:
        return 0.0

    sigma2 = config.reward_config.sigma2
    k = prior_result.k

    from btc.features.fourier import fourier, time_to_hours
    from btc.model.reward import compute_reward

    total_score = 0.0
    count = 0

    for record in data:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue

        try:
            reward = compute_reward(
                answered=record.get("answered", False),
                meeting_fixed=record.get("meeting_fixed", False),
                disposition=record.get("disposition", "UNKNOWN"),
                reward_config=config.reward_config,
            )
        except ValueError:
            continue

        hours = time_to_hours(call_start)
        phi = fourier(hours, k=k)

        # Use this record's segment prior, falling back to the global prior
        sp = prior_result.segment_priors.get(
            record.get("segment"), prior_result.global_prior
        )
        s2 = sigma2 + float(phi @ sp.Sigma0 @ phi)
        error = reward - float(phi @ sp.mu0)
        total_score += -(error ** 2) / (2.0 * s2) - 0.5 * np.log(s2)
        count += 1

    return total_score / max(count, 1)


# ---------------------------------------------------------------------------
# save_bundle
# ---------------------------------------------------------------------------


def save_bundle(bundle: PriorBundle, output_dir: str) -> str:
    """TRAIN-07: Save bundle as JSON metadata + NPZ arrays.

    Creates the following file structure in output_dir:

        output_dir/
            metadata.json    # Bundle JSON metadata
            arrays.npz       # Numeric arrays (non-pickled)
            checksums.txt    # SHA-256 of each file

    Parameters
    ----------
    bundle : PriorBundle
        Bundle to save.
    output_dir : str
        Directory to save the bundle in.

    Returns
    -------
    str
        Path to the saved bundle directory.

    Raises
    ------
    ValueError
        If bundle arrays are missing or non-finite.
    """
    # Validate bundle
    _validate_bundle_for_save(bundle)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Serialize metadata
    metadata = bundle.to_metadata_dict()
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=True)

    # Serialize arrays to NPZ
    arrays: dict[str, npt.NDArray[np.float64]] = {}

    if bundle.global_mu0 is not None:
        arrays["global_mu0"] = bundle.global_mu0
    if bundle.global_sigma0 is not None:
        arrays["global_sigma0"] = bundle.global_sigma0
    if bundle.global_lambda0 is not None:
        arrays["global_lambda0"] = bundle.global_lambda0
    if bundle.global_eta0 is not None:
        arrays["global_eta0"] = bundle.global_eta0
    if bundle.segment_mu0 is not None:
        arrays["segment_mu0"] = bundle.segment_mu0
    if bundle.segment_sigma0 is not None:
        arrays["segment_sigma0"] = bundle.segment_sigma0
    if bundle.segment_lambda0 is not None:
        arrays["segment_lambda0"] = bundle.segment_lambda0
    if bundle.segment_eta0 is not None:
        arrays["segment_eta0"] = bundle.segment_eta0
    if bundle.segment_n_obs is not None:
        arrays["segment_n_obs"] = bundle.segment_n_obs
    if bundle.segment_n_sellers is not None:
        arrays["segment_n_sellers"] = bundle.segment_n_sellers
    if bundle.segment_is_shrunk is not None:
        arrays["segment_is_shrunk"] = bundle.segment_is_shrunk

    arrays_path = os.path.join(output_dir, "arrays.npz")
    if arrays:
        np.savez_compressed(arrays_path, **arrays)
    else:
        np.savez_compressed(arrays_path)  # Empty NPZ

    # Compute checksums
    checksum_lines: list[str] = []
    for filename in ["metadata.json", "arrays.npz"]:
        filepath = os.path.join(output_dir, filename)
        sha256 = _compute_file_sha256(filepath)
        checksum_lines.append(f"{sha256}  {filename}")

    checksums_path = os.path.join(output_dir, "checksums.txt")
    with open(checksums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    logger.info("Bundle saved to %s (id=%s)", output_dir, bundle.bundle_id)
    return output_dir


def _validate_bundle_for_save(bundle: PriorBundle) -> None:
    """Validate bundle arrays before saving.

    Parameters
    ----------
    bundle : PriorBundle
        Bundle to validate.

    Raises
    ------
    ValueError
        If any array is missing, has wrong shape, or contains non-finite values.
    """
    required_arrays = {
        "global_mu0": bundle.global_mu0,
        "global_sigma0": bundle.global_sigma0,
        "global_lambda0": bundle.global_lambda0,
        "global_eta0": bundle.global_eta0,
        "segment_mu0": bundle.segment_mu0,
        "segment_sigma0": bundle.segment_sigma0,
        "segment_lambda0": bundle.segment_lambda0,
        "segment_eta0": bundle.segment_eta0,
    }

    for name, arr in required_arrays.items():
        if arr is None:
            raise ValueError(f"Missing required array: {name}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values in array: {name}")
        if arr.dtype != np.float64:
            raise ValueError(f"Array {name} must be float64, got {arr.dtype}")

    # Validate segment arrays have consistent dimensions
    n_segments = len(bundle.segment_keys)
    d = bundle.d

    if bundle.segment_mu0 is not None:
        if bundle.segment_mu0.shape != (n_segments, d):
            raise ValueError(
                f"segment_mu0 shape {bundle.segment_mu0.shape} "
                f"expected ({n_segments}, {d})"
            )

    if bundle.segment_sigma0 is not None:
        expected_shape = (n_segments, d, d)
        if bundle.segment_sigma0.shape != expected_shape:
            raise ValueError(
                f"segment_sigma0 shape {bundle.segment_sigma0.shape} "
                f"expected {expected_shape}"
            )

    if bundle.segment_lambda0 is not None:
        expected_shape = (n_segments, d, d)
        if bundle.segment_lambda0.shape != expected_shape:
            raise ValueError(
                f"segment_lambda0 shape {bundle.segment_lambda0.shape} "
                f"expected {expected_shape}"
            )

    if bundle.segment_eta0 is not None:
        if bundle.segment_eta0.shape != (n_segments, d):
            raise ValueError(
                f"segment_eta0 shape {bundle.segment_eta0.shape} "
                f"expected ({n_segments}, {d})"
            )


def _compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 checksum of a file.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# load_bundle
# ---------------------------------------------------------------------------


def load_bundle(bundle_dir: str) -> PriorBundle:
    """Load bundle from disk and validate.

    Reads metadata.json, arrays.npz, and verifies checksums.txt.
    Validates dimensions, finite values, SPD, and checksums on load.

    Parameters
    ----------
    bundle_dir : str
        Path to the bundle directory.

    Returns
    -------
    PriorBundle
        Loaded and validated bundle.

    Raises
    ------
    FileNotFoundError
        If bundle directory or required files are missing.
    ValueError
        If validation fails (dimensions, finite values, SPD, checksums).
    """
    # Check directory exists
    if not os.path.isdir(bundle_dir):
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    # Load metadata
    metadata_path = os.path.join(bundle_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {bundle_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load arrays
    arrays_path = os.path.join(bundle_dir, "arrays.npz")
    if not os.path.isfile(arrays_path):
        raise FileNotFoundError(f"arrays.npz not found in {bundle_dir}")

    raw_arrays = dict(np.load(arrays_path, allow_pickle=False))

    # Convert to proper numpy arrays
    arrays: dict[str, npt.NDArray[np.float64]] = {}
    for key, value in raw_arrays.items():
        if value.ndim == 0:
            # Scalar — skip
            continue
        arrays[key] = value.astype(np.float64, copy=True)

    # Reconstruct bundle
    bundle = PriorBundle.from_metadata_dict(metadata, arrays)

    # Validate loaded bundle
    _validate_bundle_on_load(bundle, arrays)

    # Verify checksums if present
    checksums_path = os.path.join(bundle_dir, "checksums.txt")
    if os.path.isfile(checksums_path):
        _verify_checksums(bundle_dir, checksums_path)

    logger.info("Bundle loaded from %s (id=%s)", bundle_dir, bundle.bundle_id)
    return bundle


def _validate_bundle_on_load(
    bundle: PriorBundle,
    arrays: dict[str, npt.NDArray[np.float64]],
) -> None:
    """Validate loaded bundle arrays.

    Checks:
    - Dimensions match expected shapes
    - All values are finite
    - SPD property for covariance matrices
    - Consistency between global and segment arrays

    Parameters
    ----------
    bundle : PriorBundle
        Bundle to validate.
    arrays : dict[str, np.ndarray]
        Raw arrays from NPZ file.

    Raises
    ------
    ValueError
        If validation fails.
    """
    d = bundle.d
    n_segments = len(bundle.segment_keys)

    if d <= 0:
        raise ValueError(f"Invalid feature dimension: d={d}")

    if n_segments <= 0:
        raise ValueError(f"No segments in bundle: {n_segments}")

    # Validate global arrays
    if bundle.global_mu0 is not None:
        if bundle.global_mu0.shape != (d,):
            raise ValueError(
                f"global_mu0 shape {bundle.global_mu0.shape} "
                f"expected ({d},)"
            )
        if not np.all(np.isfinite(bundle.global_mu0)):
            raise ValueError("global_mu0 contains non-finite values")

    if bundle.global_sigma0 is not None:
        if bundle.global_sigma0.shape != (d, d):
            raise ValueError(
                f"global_sigma0 shape {bundle.global_sigma0.shape} "
                f"expected ({d}, {d})"
            )
        if not np.all(np.isfinite(bundle.global_sigma0)):
            raise ValueError("global_sigma0 contains non-finite values")
        if not _is_spd(bundle.global_sigma0):
            raise ValueError("global_sigma0 is not symmetric positive definite")

    if bundle.global_lambda0 is not None:
        if bundle.global_lambda0.shape != (d, d):
            raise ValueError(
                f"global_lambda0 shape {bundle.global_lambda0.shape} "
                f"expected ({d}, {d})"
            )
        if not np.all(np.isfinite(bundle.global_lambda0)):
            raise ValueError("global_lambda0 contains non-finite values")

    if bundle.global_eta0 is not None:
        if bundle.global_eta0.shape != (d,):
            raise ValueError(
                f"global_eta0 shape {bundle.global_eta0.shape} "
                f"expected ({d},)"
            )
        if not np.all(np.isfinite(bundle.global_eta0)):
            raise ValueError("global_eta0 contains non-finite values")

    # Validate segment arrays
    if bundle.segment_mu0 is not None:
        if bundle.segment_mu0.shape != (n_segments, d):
            raise ValueError(
                f"segment_mu0 shape {bundle.segment_mu0.shape} "
                f"expected ({n_segments}, {d})"
            )
        if not np.all(np.isfinite(bundle.segment_mu0)):
            raise ValueError("segment_mu0 contains non-finite values")

    if bundle.segment_sigma0 is not None:
        if bundle.segment_sigma0.shape != (n_segments, d, d):
            raise ValueError(
                f"segment_sigma0 shape {bundle.segment_sigma0.shape} "
                f"expected ({n_segments}, {d}, {d})"
            )
        if not np.all(np.isfinite(bundle.segment_sigma0)):
            raise ValueError("segment_sigma0 contains non-finite values")
        # Check SPD for each segment
        for i in range(n_segments):
            if not _is_spd(bundle.segment_sigma0[i]):
                seg_key = bundle.segment_keys[i] if i < len(bundle.segment_keys) else f"segment_{i}"
                raise ValueError(
                    f"segment_sigma0[{i}] for {seg_key} is not SPD"
                )

    if bundle.segment_lambda0 is not None:
        if bundle.segment_lambda0.shape != (n_segments, d, d):
            raise ValueError(
                f"segment_lambda0 shape {bundle.segment_lambda0.shape} "
                f"expected ({n_segments}, {d}, {d})"
            )
        if not np.all(np.isfinite(bundle.segment_lambda0)):
            raise ValueError("segment_lambda0 contains non-finite values")

    if bundle.segment_eta0 is not None:
        if bundle.segment_eta0.shape != (n_segments, d):
            raise ValueError(
                f"segment_eta0 shape {bundle.segment_eta0.shape} "
                f"expected ({n_segments}, {d})"
            )
        if not np.all(np.isfinite(bundle.segment_eta0)):
            raise ValueError("segment_eta0 contains non-finite values")

    # Validate consistency: eta ≈ Lambda @ mu for each segment
    if (bundle.segment_eta0 is not None and
            bundle.segment_lambda0 is not None and
            bundle.segment_mu0 is not None):
        for i in range(n_segments):
            expected_eta = bundle.segment_lambda0[i] @ bundle.segment_mu0[i]
            if not np.allclose(
                bundle.segment_eta0[i], expected_eta, atol=1e-10
            ):
                seg_key = bundle.segment_keys[i] if i < len(bundle.segment_keys) else f"segment_{i}"
                raise ValueError(
                    f"segment_eta0[{i}] for {seg_key} inconsistent "
                    f"with Lambda @ mu"
                )


def _verify_checksums(bundle_dir: str, checksums_path: str) -> None:
    """Verify SHA-256 checksums of bundle files.

    Parameters
    ----------
    bundle_dir : str
        Bundle directory path.
    checksums_path : str
        Path to checksums.txt.

    Raises
    ------
    ValueError
        If any checksum doesn't match.
    """
    with open(checksums_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    for line in lines:
        if ":" not in line:
            continue
        filename, expected_checksum = line.split(None, 1)
        filepath = os.path.join(bundle_dir, filename.strip())

        if not os.path.isfile(filepath):
            raise ValueError(f"File referenced in checksums.txt not found: {filename}")

        actual_checksum = _compute_file_sha256(filepath)
        if actual_checksum != expected_checksum.strip():
            raise ValueError(
                f"Checksum mismatch for {filename}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )


# ---------------------------------------------------------------------------
# check_compatibility
# ---------------------------------------------------------------------------


def check_compatibility(
    bundle: PriorBundle,
    state_namespace: str,
) -> bool:
    """TRAIN-08: Check if bundle is compatible with state namespace.

    A bundle is compatible with a state namespace when all of the following
    match:
    - Feature definition (k, d)
    - Reward parameters (w_meeting, w_answered, c_dial, w_not_interested)
    - Noise variance (sigma2)
    - State history start date
    - Normalization parameters

    The compatibility ID is computed from these parameters and compared
    against the bundle's stored compatibility_id.

    Parameters
    ----------
    bundle : PriorBundle
        Bundle to check.
    state_namespace : str
        State namespace identifier (expected compatibility ID).

    Returns
    -------
    bool
        True if bundle is compatible with the state namespace.

    Examples
    --------
    >>> compatible = check_compatibility(bundle, expected_namespace)
    >>> isinstance(compatible, bool)
    True
    """
    if not bundle.bundle_id:
        return False

    # Recompute compatibility ID from bundle parameters
    computed_id = generate_compatibility_id(
        k=bundle.k,
        sigma2=bundle.sigma2,
        reward_params=bundle.reward_params,
        state_history_start=bundle.state_history_start,
    )

    return computed_id == state_namespace


# ---------------------------------------------------------------------------
# run_training_pipeline
# ---------------------------------------------------------------------------


def run_training_pipeline(
    data_dir: str,
    config_path: Optional[str] = None,
    output_dir: str = "",
    dry_run: bool = False,
    config: Optional[Config] = None,
) -> dict:
    """Full training pipeline for the Best Time to Call model.

    Implements the complete training workflow:

    1. Load config from YAML (or use provided config object)
    2. Load CSV data via btc.data.adapters
    3. Normalize via btc.data.normalization
    4. Create chronological splits (TRAIN-06)
    5. Compute segment statistics (TRAIN-01)
    6. Compute support bins (TRAIN-05)
    7. Fit hierarchical priors (TRAIN-02)
    8. Grid search hyperparameters (TRAIN-04)
    9. Create PriorBundle (TRAIN-07)
    10. Validate bundle (TRAIN-07)
    11. Save bundle to output_dir

    Parameters
    ----------
    data_dir : str
        Directory containing raw CSV data files.
        Expected files:
        - Best-Time-to-Call - Call Attempts *.csv
        - Best-Time-to-Call - Sellers.csv
    config_path : str, optional
        Path to YAML configuration file.
        If both ``config_path`` and ``config`` are provided, ``config`` takes
        precedence.
    output_dir : str
        Directory to save the trained bundle.
    dry_run : bool
        If True, run all steps except saving the bundle.
    config : Config, optional
        Pre-loaded Config object. If provided, ``config_path`` is ignored.
        This allows training without reading a YAML file.

    Returns
    -------
    dict
        Training report with keys:
        - bundle_id: str
        - compatibility_id: str
        - n_attempts: int (raw input rows)
        - n_normalized: int (after cleaning)
        - n_splits: dict (records per split purpose)
        - n_segments: int (eligible segments)
        - n_support_bins: dict (supported bins per segment)
        - hyperparameter_best: dict (best hyperparameters)
        - bundle_path: str | None (path to saved bundle, None if dry_run)
        - dry_run: bool

    Raises
    ------
    FileNotFoundError
        If config file or data files are missing.
    ValueError
        If data or configuration is invalid.

    Examples
    --------
    >>> # From config file
    >>> report = run_training_pipeline(
    ...     data_dir="data",
    ...     config_path="config.yaml",
    ...     output_dir="bundles",
    ... )

    >>> # From Config object (no YAML file needed)
    >>> from btc.config import default_config
    >>> report = run_training_pipeline(
    ...     data_dir="data",
    ...     config=default_config(),
    ...     output_dir="bundles",
    ... )
    """
    logger.info("Starting training pipeline")

    # Load config from path or use provided object
    if config is not None:
        logger.info("Step 1/11: Using provided Config object")
        config_obj: Config = config
    elif config_path is not None and os.path.isfile(config_path):
        logger.info("Step 1/11: Loading config from %s", config_path)
        config_obj = load_config(config_path)
    else:
        # Default config if no path provided
        logger.info("Step 1/11: Using default config")
        from btc.config import default_config
        config_obj = default_config()
    model_config: ModelConfig = config_obj.model
    logger.info(
        "  k=%d, sigma2=%.4f, lambda_smooth=%.2f, lambda_parent=%.1f, alpha=%.3f",
        model_config.k,
        config_obj.model.reward_config.sigma2,
        model_config.lambda_smooth,
        model_config.lambda_parent,
        model_config.alpha,
    )

    # Step 2: Load CSV data
    logger.info("Step 2/11: Loading CSV data from %s", data_dir)
    attempts_file = os.path.join(data_dir, "attempts.csv")
    sellers_file = os.path.join(data_dir, "sellers.csv")

    # Try alternative file naming patterns
    if not os.path.isfile(attempts_file):
        attempts_file = _find_csv(data_dir, "attempts")
    if not os.path.isfile(sellers_file):
        sellers_file = _find_csv(data_dir, "sellers")

    if not os.path.isfile(attempts_file):
        raise FileNotFoundError(
            f"Attempts CSV not found in {data_dir}. "
            f"Expected 'attempts.csv' or file containing 'attempts' in name."
        )
    if not os.path.isfile(sellers_file):
        raise FileNotFoundError(
            f"Sellers CSV not found in {data_dir}. "
            f"Expected 'sellers.csv' or file containing 'sellers' in name."
        )

    raw_attempts, attempts_report = load_attempts_csv(attempts_file)
    sellers = load_sellers_csv(sellers_file)
    joined = join_attempts_sellers(raw_attempts, sellers)

    n_raw = attempts_report.get("input_rows", 0)
    n_normalized = attempts_report.get("retained_rows", 0)
    logger.info("  Loaded %d attempts, %d after cleaning (%d sellers)",
                n_raw, n_normalized, len(sellers))

    # Step 3: Normalize data
    logger.info("Step 3/11: Normalizing data")
    # Data is already normalized by adapters; use as-is
    normalized_data = joined
    logger.info("  Normalized %d records", len(normalized_data))

    # Compute data checksum for reproducibility
    data_content = json.dumps(
        sorted([r.get("attempt_id", "") for r in normalized_data]),
        sort_keys=True,
    )
    data_checksum = hashlib.sha256(data_content.encode("utf-8")).hexdigest()

    # Step 4: Create chronological splits (TRAIN-06)
    logger.info("Step 4/11: Creating chronological splits (TRAIN-06)")
    splits = create_chronological_splits(normalized_data, config_obj.timezone)
    split_counts = {purpose: len(records) for purpose, records in splits.items()}
    logger.info("  Split counts: %s", split_counts)

    # Validate split integrity
    integrity = validate_split_integrity(splits)
    logger.info("  Split integrity: %s", integrity)

    # Step 5: Compute segment statistics (TRAIN-01)
    logger.info("Step 5/11: Computing segment statistics (TRAIN-01)")
    segment_stats = compute_segment_statistics(
        normalized_data,
        min_attempts=model_config.segment_min_attempts,
        min_sellers=model_config.segment_min_sellers,
    )
    n_eligible = sum(1 for s in segment_stats.values() if s.get("eligible", False))
    logger.info("  %d segments, %d eligible", len(segment_stats), n_eligible)

    # Step 6: Compute support bins (TRAIN-05)
    logger.info("Step 6/11: Computing support bins (TRAIN-05)")
    support_bins = compute_support_bins(
        normalized_data,
        min_attempts=model_config.support_bin_min_attempts,
        min_sellers=model_config.support_bin_min_sellers,
    )
    total_bins = sum(len(bins) for bins in support_bins.values())
    logger.info("  %d segments with support bins, %d total supported bins",
                len(support_bins), total_bins)

    # Step 7: Fit hierarchical priors (TRAIN-02)
    logger.info("Step 7/11: Fitting hierarchical priors (TRAIN-02)")
    prior_fit_data = splits.get("prior_fit", [])
    if not prior_fit_data:
        raise ValueError("No prior_fit data available for fitting priors")

    prior_result = fit_hierarchical_priors(prior_fit_data, model_config)
    logger.info(
        "  Fitted %d segment priors + global prior, d=%d",
        len(prior_result.segment_priors), prior_result.d,
    )

    # Step 8: Grid search hyperparameters (TRAIN-04)
    logger.info("Step 8/11: Grid searching hyperparameters (TRAIN-04)")
    hyper_report = _grid_search_hyperparameters(normalized_data, model_config, splits)
    best_config = hyper_report.get("best_config", {})
    logger.info("  Best config: %s (score=%.4f)", best_config, hyper_report.get("best_score", 0))

    # Apply best hyperparameters
    if best_config:
        model_config.lambda_smooth = best_config.get("lambda_smooth", model_config.lambda_smooth)
        model_config.lambda_parent = best_config.get("lambda_parent", model_config.lambda_parent)
        model_config.alpha = best_config.get("alpha", model_config.alpha)

        # Refit priors with best hyperparameters
        logger.info("  Refitting priors with best hyperparameters")
        prior_result = fit_hierarchical_priors(prior_fit_data, model_config)

    # Step 9: Create PriorBundle (TRAIN-07)
    logger.info("Step 9/11: Creating PriorBundle (TRAIN-07)")

    # Build reward params dict
    rc = config_obj.model.reward_config
    reward_params = {
        "w_meeting": rc.w_meeting,
        "w_answered": rc.w_answered,
        "c_dial": rc.c_dial,
        "w_not_interested": rc.w_not_interested,
    }

    # Collect segment keys (TRAIN-09: canonical JSON arrays).
    # TRAIN-01: only eligible segments are stored as real priors; thin
    # segments fall back to their parent at inference time.
    segment_keys: list[str] = []
    dropped_thin: list[str] = []
    missing_stats: list[str] = []

    for seg_key in sorted(prior_result.segment_priors.keys()):
        # Validate segment key is a canonical JSON array (TRAIN-09)
        try:
            parsed = json.loads(seg_key)
            if not isinstance(parsed, list):
                raise ValueError("Segment key must be a JSON array")
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f"Invalid segment key (TRAIN-09): {seg_key!r}")

        # The global key is never filtered (it has no segment_stats entry)
        if seg_key == GLOBAL_SEGMENT_KEY:
            segment_keys.append(seg_key)
            continue

        stats = segment_stats.get(seg_key)
        if stats is None:
            missing_stats.append(seg_key)
        elif stats.get("eligible", False):
            segment_keys.append(seg_key)
        else:
            dropped_thin.append(seg_key)

    # Fail loudly on key mismatches instead of silently dropping segments
    if missing_stats:
        raise ValueError(
            f"{len(missing_stats)} fitted segment(s) have no entry in "
            f"segment_stats (key format mismatch?): {missing_stats[:5]}"
        )
    if not segment_keys:
        raise ValueError("No eligible segments after applying TRAIN-01 thresholds")

    logger.info("  Kept %d eligible segments, dropped %d thin segments",
                len(segment_keys), len(dropped_thin))

    # Re-point parents to the nearest surviving ancestor
    kept = set(segment_keys)
    full_hierarchy = prior_result.segment_hierarchy

    def _nearest_kept_parent(key: str) -> Optional[str]:
        parent = full_hierarchy.get(key)
        seen: set[str] = set()
        while (parent is not None and parent not in kept
               and parent != GLOBAL_SEGMENT_KEY and parent not in seen):
            seen.add(parent)
            parent = full_hierarchy.get(parent)
        return parent

    segment_hierarchy = {k: _nearest_kept_parent(k) for k in segment_keys}

    # Build array data
    n_segments = len(segment_keys)
    d = prior_result.d

    global_prior = prior_result.global_prior
    segment_mu0 = np.zeros((n_segments, d), dtype=np.float64)
    segment_sigma0 = np.zeros((n_segments, d, d), dtype=np.float64)
    segment_lambda0 = np.zeros((n_segments, d, d), dtype=np.float64)
    segment_eta0 = np.zeros((n_segments, d), dtype=np.float64)
    segment_n_obs = np.zeros(n_segments, dtype=np.float64)
    segment_n_sellers = np.zeros(n_segments, dtype=np.float64)
    segment_is_shrunk = np.zeros(n_segments, dtype=np.float64)
    segment_parent_keys: list[Optional[str]] = []

    for i, seg_key in enumerate(segment_keys):
        prior = prior_result.segment_priors[seg_key]
        segment_mu0[i] = prior.mu0
        segment_sigma0[i] = prior.Sigma0
        segment_lambda0[i] = prior.Lambda0
        segment_eta0[i] = prior.eta0
        segment_n_obs[i] = prior.n_observations
        segment_n_sellers[i] = prior.n_sellers
        segment_is_shrunk[i] = 1.0 if prior.is_shrunk else 0.0
        segment_parent_keys.append(segment_hierarchy[seg_key]) 

    # Build normalization params (simple feature stats from data)
    normalization = _compute_normalization(normalized_data, model_config.k)

    # Generate IDs
    bundle_id = generate_bundle_id()
    compatibility_id = generate_compatibility_id(
        k=model_config.k,
        sigma2=config_obj.model.reward_config.sigma2,
        reward_params=reward_params,
        state_history_start=model_config.state_history_start,
    )

    bundle = PriorBundle(
        bundle_id=bundle_id,
        compatibility_id=compatibility_id,
        version="1.0.0",
        format_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        k=model_config.k,
        d=d,
        sigma2=config_obj.model.reward_config.sigma2,
        reward_params=reward_params,
        state_history_start=model_config.state_history_start,
        state_history_end=model_config.state_history_end,
        normalization=normalization,
        feature_ordering=["intercept"] + [f"sin_{i}" for i in range(1, model_config.k+1)] + [f"cos_{i}" for i in range(1, model_config.k+1)],
        segment_keys=segment_keys,
        global_mu0=global_prior.mu0,
        global_mean=global_prior.mu0,
        prior_alpha=model_config.alpha,
        global_sigma0=global_prior.Sigma0,
        global_lambda0=global_prior.Lambda0,
        global_eta0=global_prior.eta0,
        segment_mu0=segment_mu0,
        segment_sigma0=segment_sigma0,
        segment_lambda0=segment_lambda0,
        segment_eta0=segment_eta0,
        segment_n_obs=segment_n_obs,
        segment_n_sellers=segment_n_sellers,
        segment_is_shrunk=segment_is_shrunk,
        segment_parent_keys=segment_parent_keys,
        segment_hierarchy=segment_hierarchy,
        data_checksum=data_checksum,
        split_integrity=integrity,
        segment_statistics=segment_stats,
        support_bins=support_bins,
        hyperparameter_report=hyper_report,
    )

    # Step 10: Validate bundle (TRAIN-07)
    logger.info("Step 10/11: Validating bundle (TRAIN-07)")
    _validate_bundle_for_save(bundle)
    logger.info("  Bundle validated: %d segments, d=%d", n_segments, d)

    # Step 11: Save bundle (unless dry_run)
    bundle_path: Optional[str] = None
    if not dry_run:
        logger.info("Step 11/11: Saving bundle to %s", output_dir)
        bundle_path = save_bundle(bundle, output_dir)
        logger.info("  Bundle saved: %s", bundle_path)
    else:
        logger.info("Step 11/11: Dry run — skipping bundle save")

    # Build training report
    report = {
        "bundle_id": bundle_id,
        "compatibility_id": compatibility_id,
        "n_attempts": n_raw,
        "n_normalized": n_normalized,
        "n_splits": split_counts,
        "n_segments": n_eligible,
        "n_support_bins": {k: len(v) for k, v in support_bins.items()},
        "hyperparameter_best": best_config,
        "bundle_path": bundle_path,
        "dry_run": dry_run,
    }

    logger.info("Training pipeline complete. Report: %s", report)
    return report


def _compute_normalization(
    data: list[dict],
    k: int,
) -> dict:
    """Compute normalization parameters from data.

    Computes mean and std of Fourier features across all records.

    Parameters
    ----------
    data : list[dict]
        Normalized data.
    k : int
        Fourier basis order.

    Returns
    -------
    dict
        Normalization params with 'feature_mean' and 'feature_std' arrays.
    """
    from btc.features.fourier import fourier, time_to_hours

    d = feature_dim(k)
    n = len(data)
    if n == 0:
        return {
            "feature_mean": np.zeros(d, dtype=np.float64).tolist(),
            "feature_std": np.ones(d, dtype=np.float64).tolist(),
        }

    feature_sums = np.zeros(d, dtype=np.float64)
    feature_sq_sums = np.zeros(d, dtype=np.float64)
    count = 0

    for record in data:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue

        hours = time_to_hours(call_start)
        phi = fourier(hours, k=k)
        feature_sums += phi
        feature_sq_sums += phi ** 2
        count += 1

    if count == 0:
        return {
            "feature_mean": np.zeros(d, dtype=np.float64).tolist(),
            "feature_std": np.ones(d, dtype=np.float64).tolist(),
        }

    feature_mean = feature_sums / count
    feature_var = feature_sq_sums / count - feature_mean ** 2
    # Ensure non-negative variance
    feature_var = np.maximum(feature_var, 1e-10)
    feature_std = np.sqrt(feature_var)

    return {
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "n_normalizing": count,
    }


def _find_csv(data_dir: str, pattern: str) -> Optional[str]:
    """Find a CSV file matching a pattern in the data directory.

    Parameters
    ----------
    data_dir : str
        Directory to search.
    pattern : str
        Substring pattern to match in filename.

    Returns
    -------
    str | None
        Path to the matching CSV file, or None if not found.
    """
    if not os.path.isdir(data_dir):
        return None

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".csv") and pattern.lower() in filename.lower():
            return os.path.join(data_dir, filename)

    return None
