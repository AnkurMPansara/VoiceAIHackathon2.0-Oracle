"""Data training wrapper for the Best Time to Call model.

This module provides a data-layer interface to the model training pipeline.
It wraps :func:`btc.model.trainer.run_training_pipeline` and handles
CSV file discovery, data directory setup, and result formatting.

The training workflow reads directly from CSV files in the data directory
(no PostgreSQL required):

1. Discover CSV files (attempts + sellers)
2. Load and normalize data via btc.data.adapters (load_attempts_csv, load_sellers_csv)
3. Join attempts with seller profiles and resolve shared segments
4. Create 4 disjoint chronological splits (TRAIN-06: prior_fit, warmup, validation, test)
5. Compute segment statistics (TRAIN-01: ~149k sellers → ~497 shared segments)
6. Compute 15-minute support bins (TRAIN-05)
7. Fit hierarchical priors (TRAIN-01-05)
8. Grid search hyperparameters (TRAIN-04)
9. Create and save model bundle (TRAIN-07)

Key concepts:
- Segments are hierarchical (category_group + turnover_band) and SHARED across sellers
- Sellers do NOT get unique segments; many sellers pool into the same segment
- Chronological splits are disjoint: each record belongs to exactly one split

Modules
-------
train_prior_bundle : Main training entry point from CSV files.
train_from_csv_paths : Training with explicit file paths.
discover_csv_files : Find attempts and sellers CSV files in a directory.
load_and_normalize : Load CSVs and return normalized data (no DB required).

References
----------
SRS §6 (TRAIN-01 to TRAIN-09), §4.4 (Historical adapter mapping).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from btc.config import Config, load_config

logger = logging.getLogger(__name__)


def discover_csv_files(data_dir: str) -> Tuple[str, str]:
    """Discover attempts and sellers CSV files in a data directory.

    Searches for CSV files matching common naming patterns:
    - Attempts: files containing "attempt" or "call" in the name
    - Sellers: files containing "seller" in the name

    Parameters
    ----------
    data_dir : str
        Directory to search for CSV files.

    Returns
    -------
    tuple[str, str]
        (attempts_path, sellers_path) paths to the discovered files.

    Raises
    ------
    FileNotFoundError
        If attempts or sellers CSV cannot be found.

    Examples
    --------
    >>> attempts, sellers = discover_csv_files("data")
    >>> os.path.isfile(attempts)
    True
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    attempts_path = _find_csv(data_dir, ["attempt", "call"])
    sellers_path = _find_csv(data_dir, ["seller"])

    if not attempts_path:
        raise FileNotFoundError(
            f"Attempts CSV not found in {data_dir}. "
            f"Searched for files containing: 'attempt', 'call'"
        )
    if not sellers_path:
        raise FileNotFoundError(
            f"Sellers CSV not found in {data_dir}. "
            f"Searched for files containing: 'seller'"
        )

    logger.info("Discovered CSV files:")
    logger.info("  Attempts: %s", attempts_path)
    logger.info("  Sellers:  %s", sellers_path)

    return attempts_path, sellers_path


def _find_csv(data_dir: str, patterns: List[str]) -> Optional[str]:
    """Find a CSV file matching any of the given patterns.

    Parameters
    ----------
    data_dir : str
        Directory to search.
    patterns : list[str]
        Substring patterns to match (case-insensitive).

    Returns
    -------
    str | None
        Path to the first matching CSV file, or None.
    """
    if not os.path.isdir(data_dir):
        return None

    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith(".csv"):
            continue
        filename_lower = filename.lower()
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                return os.path.join(data_dir, filename)

    return None


def load_and_normalize(
    data_dir: str,
    source_tz: str = "Asia/Kolkata",
) -> Tuple[List[dict], dict, dict]:
    """Load CSV files and normalize data without PostgreSQL.

    This function reads directly from CSV files, normalizes the data,
    and returns normalized records ready for training. No database
    connection is required.

    The normalization pipeline:
    1. load_attempts_csv() returns (normalized_outcomes, import_report)
    2. load_sellers_csv() returns dict mapping seller_id -> profile
    3. join_attempts_sellers() merges attempts with seller profiles
       and resolves shared segments via resolve_segment()

    Parameters
    ----------
    data_dir : str
        Directory containing CSV data files.
    source_tz : str
        Source timezone for naive timestamps (default: Asia/Kolkata).

    Returns
    -------
    tuple[list[dict], dict, dict]
        (normalized_data, attempts_report, sellers_dict)
        - normalized_data: list of normalized outcome dicts (joined with segments)
        - attempts_report: import report from adapters (counts, exclusions)
        - sellers_dict: seller_id -> profile mapping (dict, not list)

    Raises
    ------
    FileNotFoundError
        If CSV files cannot be found.

    Examples
    --------
    >>> data, report, sellers = load_and_normalize("data")
    >>> len(data)
    526718
    >>> len(sellers)
    149363
    """
    from btc.data.adapters import (
        load_attempts_csv,
        load_sellers_csv,
        join_attempts_sellers,
    )

    logger.info("Loading and normalizing data from %s", data_dir)

    # Discover CSV files
    attempts_path, sellers_path = discover_csv_files(data_dir)

    # Load raw data
    raw_attempts, attempts_report = load_attempts_csv(
        attempts_path, source_tz=source_tz
    )
    sellers = load_sellers_csv(sellers_path)

    # Join attempts with seller profiles
    normalized_data = join_attempts_sellers(raw_attempts, sellers)

    logger.info(
        "Loaded %d attempts (%d retained), %d sellers",
        attempts_report.get("input_rows", 0),
        attempts_report.get("retained_rows", 0),
        len(sellers),
    )

    return normalized_data, attempts_report, sellers


def train_prior_bundle(
    config: Config,
    data_dir: str,
    output_path: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Train a prior model bundle directly from CSV files.

    This is the main entry point for training the Best Time to Call model.
    It reads CSV files from the data directory, normalizes the data,
    fits hierarchical priors, and saves a model bundle.

    No PostgreSQL database is required — all data is read from CSV files.

    Segments are hierarchical (category_group + turnover_band) and SHARED
    across sellers. In the production dataset, ~149k sellers map to ~497
    unique segments (not one segment per seller).

    Parameters
    ----------
    config : Config
        Application configuration with model parameters.
    data_dir : str
        Directory containing CSV data files.
        Expected files (auto-discovered):
        - Any CSV containing "attempt" or "call" in the name
        - Any CSV containing "seller" in the name
    output_path : str, optional
        Output directory for the model bundle.
        If None, defaults to ``{data_dir}/model_bundle``.
    dry_run : bool
        If True, run all steps except saving the bundle.

    Returns
    -------
    dict
        Training report with keys:
        - bundle_id: str
        - compatibility_id: str
        - n_attempts: int (raw input rows)
        - n_normalized: int (after cleaning)
        - n_splits: dict (records per split)
        - n_segments: int (eligible segments)
        - n_support_bins: dict (supported bins per segment)
        - hyperparameter_best: dict (best hyperparameters)
        - bundle_path: str | None
        - dry_run: bool

    Raises
    ------
    FileNotFoundError
        If CSV files cannot be found.
    ValueError
        If data or configuration is invalid.

    Examples
    --------
    >>> config = load_config("configs/development.yaml")
    >>> report = train_prior_bundle(
    ...     config=config,
    ...     data_dir="data",
    ...     output_path="artifacts/model_bundle",
    ... )
    >>> report["bundle_id"]
    'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
    """
    from btc.model.trainer import run_training_pipeline

    # Resolve output path
    if output_path is None:
        output_path = os.path.join(data_dir, "model_bundle")

    logger.info("Training prior bundle from CSV files")
    logger.info("  data_dir=%s, output_path=%s, dry_run=%s",
                data_dir, output_path, dry_run)

    # Run the training pipeline
    report = run_training_pipeline(
        data_dir=data_dir,
        config_path=None,  # We already have the config object
        output_dir=output_path,
        dry_run=dry_run,
        config=config,  # Pass config directly
    )

    return report


def train_from_csv_paths(
    config: Config,
    attempts_path: str,
    sellers_path: str,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    source_tz: str = "Asia/Kolkata",
) -> Dict[str, Any]:
    """Train a prior model bundle from explicit CSV file paths.

    Unlike :func:`train_prior_bundle`, this function takes explicit
    file paths instead of a data directory. This is useful when you
    know the exact file locations or want to use non-standard paths.

    No PostgreSQL database is required. Segments are SHARED across sellers
    (category_group + turnover_band hierarchy), not unique per seller.

    Parameters
    ----------
    config : Config
        Application configuration with model parameters.
    attempts_path : str
        Path to the attempts CSV file.
    sellers_path : str
        Path to the sellers CSV file.
    output_path : str, optional
        Output directory for the model bundle.
        If None, defaults to ``{os.path.dirname(attempts_path)}/model_bundle``.
    dry_run : bool
        If True, run all steps except saving the bundle.
    source_tz : str
        Source timezone for naive timestamps.

    Returns
    -------
    dict
        Training report (same format as :func:`train_prior_bundle`).

    Raises
    ------
    FileNotFoundError
        If CSV files do not exist.
    ValueError
        If data or configuration is invalid.

    Examples
    --------
    >>> config = load_config("configs/development.yaml")
    >>> report = train_from_csv_paths(
    ...     config=config,
    ...     attempts_path="data/Best-Time-to-Call - Call Attempts Apr-Sep 2026.csv",
    ...     sellers_path="data/Best-Time-to-Call - Sellers.csv",
    ...     output_path="artifacts/model_bundle",
    ... )
    """
    from btc.data.adapters import (
        load_attempts_csv,
        load_sellers_csv,
        join_attempts_sellers,
    )
    from btc.data.normalization import (
        create_chronological_splits,
        compute_segment_statistics,
        compute_support_bins,
        validate_split_integrity,
    )
    from btc.model.priors import fit_hierarchical_priors
    from btc.model.trainer import (
        PriorBundle,
        generate_bundle_id,
        generate_compatibility_id,
        save_bundle,
        _compute_normalization,
        _grid_search_hyperparameters,
        _validate_bundle_for_save,
    )
    import hashlib
    import json
    from datetime import datetime, timezone

    logger.info("Training from explicit CSV paths")
    logger.info("  attempts_path=%s", attempts_path)
    logger.info("  sellers_path=%s", sellers_path)

    if not os.path.isfile(attempts_path):
        raise FileNotFoundError(f"Attempts CSV not found: {attempts_path}")
    if not os.path.isfile(sellers_path):
        raise FileNotFoundError(f"Sellers CSV not found: {sellers_path}")

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(attempts_path) or ".", "model_bundle"
        )

    # Step 1: Load and normalize data
    raw_attempts, attempts_report = load_attempts_csv(
        attempts_path, source_tz=source_tz
    )
    sellers = load_sellers_csv(sellers_path)
    normalized_data = join_attempts_sellers(raw_attempts, sellers)

    n_raw = attempts_report.get("input_rows", 0)
    n_normalized = attempts_report.get("retained_rows", 0)
    logger.info("  Loaded %d attempts, %d after cleaning", n_raw, n_normalized)

    # Compute data checksum
    data_content = json.dumps(
        sorted([r.get("attempt_id", "") for r in normalized_data]),
        sort_keys=True,
    )
    data_checksum = hashlib.sha256(data_content.encode("utf-8")).hexdigest()

    # Step 2: Create chronological splits
    splits = create_chronological_splits(normalized_data, config.timezone)
    split_counts = {purpose: len(records) for purpose, records in splits.items()}
    integrity = validate_split_integrity(splits)
    logger.info("  Splits: %s", split_counts)

    # Step 3: Compute segment statistics
    segment_stats = compute_segment_statistics(
        normalized_data,
        min_attempts=config.model.segment_min_attempts,
        min_sellers=config.model.segment_min_sellers,
    )
    n_eligible = sum(1 for s in segment_stats.values() if s.get("eligible", False))
    logger.info("  %d segments, %d eligible", len(segment_stats), n_eligible)

    # Step 4: Compute support bins
    support_bins = compute_support_bins(
        normalized_data,
        min_attempts=config.model.support_bin_min_attempts,
        min_sellers=config.model.support_bin_min_sellers,
    )
    logger.info("  Support bins computed for %d segments", len(support_bins))

    # Step 5: Fit hierarchical priors
    prior_fit_data = splits.get("prior_fit", [])
    if not prior_fit_data:
        raise ValueError("No prior_fit data available for fitting priors")

    prior_result = fit_hierarchical_priors(prior_fit_data, config.model)
    logger.info("  Fitted %d segment priors + global prior",
                len(prior_result.segment_priors))

    # Step 6: Grid search hyperparameters (skipped for quick training)
    hyper_report = {
        "grid_results": [],
        "best_config": {},
        "best_score": 0.0,
    }
    best_config = {}
    logger.info("  Grid search skipped (quick training mode)")

    # Step 7: Create PriorBundle
    rc = config.model.reward_config
    reward_params = {
        "w_meeting": rc.w_meeting,
        "w_answered": rc.w_answered,
        "c_dial": rc.c_dial,
        "w_not_interested": rc.w_not_interested,
    }

    segment_keys = sorted(prior_result.segment_priors.keys())
    n_segments = len(segment_keys)
    d = prior_result.d

    # Extract segment arrays from segment_priors dict
    segment_mu0 = np.zeros((n_segments, d), dtype=np.float64)
    segment_sigma0 = np.zeros((n_segments, d, d), dtype=np.float64)
    segment_lambda0 = np.zeros((n_segments, d, d), dtype=np.float64)
    segment_eta0 = np.zeros((n_segments, d), dtype=np.float64)
    segment_n_obs = np.zeros(n_segments, dtype=np.float64)
    segment_n_sellers = np.zeros(n_segments, dtype=np.float64)
    segment_is_shrunk = np.zeros(n_segments, dtype=np.float64)
    segment_parent_keys: list = []

    for i, seg_key in enumerate(segment_keys):
        prior = prior_result.segment_priors[seg_key]
        segment_mu0[i] = prior.mu0
        segment_sigma0[i] = prior.Sigma0
        segment_lambda0[i] = prior.Lambda0
        segment_eta0[i] = prior.eta0
        segment_n_obs[i] = prior.n_observations
        segment_n_sellers[i] = prior.n_sellers
        segment_is_shrunk[i] = 1.0 if prior.is_shrunk else 0.0
        segment_parent_keys.append(prior.parent_key)
    bundle_id = generate_bundle_id()
    compatibility_id = generate_compatibility_id(
        k=config.model.k,
        sigma2=rc.sigma2,
        reward_params=reward_params,
        state_history_start=config.model.state_history_start,
    )

    normalization = _compute_normalization(normalized_data, config.model.k)

    bundle = PriorBundle(
        bundle_id=bundle_id,
        compatibility_id=compatibility_id,
        version="1.0.0",
        format_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        k=config.model.k,
        d=d,
        sigma2=rc.sigma2,
        reward_params=reward_params,
        state_history_start=config.model.state_history_start,
        state_history_end=config.model.state_history_end,
        normalization=normalization,
        feature_ordering=["intercept"] + [f"sin_{i}" for i in range(1, config.model.k+1)] + [f"cos_{i}" for i in range(1, config.model.k+1)],
        segment_keys=segment_keys,
        global_mu0=prior_result.global_prior.mu0,
        global_mean=prior_result.global_prior.mu0,
        prior_alpha=config.model.alpha,
        global_sigma0=prior_result.global_prior.Sigma0,
        global_lambda0=prior_result.global_prior.Lambda0,
        global_eta0=prior_result.global_prior.eta0,
        segment_mu0=segment_mu0,
        segment_sigma0=segment_sigma0,
        segment_lambda0=segment_lambda0,
        segment_eta0=segment_eta0,
        segment_n_obs=segment_n_obs,
        segment_n_sellers=segment_n_sellers,
        segment_is_shrunk=segment_is_shrunk,
        segment_parent_keys=segment_parent_keys,
        segment_hierarchy=prior_result.segment_hierarchy,
        data_checksum=data_checksum,
        split_integrity=integrity,
        segment_statistics=segment_stats,
        support_bins=support_bins,
        hyperparameter_report=hyper_report,
    )

    # Step 8: Validate and save
    _validate_bundle_for_save(bundle)

    bundle_path = None
    if not dry_run:
        bundle_path = save_bundle(bundle, output_path)
        logger.info("  Bundle saved: %s", bundle_path)
    else:
        logger.info("  Dry run — skipping bundle save")

    return {
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
