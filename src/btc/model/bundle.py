"""Model bundle validation and management.

SRS TRAIN-07: Bundles contain format version, unique bundle ID, model
compatibility ID, K, exact feature ordering, sigma2, reward parameters,
support masks, segment dictionary, means and precisions, profile-mapping
version, data selection boundaries, training statistics, validation
metrics, config hash, code revision, and checksums. Bundles use JSON
metadata plus non-pickled numeric arrays. Dimensions, finite values,
symmetry, SPD, and checksums are validated on load.

References
----------
SRS TRAIN-07, TRAIN-08, MOD-07.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------

def _metadata_path(bundle_dir: str) -> str:
    return os.path.join(bundle_dir, "metadata.json")


def _arrays_path(bundle_dir: str) -> str:
    return os.path.join(bundle_dir, "arrays.npz")


def _checksums_path(bundle_dir: str) -> str:
    return os.path.join(bundle_dir, "checksums.txt")


def _compute_sha256(path: str) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksums(bundle_dir: str) -> Dict[str, str]:
    """Load expected checksums from checksums.txt.

    Expected format: ``<sha256>  <filename>`` (hash first, two spaces, then filename).
    Returns ``{filename: expected_sha256}``.
    """
    path = _checksums_path(bundle_dir)
    if not os.path.exists(path):
        return {}
    checksums: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                checksums[parts[1]] = parts[0]
    return checksums


def _verify_checksums(bundle_dir: str) -> List[str]:
    """Verify file checksums. Returns list of failed file names."""
    expected = _load_checksums(bundle_dir)
    failures: List[str] = []
    for filename, expected_hash in expected.items():
        path = os.path.join(bundle_dir, filename)
        if not os.path.exists(path):
            failures.append(filename)
            continue
        actual = _compute_sha256(path)
        if actual != expected_hash:
            failures.append(filename)
    return failures


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_bundle(bundle_path: str, verbose: bool = False) -> Dict[str, Any]:
    """Validate a model bundle for integrity and correctness.

    Checks performed (TRAIN-07):
    1. Bundle directory exists
    2. metadata.json exists and is valid JSON
    3. arrays.npz exists
    4. Checksums match (if checksums.txt present)
    5. Required metadata fields present
    6. Numeric arrays have correct dimensions
    7. All array values are finite
    8. Precision matrices are symmetric
    9. Precision matrices are SPD (via Cholesky)
    10. Feature ordering matches K

    Parameters
    ----------
    bundle_path : str
        Path to the bundle directory.
    verbose : bool
        If True, include detailed validation results.

    Returns
    -------
    dict
        {"valid": bool, "errors": List[str], "warnings": List[str],
         "bundle_id": str, "compatibility_id": str, "k": int,
         "details": dict (only if verbose)}

    References
    ----------
    SRS TRAIN-07, TRAIN-08.
    """
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {} if verbose else {}
    metadata: Dict[str, Any] = {}

    # 1. Directory exists
    if not os.path.isdir(bundle_path):
        return {
            "valid": False,
            "errors": [f"Bundle directory not found: {bundle_path}"],
            "warnings": [],
            "bundle_id": "",
            "compatibility_id": "",
            "k": 0,
        }

    # 2. metadata.json
    meta_path = _metadata_path(bundle_path)
    if not os.path.exists(meta_path):
        errors.append("metadata.json not found")
    else:
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(f"metadata.json is not valid JSON: {exc}")
            metadata = {}

        # 3. arrays.npz
        arr_path = _arrays_path(bundle_path)
        if not os.path.exists(arr_path):
            errors.append("arrays.npz not found")
        else:
            # 4. Checksums
            checksum_failures = _verify_checksums(bundle_path)
            if checksum_failures:
                errors.append(
                    f"Checksum verification failed for: {checksum_failures}"
                )

            # 5. Required metadata fields
            required_fields = [
                "format_version", "bundle_id", "compatibility_id", "k",
                "feature_ordering", "sigma2", "reward_params",
                "global_mean", "prior_alpha",
            ]
            missing_fields = [
                f for f in required_fields if f not in metadata
            ]
            if missing_fields:
                errors.append(
                    f"Missing required metadata fields: {missing_fields}"
                )

            # 6. K validation
            k = metadata.get("k", 0)
            if k not in (2, 3, 4):
                errors.append(f"Invalid K={k}; must be 2, 3, or 4")

            # 7. Feature ordering matches K
            feature_ordering = metadata.get("feature_ordering", [])
            expected_d = 2 * k + 1 if k in (2, 3, 4) else 0
            if feature_ordering and len(feature_ordering) != expected_d:
                errors.append(
                    f"Feature ordering length {len(feature_ordering)} "
                    f"does not match d={expected_d} for K={k}"
                )

            # 8. sigma2 > 0
            sigma2 = metadata.get("sigma2")
            if sigma2 is not None:
                if not isinstance(sigma2, (int, float)):
                    errors.append("sigma2 must be numeric")
                elif sigma2 <= 0:
                    errors.append(f"sigma2 must be > 0, got {sigma2}")
                elif sigma2 < 1e-4:
                    warnings.append(
                        f"sigma2={sigma2} is below recommended floor of 1e-4"
                    )

            # 9. Load and validate arrays
            if os.path.exists(arr_path):
                try:
                    arrays = np.load(arr_path, allow_pickle=False)
                    details["array_keys"] = list(arrays.keys())

                    # Check all values are finite
                    for key in arrays.keys():
                        arr = arrays[key]
                        if not np.all(np.isfinite(arr)):
                            errors.append(
                                f"Array '{key}' contains non-finite values"
                            )

                    # Validate global_mean dimensions
                    if "global_mean" in arrays:
                        gm = arrays["global_mean"]
                        if gm.shape != (expected_d,):
                            errors.append(
                                f"global_mean shape {gm.shape} "
                                f"does not match (d={expected_d},)"
                            )

                    # Validate segment means
                    for key in arrays.keys():
                        if key.startswith("segment_mean_"):
                            seg_key = key[len("segment_mean_"):]
                            seg_mean = arrays[key]
                            if seg_mean.shape != (expected_d,):
                                errors.append(
                                    f"segment_mean_{seg_key} shape "
                                    f"{seg_mean.shape} does not match "
                                    f"(d={expected_d},)"
                                )

                    # Validate precision matrices (symmetry + SPD)
                    for key in arrays.keys():
                        if key.startswith("segment_precision_"):
                            seg_key = key[len("segment_precision_"):]
                            prec = arrays[key]
                            if prec.shape != (expected_d, expected_d):
                                errors.append(
                                    f"precision_{seg_key} shape "
                                    f"{prec.shape} does not match "
                                    f"({expected_d},{expected_d})"
                                )
                            else:
                                # Symmetry check
                                if not np.allclose(prec, prec.T, atol=1e-9):
                                    errors.append(
                                        f"precision_{seg_key} is not symmetric"
                                    )
                                # SPD check via Cholesky
                                try:
                                    np.linalg.cholesky(prec)
                                except np.linalg.LinAlgError:
                                    errors.append(
                                        f"precision_{seg_key} is not "
                                        f"positive-definite"
                                    )

                    # Validate Sigma0 (prior covariance)
                    if "Sigma0" in arrays:
                        sigma0 = arrays["Sigma0"]
                        if sigma0.shape != (expected_d, expected_d):
                            errors.append(
                                f"Sigma0 shape {sigma0.shape} "
                                f"does not match ({expected_d},{expected_d})"
                            )
                        else:
                            try:
                                np.linalg.cholesky(sigma0)
                            except np.linalg.LinAlgError:
                                errors.append("Sigma0 is not positive-definite")

                    details["array_shapes"] = {
                        k: arrays[k].shape for k in arrays.keys()
                    }
                    details["array_dtypes"] = {
                        k: str(arrays[k].dtype) for k in arrays.keys()
                    }

                except Exception as exc:
                    errors.append(f"Failed to load arrays.npz: {exc}")

            # 10. Segment dictionary consistency
            segment_means = metadata.get("segment_means", {})
            segment_precisions = metadata.get("segment_precisions", {})
            if segment_means and segment_precisions:
                for seg_key in segment_means:
                    if seg_key not in segment_precisions:
                        warnings.append(
                            f"Segment {seg_key} has mean but no precision"
                        )
                for seg_key in segment_precisions:
                    if seg_key not in segment_means:
                        warnings.append(
                            f"Segment {seg_key} has precision but no mean"
                        )

            # Store metadata in details
            if verbose:
                details["metadata_keys"] = list(metadata.keys())
                details["bundle_id"] = metadata.get("bundle_id", "")
                details["compatibility_id"] = metadata.get(
                    "compatibility_id", ""
                )
                details["k"] = metadata.get("k", 0)
                details["format_version"] = metadata.get(
                    "format_version", 0
                )

        # 11. Support mask validation
        support_masks = metadata.get("support_masks", {})
        if support_masks:
            for seg_key, bins in support_masks.items():
                if isinstance(bins, list):
                    for bin_key in bins:
                        if not isinstance(bin_key, str):
                            errors.append(
                                f"Support bin key for segment {seg_key} "
                                f"is not a string: {bin_key}"
                            )

    valid = len(errors) == 0

    result: Dict[str, Any] = {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "bundle_id": metadata.get("bundle_id", "") if metadata else "",
        "compatibility_id": metadata.get("compatibility_id", "")
        if metadata
        else "",
        "k": metadata.get("k", 0) if metadata else 0,
    }

    if verbose:
        result["details"] = details

    return result


def load_bundle(bundle_dir: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load a validated bundle's metadata and arrays.

    Parameters
    ----------
    bundle_dir : str
        Path to the bundle directory.

    Returns
    -------
    tuple[dict, dict]
        (metadata, arrays) where arrays is {key: np.ndarray}.

    Raises
    ------
    ValueError
        If bundle validation fails.
    """
    validation = validate_bundle(bundle_dir, verbose=True)
    if not validation["valid"]:
        raise ValueError(
            f"Bundle validation failed: {validation['errors']}"
        )

    meta_path = _metadata_path(bundle_dir)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    arr_path = _arrays_path(bundle_dir)
    arrays = np.load(arr_path, allow_pickle=False)

    return metadata, dict(arrays)


def save_bundle(
    metadata: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    bundle_dir: str,
    code_revision: str = "unknown",
) -> str:
    """Save a model bundle to disk.

    Parameters
    ----------
    metadata : dict
        Bundle metadata dictionary.
    arrays : dict
        {array_name: numpy_array} to save.
    bundle_dir : str
        Output directory path.
    code_revision : str
        Git revision or commit hash.

    Returns
    -------
    str
        Path to the saved bundle directory.
    """
    os.makedirs(bundle_dir, exist_ok=True)

    # Serialize metadata
    meta_path = _metadata_path(bundle_dir)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save arrays
    arr_path = _arrays_path(bundle_dir)
    np.savez_compressed(arr_path, **arrays)

    # Compute and save checksums
    checksums: Dict[str, str] = {
        "metadata.json": _compute_sha256(meta_path),
        "arrays.npz": _compute_sha256(arr_path),
    }
    cksum_path = _checksums_path(bundle_dir)
    with open(cksum_path, "w") as f:
        f.write(f"# SHA-256 checksums\n")
        f.write(f"# Generated at: {code_revision}\n")
        for filename, checksum in checksums.items():
            f.write(f"{checksum}  {filename}\n")

    return bundle_dir
