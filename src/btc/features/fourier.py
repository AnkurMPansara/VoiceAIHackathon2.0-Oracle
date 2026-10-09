"""Fourier basis functions for the Best Time to Call system.

Implements MOD-02: Interleaved Fourier basis features for candidate call times.
Maps a 24-hour periodic time variable to a (2K+1)-dimensional Fourier basis,
enabling the model to learn periodic patterns in seller availability.

Modules
-------
fourier : Compute Fourier basis features for candidate timestamps.
time_to_hours : Convert timezone-aware datetime to fractional hours.
feature_dim : Return the number of Fourier dimensions for a given K.

Examples
--------
>>> import numpy as np
>>> fourier(0.0, k=2)
array([1., 0., 1., 0., 1.])
"""

from __future__ import annotations

from datetime import datetime
from typing import Union

import numpy as np
import numpy.typing as npt

# ── Constants ─────────────────────────────────────────────────────────────────

_PERIOD = 24.0
"""Fourier basis period in hours (SRS MOD-02)."""

_VALID_K = frozenset({2, 3, 4})
"""Valid Fourier basis orders (SRS MOD-02, §14)."""


# ── Core function ─────────────────────────────────────────────────────────────


def fourier(times: Union[np.ndarray, float], k: int) -> np.ndarray:
    """Compute Fourier basis features for candidate timestamps.

    MOD-02: Interleaved Fourier basis with exact ordering.

    The basis vector is:

        phi(t) = [1,
                  sin(2*pi*t/24), cos(2*pi*t/24),
                  sin(4*pi*t/24), cos(4*pi*t/24),
                  ...
                  sin(2*K*pi*t/24), cos(2*K*pi*t/24)]

    with total dimension d = 2*K + 1.

    Parameters
    ----------
    times : np.ndarray | float
        Input times in hours (float64). Accepts:
        - Scalar (0-D array or float): returns shape (d,)
        - 1-D array of n times: returns shape (n, d)
        - Empty array: returns shape (0, d)
        All values MUST be finite and in range [0, 24).
    k : int
        Number of sine-cosine pairs. Must be in {2, 3, 4}.
        Total dimensions d = 2*k + 1.

    Returns
    -------
    np.ndarray
        Fourier basis matrix with dtype float64.
        Shape contracts:
        - Scalar input → (d,)
        - n inputs → (n, d)
        - Empty → (0, d)

    Raises
    ------
    ValueError
        If k not in {2, 3, 4}, times contain non-finite values,
        or times are outside [0, 24).

    Notes
    -----
    - Ordering: [1, sin(2πt/24), cos(2πt/24), sin(4πt/24), cos(4πt/24), ...]
    - Periodicity: phi(t) == phi(t + 24) for any t
    - phi(0) = [1, 0, 1, 0, 1, ...] (all sin=0, all cos=1)
    - Uses float64 for numerical precision
    - This is a PURE function with no side effects
    """
    # Validate k
    if k not in _VALID_K:
        raise ValueError(
            f"k (Fourier basis order) must be in {sorted(_VALID_K)}, got {k}"
        )

    # Convert to numpy array, handling scalar input
    is_scalar = np.isscalar(times)
    if is_scalar:
        times_arr = np.array([times], dtype=np.float64)
    elif isinstance(times, np.ndarray):
        times_arr = np.asarray(times, dtype=np.float64)
    else:
        times_arr = np.asarray(times, dtype=np.float64)

    # Ensure 1-D
    if times_arr.ndim != 1:
        raise ValueError("times must be a scalar or 1-D array")

    n = len(times_arr)

    # Handle empty array
    if n == 0:
        d = 2 * k + 1
        return np.zeros((0, d), dtype=np.float64)

    # Validate finite values
    if not np.all(np.isfinite(times_arr)):
        raise ValueError("times must contain only finite values")

    # Validate range [0, 24)
    if np.any(times_arr < 0) or np.any(times_arr >= _PERIOD):
        raise ValueError(
            f"times must be in range [0, {_PERIOD}), "
            f"got min={times_arr.min()}, max={times_arr.max()}"
        )

    # Compute Fourier basis
    d = 2 * k + 1
    result = np.empty((n, d), dtype=np.float64)
    result[:, 0] = 1.0  # bias term

    for j in range(1, k + 1):
        col_sin = 2 * j * np.pi * times_arr / _PERIOD
        sin_col = 2 * j - 1  # sin column index: 2*j - 1
        cos_col = 2 * j      # cos column index: 2*j
        result[:, sin_col] = np.sin(col_sin)
        result[:, cos_col] = np.cos(col_sin)

    # Collapse to 1-D if input was scalar
    if is_scalar:
        return result[0]

    return result


# ── Helper functions ──────────────────────────────────────────────────────────


def time_to_hours(timestamp: datetime) -> float:
    """Convert a timezone-aware datetime to fractional hours in [0, 24).

    Parameters
    ----------
    timestamp : datetime
        Timezone-aware datetime (must have tzinfo).

    Returns
    -------
    float
        Fractional hours: hour + minute/60 + second/3600.
        The result is in [0, 24).

    Raises
    ------
    ValueError
        If timestamp is naive (no tzinfo).

    Notes
    -----
    The conversion uses the local time components of the timestamp
    after timezone normalisation. This ensures that phi(t) respects
    the periodicity phi(t) == phi(t + 24) regardless of timezone.
    """
    if timestamp.tzinfo is None:
        raise ValueError(
            "timestamp must be timezone-aware (has tzinfo); "
            "naive datetimes are not permitted"
        )

    hour = timestamp.hour
    minute = timestamp.minute
    second = timestamp.second + timestamp.microsecond / 1e6

    t = hour + minute / 60.0 + second / 3600.0

    # Wrap to [0, 24) for safety
    t = t % _PERIOD

    return float(t)


def feature_dim(k: int) -> int:
    """Return the number of Fourier dimensions for given K.

    Parameters
    ----------
    k : int
        Number of sine-cosine pairs.

    Returns
    -------
    int
        d = 2*k + 1

    Raises
    ------
    ValueError
        If k not in {2, 3, 4}.

    Notes
    -----
    This is a convenience function for allocating result arrays
    and validating K before calling :func:`fourier`.
    """
    if k not in _VALID_K:
        raise ValueError(
            f"k (Fourier basis order) must be in {sorted(_VALID_K)}, got {k}"
        )
    return 2 * k + 1
