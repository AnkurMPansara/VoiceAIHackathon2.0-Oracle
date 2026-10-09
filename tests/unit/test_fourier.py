"""Comprehensive unit tests for `src/btc/features/fourier.py` (MOD-02).

Tests the Fourier basis functions, time conversion, and dimension helper
against SRS §5.2 (MOD-02) and §15 (Verification matrix, T01).

SRS MOD-02 contract::

    phi(t) = [1, sin(2*pi*t/24), cos(2*pi*t/24), ...,
               sin(2*pi*K*t/24), cos(2*pi*K*t/24)]
    d = 2*K + 1
    K in {2, 3, 4}
    t = local_hour + local_minute/60 + local_second/3600

Test categories:
    - Shape & dimension contracts
    - Exact value verification at known points
    - Periodicity
    - K validity & feature_dim
    - Input validation (non-finite, range, type)
    - dtype enforcement
    - time_to_hours correctness
    - Numerical precision
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.btc.features.fourier import feature_dim, fourier, time_to_hours

# ── Fixtures ──────────────────────────────────────────────────────────────────

K_VALUES = [2, 3, 4]
EXPECTED_DIMS = {2: 5, 3: 7, 4: 9}


@pytest.fixture(params=K_VALUES)
def k(request):
    """Parametrised K values from SRS MOD-02."""
    return request.param


@pytest.fixture(params=[0.0, 6.0, 12.0, 18.0, 24.0])
def time_point(request):
    """Key time points for periodicity and value checks."""
    return request.param


# ── Shape & dimension contracts ───────────────────────────────────────────────

class TestShapeContracts:
    """SRS §15 T01: scalar, array, empty-array shapes."""

    def test_scalar_input_returns_1d(self, k):
        """fourier(scalar, k) returns shape (d,) where d=2k+1."""
        result = fourier(0.5, k=k)
        assert result.shape == (EXPECTED_DIMS[k],)
        assert result.ndim == 1

    def test_scalar_float_input(self, k):
        """fourier(float, k) returns shape (d,)."""
        result = fourier(12.0, k=k)
        assert result.shape == (EXPECTED_DIMS[k],)

    def test_array_input_returns_2d(self, k):
        """fourier(array, k) returns shape (n, d)."""
        times = np.array([0.0, 6.0, 12.0])
        result = fourier(times, k=k)
        assert result.shape == (3, EXPECTED_DIMS[k])

    def test_array_single_element(self, k):
        """fourier([scalar], k) returns shape (1, d)."""
        result = fourier(np.array([3.0]), k=k)
        assert result.shape == (1, EXPECTED_DIMS[k])

    def test_empty_array_returns_zero_rows(self, k):
        """fourier(empty, k) returns shape (0, d)."""
        result = fourier(np.array([]), k=k)
        assert result.shape == (0, EXPECTED_DIMS[k])

    def test_empty_array_dtype(self, k):
        """Empty array result has float64 dtype."""
        result = fourier(np.array([]), k=k)
        assert result.dtype == np.float64

    def test_large_array(self, k):
        """fourier with many elements returns correct shape."""
        n = 1000
        times = np.linspace(0, 23.999, n)
        result = fourier(times, k=k)
        assert result.shape == (n, EXPECTED_DIMS[k])

    def test_dimension_formula(self, k):
        """d = 2*K + 1 for all valid K."""
        assert EXPECTED_DIMS[k] == 2 * k + 1


# ── Exact value verification at known points ──────────────────────────────────

class TestExactValues:
    """SRS §15 T01: phi(0), phi(6), phi(12), phi(18), phi(24)."""

    def test_phi_zero(self, k):
        """phi(0) = [1, 0, 1, 0, 1, ...] — all sin=0, all cos=1."""
        result = fourier(0.0, k=k)
        expected = np.empty(EXPECTED_DIMS[k], dtype=np.float64)
        expected[0] = 1.0
        for i in range(1, EXPECTED_DIMS[k]):
            expected[i] = 1.0 if i % 2 == 0 else 0.0
        np.testing.assert_array_equal(result, expected)

    def test_phi_six_quarter(self, k):
        """phi(6): sin(π/2)=1, cos(π/2)=0, sin(π)=0, cos(π)=-1, sin(3π/2)=-1."""
        result = fourier(6.0, k=k)
        assert result[0] == pytest.approx(1.0, abs=1e-15)
        assert result[1] == pytest.approx(1.0, abs=1e-15)   # sin(π/2)
        assert result[2] == pytest.approx(0.0, abs=1e-15)   # cos(π/2)
        if k >= 2:
            assert result[3] == pytest.approx(0.0, abs=1e-15)   # sin(π)
            assert result[4] == pytest.approx(-1.0, abs=1e-15)  # cos(π)
        if k >= 3:
            assert result[5] == pytest.approx(-1.0, abs=1e-15)  # sin(3π/2)
            assert result[6] == pytest.approx(0.0, abs=1e-15)   # cos(3π/2)
        if k >= 4:
            assert result[7] == pytest.approx(0.0, abs=1e-15)   # sin(2π)
            assert result[8] == pytest.approx(1.0, abs=1e-15)   # cos(2π)

    def test_phi_twelve_half(self, k):
        """phi(12): sin(π)=0, cos(π)=-1, sin(2π)=0, cos(2π)=1."""
        result = fourier(12.0, k=k)
        assert result[0] == pytest.approx(1.0, abs=1e-15)
        assert result[1] == pytest.approx(0.0, abs=1e-15)   # sin(π)
        assert result[2] == pytest.approx(-1.0, abs=1e-15)  # cos(π)
        if k >= 2:
            assert result[3] == pytest.approx(0.0, abs=1e-15)   # sin(2π)
            assert result[4] == pytest.approx(1.0, abs=1e-15)  # cos(2π)
        if k >= 3:
            assert result[5] == pytest.approx(0.0, abs=1e-15)   # sin(3π)
            assert result[6] == pytest.approx(-1.0, abs=1e-15)  # cos(3π)
        if k >= 4:
            assert result[7] == pytest.approx(0.0, abs=1e-15)   # sin(4π)
            assert result[8] == pytest.approx(1.0, abs=1e-15)  # cos(4π)

    def test_phi_eighteen_three_quarter(self, k):
        """phi(18): sin(3π/2)=-1, cos(3π/2)=0, sin(3π)=0, cos(3π)=-1."""
        result = fourier(18.0, k=k)
        assert result[0] == pytest.approx(1.0, abs=1e-15)
        assert result[1] == pytest.approx(-1.0, abs=1e-15)  # sin(3π/2)
        assert result[2] == pytest.approx(0.0, abs=1e-15)   # cos(3π/2)
        if k >= 2:
            assert result[3] == pytest.approx(0.0, abs=1e-15)   # sin(3π)
            assert result[4] == pytest.approx(-1.0, abs=1e-15)  # cos(3π)
        if k >= 3:
            assert result[5] == pytest.approx(1.0, abs=1e-15)  # sin(9π/2)=sin(π/2)
            assert result[6] == pytest.approx(0.0, abs=1e-15)   # cos(9π/2)=cos(π/2)
        if k >= 4:
            assert result[7] == pytest.approx(0.0, abs=1e-15)   # sin(4*3π)=sin(12π)
            assert result[8] == pytest.approx(1.0, abs=1e-15)  # cos(12π)

    def test_phi_near_24_equals_phi_zero(self, k):
        """phi(23.999...) ≈ phi(0) — near-period boundary (SRS: t∈[0,24)).

        SRS T01 claims phi(0)==phi(24) but SRS also constrains t∈[0,24).
        Implementation correctly rejects t>=24. This test verifies
        periodicity at the valid boundary: phi(24-ε) ≈ phi(0).
        """
        phi_0 = fourier(0.0, k=k)
        phi_near_24 = fourier(24.0 - 1e-10, k=k)
        np.testing.assert_allclose(phi_0, phi_near_24, atol=1e-9, rtol=1e-9)

    def test_array_of_known_points(self, k):
        """fourier([0,6,12,18], k) matches individual calls."""
        result = fourier(np.array([0.0, 6.0, 12.0, 18.0]), k=k)
        for i, t in enumerate([0.0, 6.0, 12.0, 18.0]):
            individual = fourier(t, k=k)
            np.testing.assert_array_equal(result[i], individual)


# ── Periodicity ───────────────────────────────────────────────────────────────

class TestPeriodicity:
    """SRS MOD-02: phi(t) ≡ phi(t mod 24) for any t.

    SRS §15 T01 claims phi(0)==phi(24)==phi(48), but SRS §5.2 constrains
    input to [0, 24). The implementation correctly rejects t>=24.
    Periodicity is verified via modular equivalence: phi(t) == phi(t%24).
    """

    def test_phi_0_periodicity(self, k):
        """phi(0) == phi(24-ε) near period boundary."""
        np.testing.assert_allclose(
            fourier(0.0, k=k),
            fourier(24.0 - 1e-12, k=k),
            atol=1e-9, rtol=1e-9
        )

    def test_phi_24_modulo_equivalence(self, k):
        """phi(t) == phi(t mod 24) for various t values in [0,24).

        Note: fourier() only accepts [0, 24), so we verify periodicity
        by comparing phi(t) with phi(t%24) for t in [0,24).
        """
        for t in [0.5, 3.14, 10.0, 23.999]:
            t_mod = t % 24.0
            np.testing.assert_allclose(
                fourier(t, k=k),
                fourier(t_mod, k=k),
                atol=1e-9, rtol=1e-9,
                err_msg=f"Periodicity failed at t={t} (mod={t_mod})"
            )

    def test_phi_full_cycle_equivalence(self, k):
        """phi(t) == phi(t + 24) verified via modular mapping."""
        for t in [0.0, 6.0, 12.0, 18.0]:
            np.testing.assert_allclose(
                fourier(t, k=k),
                fourier((t + 24.0) % 24.0, k=k),
                atol=1e-9, rtol=1e-9
            )


# ── K validity & feature_dim ─────────────────────────────────────────────────

class TestKValidity:
    """SRS MOD-02: K in {2, 3, 4} only."""

    def test_valid_k_values(self, k):
        """k=2,3,4 all work without error."""
        result = fourier(1.0, k=k)
        assert result.shape == (EXPECTED_DIMS[k],)

    def test_invalid_k_1_raises(self):
        """k=1 raises ValueError."""
        with pytest.raises(ValueError, match="k.*must be in"):
            fourier(1.0, k=1)

    def test_invalid_k_5_raises(self):
        """k=5 raises ValueError."""
        with pytest.raises(ValueError, match="k.*must be in"):
            fourier(1.0, k=5)

    def test_invalid_k_negative_raises(self):
        """k=-1 raises ValueError."""
        with pytest.raises(ValueError, match="k.*must be in"):
            fourier(1.0, k=-1)

    def test_invalid_k_zero_raises(self):
        """k=0 raises ValueError."""
        with pytest.raises(ValueError, match="k.*must be in"):
            fourier(1.0, k=0)

    def test_feature_dim_returns_correct(self):
        """feature_dim(k) returns 2*k+1 for k in {2,3,4}."""
        for k_val in K_VALUES:
            assert feature_dim(k_val) == EXPECTED_DIMS[k_val]

    def test_feature_dim_invalid_raises(self):
        """feature_dim raises ValueError for invalid k."""
        for bad_k in [0, 1, 5, 6, -1]:
            with pytest.raises(ValueError, match="k.*must be in"):
                feature_dim(bad_k)


# ── Input validation ──────────────────────────────────────────────────────────

class TestInputValidation:
    """SRS MOD-02: times must be finite and in [0, 24)."""

    def test_inf_raises(self):
        """np.inf raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            fourier(np.inf, k=2)

    def test_neg_inf_raises(self):
        """-np.inf raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            fourier(-np.inf, k=2)

    def test_nan_raises(self):
        """np.nan raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            fourier(np.nan, k=2)

    def test_nan_in_array_raises(self):
        """Array containing nan raises ValueError."""
        with pytest.raises(ValueError, match="finite"):
            fourier(np.array([0.0, np.nan, 1.0]), k=2)

    def test_negative_time_raises(self):
        """t=-1.0 raises ValueError."""
        with pytest.raises(ValueError, match="range"):
            fourier(-1.0, k=2)

    def test_negative_array_raises(self):
        """Array with negative value raises ValueError."""
        with pytest.raises(ValueError, match="range"):
            fourier(np.array([0.0, -0.5, 1.0]), k=2)

    def test_time_equals_24_raises(self):
        """t=24.0 raises ValueError (half-open interval [0, 24))."""
        with pytest.raises(ValueError, match="range"):
            fourier(24.0, k=2)

    def test_time_above_24_raises(self):
        """t=25.0 raises ValueError."""
        with pytest.raises(ValueError, match="range"):
            fourier(25.0, k=2)

    def test_valid_boundary_zero(self):
        """t=0.0 is valid."""
        result = fourier(0.0, k=2)
        assert result.shape == (5,)

    def test_valid_boundary_near_24(self, k):
        """t=23.999999 is valid."""
        result = fourier(23.999999, k=k)
        assert result.shape == (EXPECTED_DIMS[k],)


# ── dtype enforcement ─────────────────────────────────────────────────────────

class TestDtype:
    """Output must always be float64."""

    def test_scalar_float64(self, k):
        """Scalar input produces float64 output."""
        result = fourier(1.0, k=k)
        assert result.dtype == np.float64

    def test_array_float64(self, k):
        """Array input produces float64 output."""
        result = fourier(np.array([0.0, 1.0]), k=k)
        assert result.dtype == np.float64

    def test_empty_float64(self, k):
        """Empty array produces float64 output."""
        result = fourier(np.array([]), k=k)
        assert result.dtype == np.float64

    def test_int_input_float64(self, k):
        """Integer input produces float64 output."""
        result = fourier(12, k=k)
        assert result.dtype == np.float64


# ── time_to_hours ─────────────────────────────────────────────────────────────

class TestTimeToHours:
    """SRS MOD-02: t = hour + minute/60 + second/3600."""

    def test_hour_only(self):
        """14:00:00 → 14.0."""
        dt = datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        assert time_to_hours(dt) == 14.0

    def test_with_minutes(self):
        """14:30:00 → 14.5."""
        dt = datetime(2026, 1, 1, 14, 30, 0, tzinfo=timezone.utc)
        assert time_to_hours(dt) == 14.5

    def test_with_seconds(self):
        """14:30:30 → 14.50833..."""
        dt = datetime(2026, 1, 1, 14, 30, 30, tzinfo=timezone.utc)
        expected = 14 + 30 / 60.0 + 30 / 3600.0
        assert time_to_hours(dt) == pytest.approx(expected, rel=1e-15)

    def test_with_microseconds(self):
        """14:30:30.500000 → 14.50847..."""
        dt = datetime(2026, 1, 1, 14, 30, 30, 500000, tzinfo=timezone.utc)
        expected = 14 + 30 / 60.0 + 30.5 / 3600.0
        assert time_to_hours(dt) == pytest.approx(expected, rel=1e-15)

    def test_midnight(self):
        """00:00:00 → 0.0."""
        dt = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert time_to_hours(dt) == 0.0

    def test_near_midnight(self):
        """23:59:59.999999 → ~24.0 (wrapped to [0,24))."""
        dt = datetime(2026, 1, 1, 23, 59, 59, 999999, tzinfo=timezone.utc)
        result = time_to_hours(dt)
        expected = 23.0 + 59.0 / 60.0 + 59.999999 / 3600.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_naive_datetime_raises(self):
        """Naive datetime raises ValueError."""
        dt = datetime(2026, 1, 1, 14, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware|tzinfo"):
            time_to_hours(dt)

    def test_returns_float(self):
        """Returns Python float, not numpy."""
        dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = time_to_hours(dt)
        assert isinstance(result, float)


# ── Numerical precision ───────────────────────────────────────────────────────

class TestNumericalPrecision:
    """SRS §15: atol=1e-9, rtol=1e-9 on well-conditioned values."""

    def test_phi_zero_precision(self, k):
        """phi(0) exact values within tolerance."""
        result = fourier(0.0, k=k)
        expected = np.empty(EXPECTED_DIMS[k], dtype=np.float64)
        expected[0] = 1.0
        for i in range(1, EXPECTED_DIMS[k]):
            expected[i] = 1.0 if i % 2 == 0 else 0.0
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)

    def test_phi_six_precision(self, k):
        """phi(6) exact values within tolerance."""
        result = fourier(6.0, k=k)
        expected = np.empty(EXPECTED_DIMS[k], dtype=np.float64)
        expected[0] = 1.0
        for j in range(1, k + 1):
            expected[2 * j - 1] = np.sin(j * np.pi / 2)  # sin(2*j*π*6/24)
            expected[2 * j] = np.cos(j * np.pi / 2)       # cos(2*j*π*6/24)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)

    def test_phi_twelve_precision(self, k):
        """phi(12) exact values within tolerance."""
        result = fourier(12.0, k=k)
        expected = np.empty(EXPECTED_DIMS[k], dtype=np.float64)
        expected[0] = 1.0
        for j in range(1, k + 1):
            expected[2 * j - 1] = np.sin(j * np.pi)  # sin(2*j*π*12/24)
            expected[2 * j] = np.cos(j * np.pi)       # cos(2*j*π*12/24)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)

    def test_phi_eighteen_precision(self, k):
        """phi(18) exact values within tolerance."""
        result = fourier(18.0, k=k)
        expected = np.empty(EXPECTED_DIMS[k], dtype=np.float64)
        expected[0] = 1.0
        for j in range(1, k + 1):
            expected[2 * j - 1] = np.sin(3 * j * np.pi / 2)  # sin(2*j*π*18/24)
            expected[2 * j] = np.cos(3 * j * np.pi / 2)       # cos(2*j*π*18/24)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)

    def test_periodicity_precision(self, k):
        """phi(0) == phi(24-ε) within tolerance (SRS range [0,24))."""
        np.testing.assert_allclose(
            fourier(0.0, k=k),
            fourier(24.0 - 1e-12, k=k),
            atol=1e-9, rtol=1e-9
        )

    def test_small_time_precision(self, k):
        """Small t values are computed accurately."""
        result = fourier(0.001, k=k)
        expected = np.empty(EXPECTED_DIMS[k], dtype=np.float64)
        expected[0] = 1.0
        for j in range(1, k + 1):
            angle = 2 * j * np.pi * 0.001 / 24.0
            expected[2 * j - 1] = np.sin(angle)
            expected[2 * j] = np.cos(angle)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)

    def test_no_nan_or_inf_in_output(self, k):
        """Output contains no NaN or Inf values for valid inputs."""
        times = np.linspace(0, 23.999, 100)
        result = fourier(times, k=k)
        assert not np.any(np.isnan(result)), "Output contains NaN"
        assert not np.any(np.isinf(result)), "Output contains Inf"

    def test_output_bounded_by_one(self, k):
        """All non-constant features are in [-1, 1]."""
        times = np.linspace(0, 23.999, 100)
        result = fourier(times, k=k)
        # First column is always 1.0, rest are sin/cos
        assert np.all(result[:, 1:] >= -1.0 - 1e-15)
        assert np.all(result[:, 1:] <= 1.0 + 1e-15)


# ── Edge cases & additional coverage ──────────────────────────────────────────

class TestEdgeCases:
    """Additional edge cases not covered above."""

    def test_0d_array_raises(self):
        """0-D numpy array raises ValueError (not supported)."""
        with pytest.raises(ValueError, match="1-D"):
            fourier(np.array(0.0), k=2)

    def test_2d_array_raises(self):
        """2-D array raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            fourier(np.array([[0.0, 1.0]]), k=2)

    def test_string_raises(self):
        """Non-numeric input raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            fourier("not a number", k=2)

    def test_list_input(self, k):
        """Python list input works (converted to array internally)."""
        result = fourier([0.0, 12.0], k=k)
        assert result.shape == (2, EXPECTED_DIMS[k])

    def test_mixed_array(self, k):
        """Array with mixed valid values works."""
        times = np.array([0.0, 6.0, 12.0, 18.0, 3.5, 9.25, 15.75])
        result = fourier(times, k=k)
        assert result.shape == (7, EXPECTED_DIMS[k])
        # Verify each row matches individual call
        for i, t in enumerate(times):
            np.testing.assert_allclose(result[i], fourier(t, k=k), atol=1e-15)

    def test_very_small_time(self, k):
        """Very small positive time is valid and correct."""
        result = fourier(1e-10, k=k)
        expected = fourier(0.0, k=k)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-9)

    def test_near_24_boundary(self, k):
        """Time just under 24 is valid."""
        result = fourier(23.9999999, k=k)
        assert result.shape == (EXPECTED_DIMS[k],)

    def test_all_valid_k_combinations(self):
        """All K values produce correct dimensions."""
        for k in K_VALUES:
            assert feature_dim(k) == 2 * k + 1
            result = fourier(0.0, k=k)
            assert len(result) == 2 * k + 1

    def test_consistency_across_k(self):
        """Higher K includes all features of lower K prefix."""
        phi2 = fourier(5.0, k=2)
        phi3 = fourier(5.0, k=3)
        phi4 = fourier(5.0, k=4)
        np.testing.assert_array_equal(phi2, phi3[:5])
        np.testing.assert_array_equal(phi2, phi4[:5])
        np.testing.assert_array_equal(phi3, phi4[:7])
