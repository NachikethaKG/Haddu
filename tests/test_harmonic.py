"""Unit and integration tests for harmonic seasonal baseline decomposition."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from haddu.core.harmonic import (
    ANNUAL_PERIOD_DAYS,
    DEFAULT_MIN_VALID_OBS,
    HarmonicModel,
    construct_harmonic_design_matrix,
    extract_time_array,
    fit_harmonic_baseline,
)


def test_extract_time_array_numeric():
    """Verify extract_time_array handles already numeric inputs."""
    # Scalar numeric
    t, ref = extract_time_array(42.5)
    assert t.shape == (1,)
    assert t[0] == 42.5
    assert ref is None

    # Array numeric
    arr = np.array([0, 15.5, 30.0])
    t, ref = extract_time_array(arr)
    assert np.allclose(t, arr)
    assert ref is None


def test_extract_time_array_datetime():
    """Verify continuous and day-of-year time extraction from date strings and timestamps."""
    dates = ["2024-01-01", "2024-01-16", "2025-01-01"]

    # Continuous mode: default ref_ts is Jan 1 of earliest year (2024-01-01)
    t_cont, ref = extract_time_array(dates, time_mode="continuous")
    assert ref == pd.Timestamp("2024-01-01 00:00:00")
    assert np.isclose(t_cont[0], 0.0)
    assert np.isclose(t_cont[1], 15.0)
    # 2024 is leap year -> 366 days
    assert np.isclose(t_cont[2], 366.0)

    # Day of year mode
    t_doy, _ = extract_time_array(dates, time_mode="doy")
    assert np.isclose(t_doy[0], 1.0)
    assert np.isclose(t_doy[1], 16.0)
    assert np.isclose(t_doy[2], 1.0)

    # Custom reference_time
    t_custom, ref_custom = extract_time_array(dates, reference_time="2023-01-01")
    assert ref_custom == pd.Timestamp("2023-01-01 00:00:00")
    assert np.isclose(t_custom[0], 365.0)


def test_construct_harmonic_design_matrix():
    """Verify 4-column design matrix values and periodicity."""
    period = 365.25
    t = np.array([0.0, period / 4.0, period / 2.0, period])
    x = construct_harmonic_design_matrix(t, period=period)

    assert x.shape == (4, 4)
    # Column 0: ones
    assert np.allclose(x[:, 0], 1.0)
    # Column 1: t
    assert np.allclose(x[:, 1], t)
    # Column 2: cos(omega * t)
    assert np.allclose(x[:, 2], [1.0, 0.0, -1.0, 1.0], atol=1e-7)
    # Column 3: sin(omega * t)
    assert np.allclose(x[:, 3], [0.0, 1.0, 0.0, 0.0], atol=1e-7)


def test_synthetic_seasonal_sine_wave_decomposition():
    """Verify exact analytical decomposition of synthetic seasonal sine/cosine waves."""
    # True parameters: evergreen with slight greening trend and phenology
    true_c0 = 0.72
    true_c1 = 0.00008  # slight positive trend
    true_a1 = 0.14
    true_b1 = -0.09
    true_period = 365.25

    # 2 years of observations (48 passes, 15-day intervals)
    times = pd.date_range("2024-01-01", periods=48, freq="15D")
    t, _ = extract_time_array(times, reference_time="2024-01-01")
    x = construct_harmonic_design_matrix(t, period=true_period)
    true_beta = np.array([true_c0, true_c1, true_a1, true_b1])
    y_clean = x @ true_beta

    da = xr.DataArray(
        y_clean,
        dims=["time"],
        coords={"time": times},
        name="ndvi",
    )

    model = fit_harmonic_baseline(da, reference_time="2024-01-01", period=true_period)

    # Recovered parameters should match ground truth to high numerical precision
    assert float(model.c0) == pytest.approx(true_c0, abs=1e-5)
    assert float(model.c1) == pytest.approx(true_c1, abs=1e-6)
    assert float(model.a1) == pytest.approx(true_a1, abs=1e-5)
    assert float(model.b1) == pytest.approx(true_b1, abs=1e-5)

    # Residuals and sigma on clean data should be near zero
    assert float(model.sigma) < 1e-5
    assert float(model.r2) == pytest.approx(1.0, abs=1e-5)

    # Amplitude and phase check
    expected_amp = np.sqrt(true_a1**2 + true_b1**2)
    expected_phase = np.arctan2(true_b1, true_a1)
    assert float(model.amplitude) == pytest.approx(expected_amp, abs=1e-5)
    assert float(model.phase) == pytest.approx(expected_phase, abs=1e-5)


def test_noisy_sine_wave_and_sigma_accuracy():
    """Verify that model.sigma accurately recovers Gaussian noise standard deviation."""
    np.random.seed(42)
    true_c0 = 0.65
    true_c1 = -0.00005
    true_a1 = 0.10
    true_b1 = 0.05
    true_noise_sigma = 0.035

    times = pd.date_range("2024-01-01", periods=60, freq="12D")
    t, _ = extract_time_array(times, reference_time="2024-01-01")
    x = construct_harmonic_design_matrix(t)
    true_beta = np.array([true_c0, true_c1, true_a1, true_b1])
    y_clean = x @ true_beta
    noise = np.random.normal(0.0, true_noise_sigma, size=len(times))
    y_noisy = y_clean + noise

    da = xr.DataArray(
        y_noisy,
        dims=["time"],
        coords={"time": times},
        name="ndvi",
    )

    model = fit_harmonic_baseline(da, reference_time="2024-01-01")

    # Recovered coefficients should be close to true values
    assert float(model.c0) == pytest.approx(true_c0, abs=0.015)
    assert float(model.a1) == pytest.approx(true_a1, abs=0.015)
    assert float(model.b1) == pytest.approx(true_b1, abs=0.015)

    # Recovered sigma should be close to injected noise sigma
    assert float(model.sigma) == pytest.approx(true_noise_sigma, abs=0.006)

    # Standard error arrays should be non-empty, positive, and finite
    assert float(model.se_c0) > 0.0
    assert float(model.se_c1) > 0.0
    assert float(model.se_a1) > 0.0
    assert float(model.se_b1) > 0.0


def test_evergreen_vs_deciduous_sigma_and_amplitude_distinction():
    """Verify Acceptance Criteria: standard deviation maps distinguish evergreen from scrub."""
    np.random.seed(123)
    # Grid of 2x2 pixels
    # Pixel (0, 0): Stable Evergreen canopy (High NDVI, tiny seasonal amplitude, very low noise)
    # Pixel (0, 1): Moist Deciduous canopy (High seasonal amplitude)
    # Pixel (1, 0): Degraded Scrub canopy (High residual variance / noise)
    # Pixel (1, 1): Water body (Low constant NDVI)
    times = pd.date_range("2024-01-01", periods=50, freq="14D")
    n = len(times)
    t, _ = extract_time_array(times, reference_time="2024-01-01")
    x = construct_harmonic_design_matrix(t)

    # Evergreen profile
    y_evergreen = x @ np.array([0.82, 0.0, 0.02, -0.01]) + np.random.normal(0, 0.02, n)
    # Deciduous profile (strong phenological drop in dry season)
    y_deciduous = x @ np.array([0.65, 0.0, 0.18, -0.10]) + np.random.normal(0, 0.03, n)
    # Scrub profile (noisy, variable)
    y_scrub = x @ np.array([0.45, 0.0, 0.05, 0.02]) + np.random.normal(0, 0.09, n)
    # Water profile (stable negative/near-zero)
    y_water = x @ np.array([-0.15, 0.0, 0.01, 0.0]) + np.random.normal(0, 0.015, n)

    grid_data = np.zeros((n, 2, 2))
    grid_data[:, 0, 0] = y_evergreen
    grid_data[:, 0, 1] = y_deciduous
    grid_data[:, 1, 0] = y_scrub
    grid_data[:, 1, 1] = y_water

    da = xr.DataArray(
        grid_data,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="ndvi",
    )

    model = fit_harmonic_baseline(da, reference_time="2024-01-01")

    # 1. Evergreen residual sigma vs scrub residual sigma
    sigma_evergreen = float(model.sigma.sel(y=0, x=0))
    sigma_scrub = float(model.sigma.sel(y=1, x=0))
    assert sigma_evergreen < 0.03
    assert sigma_scrub > 0.07
    assert sigma_evergreen < sigma_scrub / 2.0  # Evergreen variance is < half of scrub

    # 2. Evergreen amplitude vs deciduous amplitude
    amp_evergreen = float(model.amplitude.sel(y=0, x=0))
    amp_deciduous = float(model.amplitude.sel(y=0, x=1))
    assert amp_evergreen < 0.05
    assert amp_deciduous > 0.15

    # 3. Model predict generates baseline arrays
    pred_mid_year = model.predict("2024-07-01")
    assert pred_mid_year.shape == (2, 2)
    assert float(pred_mid_year.sel(y=0, x=0)) > 0.75  # Evergreen stays lush


def test_pixel_missingness_and_nans():
    """Verify vectorized OLS handles clouds/shadows (NaN observations) across pixels."""
    np.random.seed(99)
    times = pd.date_range("2024-01-01", periods=40, freq="15D")
    n = len(times)
    t, _ = extract_time_array(times, reference_time="2024-01-01")
    x = construct_harmonic_design_matrix(t)

    true_b = np.array([0.70, 0.0001, 0.12, -0.06])
    y_clean = x @ true_b

    # Create 3x3 raster
    data = np.broadcast_to(y_clean[:, None, None], (n, 3, 3)).copy()
    data += np.random.normal(0, 0.02, size=data.shape)

    # Inject 30% random NaNs
    nan_mask = np.random.rand(*data.shape) < 0.30
    data[nan_mask] = np.nan

    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [10, 20, 30], "x": [100, 200, 300]},
        name="ndvi",
    )

    model = fit_harmonic_baseline(da, reference_time="2024-01-01")

    # All 9 pixels had sufficient valid observations and should have valid results
    assert not np.isnan(model.c0.values).any()
    assert not np.isnan(model.sigma.values).any()
    assert np.allclose(model.c0.values, true_b[0], atol=0.03)
    assert np.allclose(model.a1.values, true_b[2], atol=0.03)


def test_underdetermined_pixels_handling():
    """Verify pixels with fewer than min_valid_obs are gracefully flagged as NaN."""
    times = pd.date_range("2024-01-01", periods=10, freq="15D")
    n = len(times)
    data = np.full((n, 1, 3), 0.7)

    # Pixel 0: all 10 valid
    # Pixel 1: only 2 valid (< min_valid_obs=4)
    data[2:, 0, 1] = np.nan
    # Pixel 2: all NaN
    data[:, 0, 2] = np.nan

    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0], "x": [0, 1, 2]},
        name="ndvi",
    )

    model = fit_harmonic_baseline(da, min_valid_obs=4)

    # Pixel 0 is good
    assert not np.isnan(float(model.c0.sel(y=0, x=0)))
    assert not np.isnan(float(model.sigma.sel(y=0, x=0)))

    # Pixel 1 and 2 are underdetermined -> NaN
    assert np.isnan(float(model.c0.sel(y=0, x=1)))
    assert np.isnan(float(model.sigma.sel(y=0, x=1)))
    assert np.isnan(float(model.c0.sel(y=0, x=2)))
    assert np.isnan(float(model.sigma.sel(y=0, x=2)))


def test_predict_residuals_and_zscores():
    """Verify prediction, residual calculation, and standardized z-scores on forest loss."""
    times = pd.date_range("2024-01-01", periods=30, freq="15D")
    n = len(times)
    t, _ = extract_time_array(times, reference_time="2024-01-01")
    x = construct_harmonic_design_matrix(t)

    # Stable canopy: c0=0.80, low noise
    y_canopy = x @ np.array([0.80, 0.0, 0.02, 0.0])

    data = np.broadcast_to(y_canopy[:, None, None], (n, 2, 2)).copy()
    # At pixel (1, 1), simulate a sudden illegal deforestation clearing at pass 25
    data[25:, 1, 1] = 0.25  # severe drop from 0.80 to 0.25

    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="ndvi",
    )

    # Fit baseline on early un-cleared passes (first 20 passes)
    baseline_cube = da.isel(time=slice(0, 20))
    model = fit_harmonic_baseline(baseline_cube, reference_time="2024-01-01")

    # Evaluate residuals over full 30 passes
    residuals = model.compute_residuals(da)
    assert residuals.shape == (30, 2, 2)
    # In undisturbed pixel (0, 0), residuals should be close to 0
    assert abs(float(residuals.isel(time=26).sel(y=0, x=0))) < 0.01
    # In cleared pixel (1, 1), residual is approximately -0.55
    res_cleared = float(residuals.isel(time=26).sel(y=1, x=1))
    assert res_cleared < -0.50

    # Compute z-scores
    zscores = model.compute_zscores(da)
    # The cleared pixel should exhibit extreme negative anomaly (z < -10)
    z_cleared = float(zscores.isel(time=26).sel(y=1, x=1))
    assert z_cleared < -10.0


def test_model_unpacking_and_dataset_conversion():
    """Verify tuple unpacking, property access, and xarray Dataset export."""
    times = pd.date_range("2024-01-01", periods=10, freq="15D")
    da = xr.DataArray(
        np.full((10, 2, 2), 0.75),
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="ndvi",
    )

    model = fit_harmonic_baseline(da)

    # 1. Unpacking check: coeffs_ds, sigma_da = model
    coeffs, sigma = model
    assert isinstance(coeffs, xr.Dataset)
    assert isinstance(sigma, xr.DataArray)
    assert "c0" in coeffs.data_vars
    assert "c1" in coeffs.data_vars
    assert "a1" in coeffs.data_vars
    assert "b1" in coeffs.data_vars

    # 2. Dict-like indexing
    assert np.allclose(model["c0"].values, 0.75)
    assert np.allclose(model["sigma"].values, 0.0, atol=1e-6)

    # 3. to_dataset export
    full_ds = model.to_dataset()
    assert "amplitude" in full_ds.data_vars
    assert "phase" in full_ds.data_vars
    assert "r2" in full_ds.data_vars
    assert "se_c0" in full_ds.data_vars
    assert full_ds.attrs["title"] == "Haddu Harmonic Seasonal Baseline Model"


def test_input_validation_errors():
    """Verify descriptive ValueError/TypeError exceptions on invalid inputs."""
    # Not an xarray object
    with pytest.raises(TypeError, match="Expected xarray Dataset or DataArray"):
        fit_harmonic_baseline("not an array")

    # Missing ndvi in Dataset
    with pytest.raises(ValueError, match="Input Dataset must contain an 'ndvi'"):
        fit_harmonic_baseline(xr.Dataset({"wrong_var": xr.DataArray([1, 2])}))

    # Missing time dimension
    with pytest.raises(ValueError, match="Expected time dimension 'time'"):
        fit_harmonic_baseline(xr.DataArray([[1, 2]], dims=["y", "x"]))

    # Too few observations
    short_da = xr.DataArray(
        [0.5, 0.6],
        dims=["time"],
        coords={"time": pd.date_range("2024-01-01", periods=2)},
        name="ndvi",
    )
    with pytest.raises(ValueError, match="has only 2 observation"):
        fit_harmonic_baseline(short_da, min_valid_obs=4)


@pytest.mark.integration
def test_harmonic_baseline_two_year_real_imagery():
    """Integration test: Fit harmonic seasonal baseline over 2 years of real Sentinel-2 passes.

    Acceptance Criteria:
    - Given 2 years of dry-season NDVI observations, outputs predicted baseline arrays
      and standard deviation maps.
    - Distinguishes stable evergreen canopy (low sigma) from variable/scrub pixels.
    """
    from haddu.core.stac_client import search_passes
    from haddu.core.timeseries import build_timeseries_cube

    test_bbox = [75.10, 13.60, 75.15, 13.65]

    # Query 2 years across Western Ghats dry seasons (2024-2025)
    items = search_passes(
        bbox=test_bbox,
        start_date="2024-01-01",
        end_date="2025-12-31",
        max_cloud=15.0,
    )
    assert len(items) >= 6, "Expected at least 6 candidate Sentinel-2 passes across 2 years."

    # Sample passes spaced at least 45 days apart for well-conditioned seasonal distribution
    sampled_items = []
    last_date = None
    for item in items:
        d = item.datetime
        if last_date is None or (d - last_date).days >= 45:
            sampled_items.append(item)
            last_date = d
        if len(sampled_items) >= 8:
            break

    assert len(sampled_items) >= 5, "Expected at least 5 seasonally-spaced passes."

    # Build cube at 100m resolution for fast execution
    cube = build_timeseries_cube(
        bbox=test_bbox,
        items=sampled_items,
        resolution=100,
        compute=True,
    )

    model = fit_harmonic_baseline(cube, min_valid_obs=4)

    # 1. Output dimensions match spatial grid
    assert model.c0.shape == cube.ndvi.shape[1:]
    assert model.sigma.shape == cube.ndvi.shape[1:]
    assert model.amplitude.shape == cube.ndvi.shape[1:]

    # 2. Sigma map has valid values
    valid_sigma = model.sigma.values[~np.isnan(model.sigma.values)]
    assert len(valid_sigma) > 0
    # Canopy pixels in the Western Ghats should exhibit low residual noise (mean < 0.10)
    assert np.nanmean(valid_sigma) < 0.10

    # 3. Model predict produces baseline arrays at future dates
    pred_future = model.predict("2026-03-15")
    assert pred_future.shape == model.c0.shape
    assert not np.isnan(pred_future.values).all()

