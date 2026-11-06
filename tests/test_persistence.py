"""Unit and integration tests for persistent breakpoint and bare soil transition detector."""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from haddu.core.harmonic import (
    HarmonicModel,
    fit_harmonic_baseline,
)
from haddu.core.persistence import (
    DEFAULT_CONSECUTIVE_PASSES,
    DEFAULT_Z_THRESHOLD,
    DEFAULT_MIN_AREA_SQM,
    PersistentBreakResult,
    detect_persistent_breaks,
)


def _create_mock_harmonic_model(
    shape: tuple,
    coords: dict,
    c0_val: float = 0.80,
    sigma_val: float = 0.03,
) -> HarmonicModel:
    """Helper to construct a pre-configured HarmonicModel for synthetic tests."""
    c0 = xr.DataArray(np.full(shape, c0_val), dims=["y", "x"], coords=coords)
    c1 = xr.DataArray(np.zeros(shape), dims=["y", "x"], coords=coords)
    a1 = xr.DataArray(np.zeros(shape), dims=["y", "x"], coords=coords)
    b1 = xr.DataArray(np.zeros(shape), dims=["y", "x"], coords=coords)
    sigma = xr.DataArray(np.full(shape, sigma_val), dims=["y", "x"], coords=coords)
    amp = xr.DataArray(np.zeros(shape), dims=["y", "x"], coords=coords)
    phase = xr.DataArray(np.zeros(shape), dims=["y", "x"], coords=coords)

    return HarmonicModel(
        c0=c0,
        c1=c1,
        a1=a1,
        b1=b1,
        sigma=sigma,
        amplitude=amp,
        phase=phase,
        reference_time=pd.Timestamp("2024-01-01"),
    )


def test_persistent_clearing_detection_and_inception_date():
    """Verify that multi-pass clearings are identified with the exact inception break date."""
    h, w, t_steps = 10, 10, 8
    times = pd.date_range("2024-01-01", periods=t_steps, freq="15D")
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
        "spatial_ref": 32643,
    }
    coords = {"time": times, **spatial_coords}

    # Baseline forest: NDVI = 0.80, NDBI = -0.25
    ndvi_data = np.full((t_steps, h, w), 0.80, dtype=np.float32)
    ndbi_data = np.full((t_steps, h, w), -0.25, dtype=np.float32)

    # Inject genuine forest clearing at 3x3 patch (9 pixels >= 500 sqm) starting at time pass 3 (2024-02-15)
    # NDVI drops from 0.80 to 0.25 (severe loss)
    # NDBI jumps from -0.25 to +0.20 (bare soil transition)
    ndvi_data[3:, 2:5, 3:6] = 0.25
    ndbi_data[3:, 2:5, 3:6] = 0.20

    cube = xr.Dataset(
        data_vars={
            "ndvi": (["time", "y", "x"], ndvi_data),
            "ndbi": (["time", "y", "x"], ndbi_data),
        },
        coords=coords,
    )

    model = _create_mock_harmonic_model((h, w), spatial_coords, c0_val=0.80, sigma_val=0.03)

    result = detect_persistent_breaks(
        timeseries_cube=cube,
        harmonic_model=model,
        consecutive_passes=2,
        z_threshold=2.5,
        min_area_sqm=500.0,
    )

    assert isinstance(result, PersistentBreakResult)
    # 9 pixels detected
    assert int(np.count_nonzero(result.change_mask.values)) == 9
    assert result.num_patches == 1
    assert result.total_area_sqm == 900.0  # 9 pixels * 100 sqm

    # Check that the 3x3 patch pixels are True and outside pixels are False
    assert np.all(result.change_mask.values[2:5, 3:6])
    assert not np.any(result.change_mask.values[:2, :])

    # Check exact break date (Pass 3 = 2024-02-15)
    expected_break_date = pd.Timestamp(times[3])
    patch_dates = result.break_dates.values[2:5, 3:6]
    for d in patch_dates.ravel():
        assert pd.Timestamp(d) == expected_break_date

    assert result.earliest_break_date == expected_break_date
    assert result.latest_break_date == expected_break_date

    # Unchanged pixels have NaT
    assert pd.isna(result.break_dates.values[0, 0])
    assert result.break_indices.values[0, 0] == -1


def test_temporary_single_pass_drop_rejection():
    """Verify Acceptance Criteria: temporary single-pass drops (haze/shadow) are rejected."""
    h, w, t_steps = 10, 10, 6
    times = pd.date_range("2024-01-01", periods=t_steps, freq="15D")
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
    }
    coords = {"time": times, **spatial_coords}

    ndvi_data = np.full((t_steps, h, w), 0.80, dtype=np.float32)
    ndbi_data = np.full((t_steps, h, w), -0.25, dtype=np.float32)

    # In pass 2 only, introduce an acute drop across a 4x4 patch (e.g. cloud haze or shadow)
    ndvi_data[2, 2:6, 2:6] = 0.30
    ndbi_data[2, 2:6, 2:6] = 0.15
    # Pass 3 immediately recovers to healthy canopy (0.80)
    ndvi_data[3:, 2:6, 2:6] = 0.80
    ndbi_data[3:, 2:6, 2:6] = -0.25

    cube = xr.Dataset(
        data_vars={
            "ndvi": (["time", "y", "x"], ndvi_data),
            "ndbi": (["time", "y", "x"], ndbi_data),
        },
        coords=coords,
    )

    model = _create_mock_harmonic_model((h, w), spatial_coords, c0_val=0.80, sigma_val=0.03)

    result = detect_persistent_breaks(
        timeseries_cube=cube,
        harmonic_model=model,
        consecutive_passes=2,  # Requires at least 2 consecutive passes
        z_threshold=2.5,
    )

    # Single-pass drop must be completely rejected
    assert np.count_nonzero(result.change_mask.values) == 0
    assert result.num_patches == 0
    assert result.earliest_break_date is None


def test_water_inundation_rejected_by_spectral_confirmation():
    """Verify spectral confirmation: water inundation (negative Delta NDBI) is rejected."""
    h, w, t_steps = 10, 10, 6
    times = pd.date_range("2024-01-01", periods=t_steps, freq="15D")
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
    }
    coords = {"time": times, **spatial_coords}

    ndvi_data = np.full((t_steps, h, w), 0.80, dtype=np.float32)
    ndbi_data = np.full((t_steps, h, w), -0.25, dtype=np.float32)

    # In passes 3..5, simulate water flooding:
    # NDVI drops to -0.15 (water)
    # BUT NDBI drops to -0.55 (water strongly absorbs SWIR -> negative Delta NDBI)
    ndvi_data[3:, 2:6, 2:6] = -0.15
    ndbi_data[3:, 2:6, 2:6] = -0.55

    cube = xr.Dataset(
        data_vars={
            "ndvi": (["time", "y", "x"], ndvi_data),
            "ndbi": (["time", "y", "x"], ndbi_data),
        },
        coords=coords,
    )

    model = _create_mock_harmonic_model((h, w), spatial_coords, c0_val=0.80, sigma_val=0.03)

    # With spectral confirmation enabled (default):
    result = detect_persistent_breaks(
        timeseries_cube=cube,
        harmonic_model=model,
        consecutive_passes=2,
        confirm_ndbi=True,
    )
    assert np.count_nonzero(result.change_mask.values) == 0

    # With spectral confirmation disabled, it would flag NDVI loss:
    result_unconfirmed = detect_persistent_breaks(
        timeseries_cube=cube,
        harmonic_model=model,
        consecutive_passes=2,
        confirm_ndbi=False,
    )
    assert np.count_nonzero(result_unconfirmed.change_mask.values) == 16


def test_cloud_interspersed_persistence():
    """Verify that persistence is preserved across cloudy (NaN) observations."""
    h, w, t_steps = 10, 10, 6
    times = pd.date_range("2024-01-01", periods=t_steps, freq="15D")
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
    }
    coords = {"time": times, **spatial_coords}

    ndvi_data = np.full((t_steps, h, w), 0.80, dtype=np.float32)
    ndbi_data = np.full((t_steps, h, w), -0.25, dtype=np.float32)

    # Clearing occurs at pass 2 (3x3 patch)
    ndvi_data[2, 1:4, 1:4] = 0.25
    ndbi_data[2, 1:4, 1:4] = 0.20

    # Pass 3 is cloudy (NaN)
    ndvi_data[3, 1:4, 1:4] = np.nan
    ndbi_data[3, 1:4, 1:4] = np.nan

    # Pass 4 is clear again and still cleared bare soil
    ndvi_data[4, 1:4, 1:4] = 0.26
    ndbi_data[4, 1:4, 1:4] = 0.21

    cube = xr.Dataset(
        data_vars={
            "ndvi": (["time", "y", "x"], ndvi_data),
            "ndbi": (["time", "y", "x"], ndbi_data),
        },
        coords=coords,
    )

    model = _create_mock_harmonic_model((h, w), spatial_coords, c0_val=0.80, sigma_val=0.03)

    result = detect_persistent_breaks(
        timeseries_cube=cube,
        harmonic_model=model,
        consecutive_passes=2,
    )

    # Even with cloud in pass 3, passes 2 and 4 are consecutive valid observations
    assert np.count_nonzero(result.change_mask.values) == 9
    assert result.earliest_break_date == pd.Timestamp(times[2])


def test_morphological_opening_and_min_area_filtering():
    """Verify that isolated 1-pixel spikes and < 500 sqm clusters are filtered out."""
    h, w, t_steps = 15, 15, 6
    times = pd.date_range("2024-01-01", periods=t_steps, freq="15D")
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
    }
    coords = {"time": times, **spatial_coords}

    ndvi_data = np.full((t_steps, h, w), 0.80, dtype=np.float32)
    ndbi_data = np.full((t_steps, h, w), -0.25, dtype=np.float32)

    # Anomaly from pass 2 onward:
    # 1. Isolated 1-pixel spike at (1, 1)
    ndvi_data[2:, 1, 1] = 0.25
    ndbi_data[2:, 1, 1] = 0.20

    # 2. 2-pixel cluster at (3, 3:5) -> 200 sqm (< 500 sqm)
    ndvi_data[2:, 3, 3:5] = 0.25
    ndbi_data[2:, 3, 3:5] = 0.20

    # 3. 3x3 patch (9 pixels = 900 sqm >= 500 sqm) at (7:10, 7:10)
    ndvi_data[2:, 7:10, 7:10] = 0.25
    ndbi_data[2:, 7:10, 7:10] = 0.20

    cube = xr.Dataset(
        data_vars={
            "ndvi": (["time", "y", "x"], ndvi_data),
            "ndbi": (["time", "y", "x"], ndbi_data),
        },
        coords=coords,
    )

    model = _create_mock_harmonic_model((h, w), spatial_coords, c0_val=0.80, sigma_val=0.03)

    result = detect_persistent_breaks(
        timeseries_cube=cube,
        harmonic_model=model,
        consecutive_passes=2,
        min_area_sqm=500.0,
    )

    # Only the 9-pixel cluster survives
    assert np.count_nonzero(result.change_mask.values) == 9
    assert result.num_patches == 1
    assert result.total_area_sqm == 900.0
    assert not result.change_mask.values[1, 1]
    assert not np.any(result.change_mask.values[3, 3:5])


def test_tuple_unpacking_and_dataset_export():
    """Verify unpacking (change_mask, break_dates) and xarray Dataset export."""
    h, w, t_steps = 6, 6, 4
    times = pd.date_range("2024-01-01", periods=t_steps, freq="15D")
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
    }
    cube = xr.Dataset(
        data_vars={
            "ndvi": (["time", "y", "x"], np.full((t_steps, h, w), 0.80, dtype=np.float32)),
            "ndbi": (["time", "y", "x"], np.full((t_steps, h, w), -0.25, dtype=np.float32)),
        },
        coords={"time": times, **spatial_coords},
    )
    model = _create_mock_harmonic_model((h, w), spatial_coords)

    # 1. Unpacking check
    mask, dates = detect_persistent_breaks(cube, model)
    assert isinstance(mask, xr.DataArray)
    assert isinstance(dates, xr.DataArray)
    assert mask.shape == (h, w)
    assert dates.shape == (h, w)

    # 2. to_dataset check
    res = detect_persistent_breaks(cube, model)
    ds = res.to_dataset()
    assert isinstance(ds, xr.Dataset)
    assert "change_mask" in ds.data_vars
    assert "break_dates" in ds.data_vars
    assert "break_indices" in ds.data_vars
    assert "z_scores" in ds.data_vars
    assert "residuals" in ds.data_vars


def test_input_validation_errors():
    """Verify input validation and informative exceptions."""
    coords = {
        "time": pd.date_range("2024-01-01", periods=4),
        "y": [1, 2],
        "x": [1, 2],
    }
    cube = xr.Dataset(
        data_vars={"ndvi": (["time", "y", "x"], np.ones((4, 2, 2)))},
        coords=coords,
    )
    model = _create_mock_harmonic_model((2, 2), {"y": [1, 2], "x": [1, 2]})

    # consecutive_passes < 1
    with pytest.raises(ValueError, match="consecutive_passes must be >= 1"):
        detect_persistent_breaks(cube, model, consecutive_passes=0)

    # z_threshold <= 0
    with pytest.raises(ValueError, match="z_threshold must be positive"):
        detect_persistent_breaks(cube, model, z_threshold=-1.0)

    # missing ndvi
    with pytest.raises(ValueError, match="must contain an 'ndvi'"):
        detect_persistent_breaks(xr.Dataset({"wrong": cube["ndvi"]}), model)

    # grid shape mismatch
    model_wrong_shape = _create_mock_harmonic_model((3, 3), {"y": [1, 2, 3], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="does not match timeseries_cube"):
        detect_persistent_breaks(cube, model_wrong_shape)


@pytest.mark.integration
def test_real_imagery_persistent_breakpoint_detection():
    """Integration test: Run persistence engine on real Sentinel-2 passes over Thirthahalli.

    Acceptance Criteria:
    - Multi-temporal Sentinel-2 passes are ingested and evaluated against harmonic baseline.
    - Produces valid change mask and break dates without crashes or unhandled exceptions.
    """
    from haddu.core.stac_client import search_passes
    from haddu.core.timeseries import build_timeseries_cube

    test_bbox = [75.10, 13.60, 75.15, 13.65]

    items = search_passes(
        bbox=test_bbox,
        start_date="2024-01-01",
        end_date="2025-12-31",
        max_cloud=15.0,
    )
    assert len(items) >= 6

    # Sample 6 passes spaced at least 45 days apart
    sampled = []
    last_d = None
    for it in items:
        d = it.datetime
        if last_d is None or (d - last_d).days >= 45:
            sampled.append(it)
            last_d = d
        if len(sampled) >= 6:
            break

    cube = build_timeseries_cube(
        bbox=test_bbox,
        items=sampled,
        resolution=100,
        compute=True,
    )

    model = fit_harmonic_baseline(cube, min_valid_obs=4)

    result = detect_persistent_breaks(
        timeseries_cube=cube,
        harmonic_model=model,
        consecutive_passes=2,
        z_threshold=2.5,
        min_area_sqm=500.0,
    )

    assert isinstance(result, PersistentBreakResult)
    assert result.change_mask.shape == cube.ndvi.shape[1:]
    assert result.break_dates.shape == cube.ndvi.shape[1:]
    assert "z_scores" in result.to_dataset().data_vars
