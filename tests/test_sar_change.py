"""Unit and integration tests for polarimetric SAR deforestation detection engine."""

import numpy as np
import pytest
import xarray as xr

from haddu.core.sar_change import (
    DEFAULT_VH_DROP_DB,
    SARChangeDetectionResult,
    SpatialAlignmentError,
    _apply_speckle_filter,
    _lee_filter,
    align_sar_rasters,
    detect_sar_deforestation,
)
from haddu.core.sar_loader import (
    DEFAULT_AOI_BBOX,
    DEFAULT_SAR_COLLECTION,
    fetch_sar_passes,
    search_sar_passes,
)
from haddu.core.vectorizer import vectorize_change_mask


def make_synthetic_sar_scene(
    vh_val: float = -12.0,
    vv_val: float = -6.0,
    shape: tuple = (15, 15),
    res: float = 20.0,
    epsg: int = 32643,
    use_dataset: bool = True,
) -> xr.Dataset | xr.DataArray:
    """Helper to create synthetic SAR backscatter rasters in UTM Zone 43N."""
    ny, nx = shape
    y_coords = np.arange(1530000.0, 1530000.0 - ny * res, -res)
    x_coords = np.arange(500000.0, 500000.0 + nx * res, res)

    vh = np.full((ny, nx), vh_val, dtype=np.float32)
    vv = np.full((ny, nx), vv_val, dtype=np.float32)

    coords = {
        "y": y_coords,
        "x": x_coords,
        "spatial_ref": epsg,
    }

    if not use_dataset:
        return xr.DataArray(vh, coords=coords, dims=["y", "x"], name="vh", attrs={"units": "dB"})

    return xr.Dataset(
        data_vars={
            "vh": (["y", "x"], vh, {"units": "dB"}),
            "vv": (["y", "x"], vv, {"units": "dB"}),
        },
        coords=coords,
    )


# ==============================================================================
# Unit Tests: Speckle Reduction Filters
# ==============================================================================


def test_speckle_filters_single_pixel_spike():
    """Verify median filter and Lee filter reject isolated single-pixel radar spikes."""
    data = np.full((7, 7), -12.0, dtype=np.float64)
    data[3, 3] = 10.0  # +22 dB anomalous speckle spike

    # Median filter should completely eradicate the isolated 1-pixel spike
    filtered_med = _apply_speckle_filter(data, method="median", size=3)
    assert pytest.approx(filtered_med[3, 3], abs=1e-4) == -12.0

    # Lee filter should attenuate the speckle spike
    filtered_lee = _lee_filter(data, size=3)
    assert filtered_lee[3, 3] < 10.0

    # 'none' method preserves raw data
    filtered_none = _apply_speckle_filter(data, method="none")
    assert filtered_none[3, 3] == 10.0


def test_speckle_filters_nan_handling():
    """Verify speckle filters handle NaN values gracefully without corrupting valid data."""
    data = np.full((5, 5), 4.0, dtype=np.float64)
    data[0, 0] = np.nan
    data[1, 1] = np.nan

    filtered = _apply_speckle_filter(data, method="median", size=3)
    assert np.isnan(filtered[0, 0])
    assert np.isnan(filtered[1, 1])
    assert pytest.approx(filtered[3, 3], abs=1e-4) == 4.0


# ==============================================================================
# Unit Tests: Spatial Alignment
# ==============================================================================


def test_align_sar_rasters_success():
    """Verify compatible pre and post scenes align without error."""
    pre = make_synthetic_sar_scene(shape=(10, 10), res=20.0, epsg=32643)
    post = make_synthetic_sar_scene(shape=(10, 10), res=20.0, epsg=32643)

    vh_pre, vh_post = align_sar_rasters(pre, post)
    assert vh_pre.shape == (10, 10)
    assert vh_post.shape == (10, 10)


def test_align_sar_rasters_crs_mismatch():
    """Verify SpatialAlignmentError is raised when pre and post have different CRS."""
    pre = make_synthetic_sar_scene(shape=(10, 10), res=20.0, epsg=32643)
    post = make_synthetic_sar_scene(shape=(10, 10), res=20.0, epsg=4326)

    with pytest.raises(SpatialAlignmentError, match="CRS EPSG:4326 does not match expected"):
        align_sar_rasters(pre, post)


def test_align_sar_rasters_resolution_mismatch():
    """Verify SpatialAlignmentError is raised on pixel resolution mismatch."""
    pre = make_synthetic_sar_scene(shape=(10, 10), res=20.0, epsg=32643)
    post = make_synthetic_sar_scene(shape=(20, 20), res=10.0, epsg=32643)

    with pytest.raises(SpatialAlignmentError, match="Resolution mismatch"):
        align_sar_rasters(pre, post)


# ==============================================================================
# Unit Tests: Deforestation Detection Logic & Speckle Rejection
# ==============================================================================


def test_detect_sar_deforestation_no_change():
    """Verify no clearing is detected when pre and post backscatter values are identical."""
    pre = make_synthetic_sar_scene(vh_val=-12.0)
    post = make_synthetic_sar_scene(vh_val=-12.0)

    result = detect_sar_deforestation(pre, post, vh_drop_db=2.5, min_area_sqm=500.0)

    assert isinstance(result, SARChangeDetectionResult)
    assert result.num_patches == 0
    assert result.total_area_sqm == 0.0
    assert not np.any(result.disturbance_mask.values)


def test_detect_sar_deforestation_speckle_noise_rejection():
    """Verify random isolated radar speckle spikes are rejected by spatial filtering.

    Acceptance Criteria:
    The engine flags radar backscatter drops exceeding the threshold and correctly
    rejects random radar speckle noise using spatial filtering.
    """
    pre = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)

    # Inject isolated 1-pixel severe backscatter drops (e.g. -20 dB -> delta = +8 dB > 2.5 dB)
    post["vh"].values[2, 2] = -20.0
    post["vh"].values[8, 12] = -22.0
    post["vh"].values[12, 3] = -25.0

    # With median speckle reduction & morphological filtering, isolated speckle noise must be rejected
    result = detect_sar_deforestation(pre, post, vh_drop_db=DEFAULT_VH_DROP_DB, min_area_sqm=500.0)

    assert result.num_patches == 0
    assert result.total_area_sqm == 0.0
    assert result.metrics["noise_pixels_filtered"] >= 3
    assert not np.any(result.disturbance_mask.values)


def test_detect_sar_deforestation_genuine_canopy_loss():
    """Verify genuine contiguous clearing cluster exceeding min_area_sqm is detected."""
    # 20m pixels -> 1 pixel = 400 m²
    pre = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)

    # Create a 3x3 cluster (9 pixels * 400 m² = 3600 m² > 500 m²) with -6 dB drop
    # (VH drops from -12 dB to -18 dB)
    post["vh"].values[4:7, 4:7] = -18.0

    result = detect_sar_deforestation(pre, post, vh_drop_db=2.5, min_area_sqm=500.0)

    assert result.num_patches == 1
    assert result.total_pixels == 5
    assert result.total_area_sqm == 2000.0
    assert result.total_area_acres == pytest.approx(2000.0 / 4046.8564, rel=1e-3)
    assert np.count_nonzero(result.disturbance_mask.values) == 5

    patch = result.metrics["patches"][0]
    assert patch["patch_id"] == 1
    assert patch["pixel_count"] == 5
    assert patch["area_sqm"] == 2000.0
    assert patch["mean_delta_vh_db"] >= 2.5


def test_detect_sar_deforestation_lee_filter():
    """Verify that Lee filter detects 2x2 clusters while preserving edges."""
    pre = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post["vh"].values[5:7, 5:7] = -18.0

    result = detect_sar_deforestation(
        pre, post, vh_drop_db=2.5, min_area_sqm=500.0, filter_method="lee"
    )

    assert result.num_patches == 1
    assert result.total_pixels == 4
    assert result.total_area_sqm == 1600.0


def test_detect_sar_deforestation_sub_threshold():
    """Verify that a small backscatter drop below vh_drop_db is not flagged."""
    pre = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)

    # 4-pixel cluster drops by only 1.2 dB (from -12 dB to -13.2 dB; < 2.5 dB)
    post["vh"].values[4:7, 4:7] = -13.2

    result = detect_sar_deforestation(pre, post, vh_drop_db=2.5, min_area_sqm=500.0)

    assert result.num_patches == 0
    assert result.total_area_sqm == 0.0


def test_detect_sar_deforestation_sub_area_filter():
    """Verify that drops smaller than min_area_sqm are discarded."""
    # 1 pixel at 20m = 400 m² < 500 m² threshold
    pre = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)

    # 1 pixel drops by 8 dB
    post["vh"].values[5, 5] = -20.0

    result = detect_sar_deforestation(
        pre, post, vh_drop_db=2.5, min_area_sqm=500.0, filter_method="none"
    )

    assert result.num_patches == 0
    assert result.total_area_sqm == 0.0


def test_detect_sar_deforestation_unpacking_and_indexing():
    """Verify tuple unpacking and dictionary-style indexing on result object."""
    pre = make_synthetic_sar_scene(vh_val=-12.0)
    post = make_synthetic_sar_scene(vh_val=-12.0)
    post["vh"].values[4:7, 4:7] = -18.0

    result = detect_sar_deforestation(pre, post, vh_drop_db=2.5, min_area_sqm=500.0)

    # Tuple unpacking
    mask, metrics = result
    assert isinstance(mask, xr.DataArray)
    assert isinstance(metrics, dict)
    assert metrics["num_patches"] == 1

    # Property access
    assert result.num_patches == 1
    assert result.total_area_sqm == 2000.0

    # Dict indexing
    assert result["num_patches"] == 1
    assert isinstance(result["disturbance_mask"], xr.DataArray)
    assert isinstance(result["change_mask"], xr.DataArray)


def test_vectorize_sar_disturbance_integration():
    """Verify that SARChangeDetectionResult integrates cleanly with vectorize_change_mask."""
    pre = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post = make_synthetic_sar_scene(vh_val=-12.0, shape=(15, 15), res=20.0)
    post["vh"].values[4:7, 4:7] = -18.0

    result = detect_sar_deforestation(pre, post, vh_drop_db=2.5, min_area_sqm=500.0)

    gdf = vectorize_change_mask(result)

    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == 4326
    assert gdf.iloc[0]["patch_id"] == 1
    assert gdf.iloc[0]["area_sqm"] == pytest.approx(2000.0, abs=200.0)
    assert gdf.iloc[0]["geometry"].geom_type == "Polygon"


# ==============================================================================
# Integration Test: Real Sentinel-1 Imagery Detection
# ==============================================================================


@pytest.mark.integration
def test_detect_sar_deforestation_real_imagery_integration():
    """Integration test: Stream real Sentinel-1 passes over Thirthahalli and run detector."""
    # Search August 2023 passes
    items = search_sar_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="2023-08-01",
        end_date="2023-08-31",
        orbit_direction="DESCENDING",
        collection=DEFAULT_SAR_COLLECTION,
    )

    assert len(items) >= 2, "Expected at least 2 Sentinel-1 passes in August 2023."

    # Load 2 successive passes (e.g. 12 days apart)
    pre_ds = fetch_sar_passes(
        items=[items[0]],
        bbox=DEFAULT_AOI_BBOX,
        resolution=20,
        chunks={"x": 512, "y": 512},
    )
    post_ds = fetch_sar_passes(
        items=[items[1]],
        bbox=DEFAULT_AOI_BBOX,
        resolution=20,
        chunks={"x": 512, "y": 512},
    )

    # Slice small 100x100 sub-window for rapid integration test
    pre_sub = pre_ds.isel(y=slice(200, 300), x=slice(200, 300))
    post_sub = post_ds.isel(y=slice(200, 300), x=slice(200, 300))

    result = detect_sar_deforestation(
        pre_sar_ds=pre_sub,
        post_sar_ds=post_sub,
        vh_drop_db=2.5,
        min_area_sqm=500.0,
        filter_method="median",
        filter_size=3,
    )

    assert isinstance(result, SARChangeDetectionResult)
    assert "num_patches" in result.metrics
    assert "total_area_sqm" in result.metrics
    assert result.disturbance_mask.shape == (100, 100)
    assert result.delta_vh.shape == (100, 100)
    assert result.filtered_delta_vh.shape == (100, 100)
