"""Unit and integration tests for bi-temporal change detection and morphological cleanup."""

import numpy as np
import pytest
import xarray as xr

from haddu.core.change_detector import (
    DEFAULT_NDVI_THRESHOLD,
    DEFAULT_MIN_AREA_SQM,
    SpatialAlignmentError,
    ChangeDetectionResult,
    verify_spatial_alignment,
    detect_canopy_loss,
)
from haddu.core.stac_client import DEFAULT_AOI_BBOX, search_passes
from haddu.core.loader import load_stac_dataset


def make_synthetic_scene(
    b08_val: float = 5000.0,
    b04_val: float = 500.0,
    scl_val: int = 4,
    shape: tuple = (15, 15),
    res: float = 10.0,
    epsg: int = 32643,
) -> xr.Dataset:
    """Helper to create a synthetic 10m UTM Zone 43N dataset."""
    ny, nx = shape
    y_coords = np.arange(1530000.0, 1530000.0 - ny * res, -res)
    x_coords = np.arange(730000.0, 730000.0 + nx * res, res)

    b08 = np.full((ny, nx), b08_val, dtype=np.float32)
    b04 = np.full((ny, nx), b04_val, dtype=np.float32)
    scl = np.full((ny, nx), scl_val, dtype=np.int32)

    ds = xr.Dataset(
        data_vars={
            "B08": (["y", "x"], b08),
            "B04": (["y", "x"], b04),
            "SCL": (["y", "x"], scl),
        },
        coords={
            "y": y_coords,
            "x": x_coords,
            "spatial_ref": epsg,
        },
    )
    return ds


def test_spatial_alignment_verification():
    """Verify that verify_spatial_alignment catches CRS, resolution, and coordinate mismatches."""
    base = make_synthetic_scene(shape=(10, 10), res=10.0, epsg=32643)

    # 1. Matching scenes should succeed
    verify_spatial_alignment(base, base.copy())

    # 2. CRS mismatch
    diff_crs = make_synthetic_scene(shape=(10, 10), res=10.0, epsg=4326)
    with pytest.raises(SpatialAlignmentError, match="CRS EPSG:4326 does not match expected EPSG:32643"):
        verify_spatial_alignment(base, diff_crs)

    # 3. Resolution mismatch (20m vs 10m)
    diff_res = make_synthetic_scene(shape=(10, 10), res=20.0, epsg=32643)
    with pytest.raises(SpatialAlignmentError, match="resolution .* does not match expected 10.0m"):
        verify_spatial_alignment(base, diff_res)

    # 4. Shape mismatch
    diff_shape = make_synthetic_scene(shape=(12, 10), res=10.0, epsg=32643)
    with pytest.raises(SpatialAlignmentError, match="coordinate shape mismatch"):
        verify_spatial_alignment(base, diff_shape)


def test_detect_canopy_loss_filters_single_pixel_spikes():
    """Verify algorithm detects genuine clearing clusters and eliminates isolated noise spikes.

    Acceptance criteria:
    - 3x3 contiguous cluster (9 pixels = 900 m² >= 500 m²) is identified.
    - Single-pixel noise spike (1 pixel = 100 m²) is eliminated by binary opening.
    - 2-pixel cluster (200 m² < 500 m²) is eliminated by min_area_sqm filter.
    """
    # Pre-scene: Continuous healthy forest canopy across entire grid (NDVI ~= 0.818)
    # B08=5000, B04=500 -> (5000-500)/(5000+500) ~= 0.818
    pre_ds = make_synthetic_scene(shape=(15, 15), b08_val=5000.0, b04_val=500.0)

    # Post-scene: Baseline identical to pre-scene
    post_ds = pre_ds.copy(deep=True)

    # 1. Genuine clearing cluster: 3x3 patch at rows [3:6], cols [3:6] (9 pixels = 900 m²)
    # Drop NDVI to ~0.0 (B08=1500, B04=1500 -> delta ~= 0.818 > 0.25)
    post_ds["B08"].values[3:6, 3:6] = 1500.0
    post_ds["B04"].values[3:6, 3:6] = 1500.0

    # 2. Isolated single-pixel noise spike at (1, 1)
    post_ds["B08"].values[1, 1] = 1200.0
    post_ds["B04"].values[1, 1] = 1600.0

    # 3. 2-pixel diagonal noise spike at (1, 12) and (2, 13)
    post_ds["B08"].values[1, 12] = 1200.0
    post_ds["B04"].values[1, 12] = 1600.0
    post_ds["B08"].values[2, 13] = 1200.0
    post_ds["B04"].values[2, 13] = 1600.0

    # 4. Small 2-pixel clearing at (10, 10) and (10, 11) (200 m² < 500 m² threshold)
    post_ds["B08"].values[10, 10:12] = 1500.0
    post_ds["B04"].values[10, 10:12] = 1500.0

    result = detect_canopy_loss(
        pre_ds,
        post_ds,
        ndvi_threshold=0.25,
        min_area_sqm=500.0,
    )

    assert isinstance(result, ChangeDetectionResult)

    # Verify single-pixel spike at (1, 1) was eliminated
    assert result.change_mask.values[1, 1] == False, "Isolated single-pixel noise should be removed."

    # Verify 2-pixel diagonal noise was eliminated
    assert result.change_mask.values[1, 12] == False
    assert result.change_mask.values[2, 13] == False

    # Verify sub-500 m² cluster was eliminated
    assert result.change_mask.values[10, 10] == False
    assert result.change_mask.values[10, 11] == False

    # Verify genuine 3x3 clearing cluster was preserved
    assert np.all(result.change_mask.values[3:6, 3:6] == True), "3x3 clearing cluster must be detected."

    # Verify metrics
    assert result.num_patches == 1, f"Expected exactly 1 patch, got {result.num_patches}"
    assert result.metrics["total_pixels"] == 9
    assert result.total_area_sqm == 900.0
    assert result.total_area_acres == round(900.0 / 4046.8564224, 4)

    # Verify per-patch metrics
    patch = result.metrics["patches"][0]
    assert patch["patch_id"] == 1
    assert patch["pixel_count"] == 9
    assert patch["area_sqm"] == 900.0
    assert patch["mean_delta_ndvi"] > 0.70

    # Test tuple unpacking
    change_mask, metrics = detect_canopy_loss(pre_ds, post_ds, ndvi_threshold=0.25)
    assert isinstance(change_mask, xr.DataArray)
    assert isinstance(metrics, dict)
    assert metrics["num_patches"] == 1

    # Test dictionary-style access
    assert result["total_area_sqm"] == 900.0


def test_detect_canopy_loss_cloud_masking():
    """Verify that cloud-covered pixels are masked and not mistaken for deforestation."""
    pre_ds = make_synthetic_scene(shape=(10, 10), b08_val=5000.0, b04_val=500.0, scl_val=4)
    post_ds = pre_ds.copy(deep=True)

    # In post-scene, pixel (5, 5) has cloud shadow (SCL=3) and (5, 6) has high probability cloud (SCL=9)
    # Reflectance drops significantly
    post_ds["SCL"].values[5, 5] = 3
    post_ds["SCL"].values[5, 6] = 9
    post_ds["B08"].values[5, 5:7] = 800.0
    post_ds["B04"].values[5, 5:7] = 2000.0

    result = detect_canopy_loss(pre_ds, post_ds, ndvi_threshold=0.25, dilate_pixels=1)

    # Clouded pixels and their dilation neighborhood must be masked out as invalid (not flagged as clearing)
    assert result.change_mask.values[5, 5] == False
    assert result.change_mask.values[5, 6] == False
    assert result.num_patches == 0
    assert result.total_area_sqm == 0.0


def test_no_change_scenario():
    """Verify that when no clearing occurred, zero patches and zero area are returned."""
    pre_ds = make_synthetic_scene(shape=(10, 10), b08_val=5000.0, b04_val=500.0)
    post_ds = pre_ds.copy(deep=True)

    result = detect_canopy_loss(pre_ds, post_ds, ndvi_threshold=0.25)
    assert result.num_patches == 0
    assert result.total_area_sqm == 0.0
    assert result.total_area_acres == 0.0
    assert result.metrics["patches"] == []
    assert not np.any(result.change_mask.values)


@pytest.mark.integration
def test_real_scene_bi_temporal_change_detection():
    """Integration test: Run bi-temporal change detection on real Sentinel-2 passes over Thirthahalli."""
    # Find two clear-sky passes in Jan/Feb 2024
    items_pre = search_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="2024-01-20",
        end_date="2024-01-22",
        max_cloud=10,
    )
    items_post = search_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="2024-02-04",
        end_date="2024-02-06",
        max_cloud=10,
    )
    assert len(items_pre) > 0 and len(items_post) > 0

    target_pre = [i for i in items_pre if "T43PER" in i.id][0]
    target_post = [i for i in items_post if "T43PER" in i.id][0]

    # Load 10m bands for small subslice
    bands = ["B04", "B08", "SCL"]
    ds_pre = load_stac_dataset([target_pre], bands=bands, crs="EPSG:32643", resolution=10, bbox=DEFAULT_AOI_BBOX)
    ds_post = load_stac_dataset([target_post], bands=bands, crs="EPSG:32643", resolution=10, bbox=DEFAULT_AOI_BBOX)

    sub_pre = ds_pre.isel(time=0, y=slice(500, 600), x=slice(500, 600)).compute()
    sub_post = ds_post.isel(time=0, y=slice(500, 600), x=slice(500, 600)).compute()

    result = detect_canopy_loss(
        sub_pre,
        sub_post,
        ndvi_threshold=0.25,
        min_area_sqm=500.0,
    )

    assert isinstance(result, ChangeDetectionResult)
    assert result.change_mask.shape == (100, 100)
    assert result.delta_ndvi.shape == (100, 100)
    assert result.labeled_mask.shape == (100, 100)
    assert result.total_area_sqm >= 0.0
    assert result.num_patches >= 0
