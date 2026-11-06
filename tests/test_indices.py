"""Unit and integration tests for vectorized spectral indices (NDVI, NDRE, NBR, dNBR)."""

import numpy as np
import pytest
import xarray as xr

from haddu.core.indices import (
    align_to_grid,
    calculate_ndvi,
    calculate_ndre,
    calculate_nbr,
    calculate_dnbr,
)
from haddu.core.masking import create_valid_mask
from haddu.core.stac_client import DEFAULT_AOI_BBOX, search_passes
from haddu.core.loader import load_stac_dataset, DEFAULT_TARGET_BANDS


@pytest.fixture
def synthetic_spectra_dataset():
    """Create a synthetic 4x4 dataset with realistic spectral signatures.

    Pixels:
    - (0, 0): Dense Western Ghats evergreen forest (High NIR, Low Red, Low SWIR2, Mid RedEdge)
    - (0, 1): Water body (Tunga River) (Very low NIR, higher Red)
    - (1, 0): Bare soil / cleared land (Moderate NIR and Red, higher SWIR2)
    - (1, 1): Extreme value pixel testing clamping
    """
    coords = {"y": [40, 30, 20, 10], "x": [10, 20, 30, 40]}

    # B08 (NIR - 10m)
    b08_data = np.array([
        [5000.0, 400.0, 5200.0, 4800.0],
        [1500.0, 9999.0, 1400.0, 1600.0],
        [4900.0, 450.0, 5100.0, 4700.0],
        [1600.0, 5000.0, 1500.0, 1700.0],
    ], dtype=np.float32)

    # B04 (Red - 10m)
    b04_data = np.array([
        [450.0, 1200.0, 400.0, 500.0],
        [1400.0, 10.0, 1350.0, 1450.0],
        [420.0, 1150.0, 410.0, 490.0],
        [1500.0, 480.0, 1420.0, 1550.0],
    ], dtype=np.float32)

    # B05 (RedEdge 1 - 10m resampled grid matching B08)
    b05_data = np.array([
        [1600.0, 1500.0, 1600.0, 1550.0],
        [1500.0, 2000.0, 1500.0, 1600.0],
        [1600.0, 1450.0, 1580.0, 1520.0],
        [1450.0, 1600.0, 1480.0, 1500.0],
    ], dtype=np.float32)

    # B12 (SWIR2 - 10m resampled grid)
    b12_data = np.array([
        [350.0, 150.0, 380.0, 360.0],
        [3200.0, 100.0, 3100.0, 3300.0],
        [340.0, 160.0, 370.0, 350.0],
        [3150.0, 390.0, 3050.0, 3250.0],
    ], dtype=np.float32)

    # SCL: (0,0)=Vegetation(4), (0,1)=Water(6), (1,0)=Bare(5), (3,3)=Cloud(9)
    scl_data = np.array([
        [4, 6, 4, 4],
        [5, 4, 5, 5],
        [4, 6, 4, 4],
        [5, 4, 5, 9],
    ], dtype=np.int32)

    return xr.Dataset(
        data_vars={
            "B08": (["y", "x"], b08_data),
            "B04": (["y", "x"], b04_data),
            "B05": (["y", "x"], b05_data),
            "B12": (["y", "x"], b12_data),
            "SCL": (["y", "x"], scl_data),
        },
        coords=coords,
    )


def test_calculate_ndvi_canopy_and_water(synthetic_spectra_dataset):
    """Verify NDVI returns > 0.70 for dense forest and negative for water bodies."""
    ds = synthetic_spectra_dataset
    ndvi = calculate_ndvi(ds)

    assert isinstance(ndvi, xr.DataArray)
    assert ndvi.name == "ndvi"
    assert ndvi.shape == ds["B08"].shape

    # Pixel (0, 0): Forest (B08=5000, B04=450) -> (5000 - 450) / (5000 + 450) = 4550 / 5450 ~= 0.835
    forest_ndvi = float(ndvi.values[0, 0])
    assert forest_ndvi > 0.70, f"Expected dense evergreen forest NDVI > 0.70, got {forest_ndvi}"

    # Pixel (0, 1): Water (B08=400, B04=1200) -> (400 - 1200) / (400 + 1200) = -800 / 1600 = -0.5
    water_ndvi = float(ndvi.values[0, 1])
    assert water_ndvi < 0.0, f"Expected water body NDVI < 0, got {water_ndvi}"

    # Verify all non-nan values are strictly within [-1.0, 1.0]
    valid_values = ndvi.values[~np.isnan(ndvi.values)]
    assert np.all(valid_values >= -1.0)
    assert np.all(valid_values <= 1.0)


def test_calculate_ndre_grid_alignment(synthetic_spectra_dataset):
    """Verify NDRE computation and automatic resampling/alignment of 20m B05 to 10m B08."""
    ds = synthetic_spectra_dataset

    # Standard call with aligned dataset
    ndre = calculate_ndre(ds)
    assert isinstance(ndre, xr.DataArray)
    assert ndre.name == "ndre"
    assert ndre.shape == (4, 4)

    # Evergreen canopy should have high NDRE (> 0.50)
    forest_ndre = float(ndre.values[0, 0])
    assert forest_ndre > 0.50, f"Expected high NDRE for dense forest, got {forest_ndre}"

    # Test explicit 20m B05 DataArray (2x2) with different coordinates
    coords_20m = {"y": [35, 15], "x": [15, 35]}
    b05_20m = xr.DataArray(
        np.array([[1600.0, 1500.0], [1400.0, 1550.0]], dtype=np.float32),
        coords=coords_20m,
        dims=["y", "x"],
    )
    assert b05_20m.shape == (2, 2)

    ndre_resampled = calculate_ndre(ds, b05=b05_20m)
    assert ndre_resampled.shape == (4, 4)
    assert not np.isnan(ndre_resampled.values[0, 0])
    assert float(ndre_resampled.values[0, 0]) > 0.50

    # All values within [-1.0, 1.0]
    valid_values = ndre_resampled.values[~np.isnan(ndre_resampled.values)]
    assert np.all(valid_values >= -1.0)
    assert np.all(valid_values <= 1.0)


def test_calculate_nbr_and_dnbr(synthetic_spectra_dataset):
    """Verify NBR and dNBR calculation over healthy forest and cleared land."""
    pre_ds = synthetic_spectra_dataset
    nbr_pre = calculate_nbr(pre_ds)

    assert isinstance(nbr_pre, xr.DataArray)
    assert nbr_pre.name == "nbr"

    # Pixel (0, 0): Pre-forest (B08=5000, B12=350) -> NBR ~= (5000 - 350) / 5350 ~= 0.869
    assert float(nbr_pre.values[0, 0]) > 0.80

    # Simulate post-clearing dataset: pixel (0, 0) was deforested (B08 dropped to 1200, B12 rose to 3200)
    post_ds = pre_ds.copy(deep=True)
    post_b08 = post_ds["B08"].values.copy()
    post_b12 = post_ds["B12"].values.copy()
    post_b08[0, 0] = 1200.0
    post_b12[0, 0] = 3200.0
    post_ds["B08"].values = post_b08
    post_ds["B12"].values = post_b12

    nbr_post = calculate_nbr(post_ds)
    # Post NBR for cleared pixel should be negative
    assert float(nbr_post.values[0, 0]) < 0.0

    # dNBR = pre - post
    dnbr = calculate_dnbr(pre_ds, post_ds)
    assert isinstance(dnbr, xr.DataArray)
    assert dnbr.name == "dnbr"

    # Clearing should produce high positive dNBR (> 0.8)
    clearing_dnbr = float(dnbr.values[0, 0])
    assert clearing_dnbr > 0.8, f"Expected high positive dNBR for clearing, got {clearing_dnbr}"

    # Pixel (1, 1) had no change -> dNBR near 0
    unchanged_dnbr = float(dnbr.values[1, 1])
    assert abs(unchanged_dnbr) < 0.01


def test_indices_valid_mask_application(synthetic_spectra_dataset):
    """Verify that validity masks mask invalid pixels with NaN."""
    ds = synthetic_spectra_dataset

    # Create mask where pixel (3, 3) (cloud) and dilation neighbors are False
    valid_mask = create_valid_mask(ds["SCL"], dilate_pixels=1)

    # 1. Pass mask explicitly
    ndvi_masked = calculate_ndvi(ds, valid_mask=valid_mask)
    assert np.isnan(ndvi_masked.values[3, 3])
    assert not np.isnan(ndvi_masked.values[0, 0])

    # 2. Attach mask to dataset coordinates and test automatic pickup
    ds_with_mask = ds.copy()
    ds_with_mask["valid_mask"] = valid_mask
    ndre_masked = calculate_ndre(ds_with_mask)
    assert np.isnan(ndre_masked.values[3, 3])
    assert not np.isnan(ndre_masked.values[0, 0])


def test_align_to_grid_uncoordinated():
    """Verify align_to_grid handles uncoordinated/raw synthetic arrays via fallback."""
    b08 = xr.DataArray(np.ones((6, 6)), dims=["y", "x"])
    b05 = xr.DataArray(np.array([[1, 2], [3, 4]]), dims=["y", "x"])

    aligned = align_to_grid(b05, b08)
    assert aligned.shape == (6, 6)
    assert aligned.dims == b08.dims


@pytest.mark.integration
def test_indices_on_real_sentinel2_scene():
    """Integration test: Stream real Sentinel-2 pass and verify spectral indices over Thirthahalli."""
    items = search_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="2024-01-20",
        end_date="2024-01-22",
        max_cloud=15,
    )
    assert len(items) > 0

    # Prioritize T43PER tile covering full Thirthahalli Taluk
    per_items = [i for i in items if "T43PER" in i.id]
    target_item = per_items[0] if per_items else items[0]

    # Stream target bands including B05
    ds = load_stac_dataset(
        items=[target_item],
        bands=DEFAULT_TARGET_BANDS,
        crs="EPSG:32643",
        resolution=10,
        bbox=DEFAULT_AOI_BBOX,
        chunks={"x": 2048, "y": 2048},
    )

    # Compute a small spatial subslice within valid taluk boundaries
    sub_ds = ds.isel(time=0, y=slice(500, 600), x=slice(500, 600)).compute()

    # Create validity mask
    mask = create_valid_mask(sub_ds["SCL"], dilate_pixels=2)

    ndvi = calculate_ndvi(sub_ds, valid_mask=mask)
    ndre = calculate_ndre(sub_ds, valid_mask=mask)
    nbr = calculate_nbr(sub_ds, valid_mask=mask)

    assert isinstance(ndvi, xr.DataArray)
    assert isinstance(ndre, xr.DataArray)
    assert isinstance(nbr, xr.DataArray)

    # Check non-nan ranges
    ndvi_vals = ndvi.values[~np.isnan(ndvi.values)]
    assert len(ndvi_vals) > 0, "Expected valid NDVI values in test subslice."
    assert np.all(ndvi_vals >= -1.0) and np.all(ndvi_vals <= 1.0)

    # Western Ghats canopy in dry season should have robust positive vegetation index
    assert np.max(ndvi_vals) > 0.55
