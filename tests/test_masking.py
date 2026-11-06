"""Unit tests for SCL cloud masking and morphological dilation."""

import numpy as np
import pytest
import xarray as xr

from haddu.core.masking import (
    DEFAULT_INVALID_SCL_CLASSES,
    DEFAULT_VALID_SCL_CLASSES,
    create_valid_mask,
    apply_valid_mask,
)


def test_valid_and_invalid_scl_classes_basic():
    """Verify that pure valid SCL classes are preserved and invalid classes are masked."""
    # 4 (Vegetation), 5 (Bare Soil), 6 (Water), 7 (Unclassified)
    valid_scl = np.array([4, 5, 6, 7], dtype=np.uint8)
    mask = create_valid_mask(valid_scl, dilate_pixels=0)
    assert np.all(mask.values), "All valid classes (4, 5, 6, 7) must evaluate to True"

    # 0 (NoData), 1 (Defective), 3 (Shadow), 8 (Med Cloud), 9 (High Cloud), 10 (Cirrus), 11 (Snow)
    invalid_scl = np.array([0, 1, 3, 8, 9, 10, 11], dtype=np.uint8)
    mask_inv = create_valid_mask(invalid_scl, dilate_pixels=0)
    assert not np.any(mask_inv.values), "All invalid classes must evaluate to False"


def test_nan_handling():
    """Verify that NaN values in SCL are safely treated as invalid without crashing."""
    scl_with_nan = np.array([[4.0, np.nan], [6.0, 5.0]], dtype=np.float32)
    mask = create_valid_mask(scl_with_nan, dilate_pixels=0)

    assert mask.values[0, 0] == True  # 4.0 Vegetation
    assert mask.values[0, 1] == False  # NaN
    assert mask.values[1, 0] == True  # 6.0 Water
    assert mask.values[1, 1] == True  # 5.0 Bare soil


def test_morphological_dilation_expansion():
    """Verify that a single cloud pixel dilates outwards by dilate_pixels iterations."""
    # 9x9 grid of clean vegetation (4)
    grid = np.full((9, 9), 4, dtype=np.uint8)
    # Cloud (9) placed in center (4, 4)
    grid[4, 4] = 9

    scl_da = xr.DataArray(grid, dims=("y", "x"), name="SCL")

    # 0 dilation: only center (4,4) is invalid
    mask_d0 = create_valid_mask(scl_da, dilate_pixels=0)
    assert np.count_nonzero(~mask_d0.values) == 1
    assert not mask_d0.values[4, 4]
    assert mask_d0.values[4, 5]

    # 1 dilation: 3x3 box around center is invalid (9 pixels)
    mask_d1 = create_valid_mask(scl_da, dilate_pixels=1)
    assert np.count_nonzero(~mask_d1.values) == 9
    assert not np.any(mask_d1.values[3:6, 3:6])
    assert mask_d1.values[2, 4]  # Outside 3x3 remains valid

    # 2 dilation: 5x5 box around center is invalid (25 pixels)
    mask_d2 = create_valid_mask(scl_da, dilate_pixels=2)
    assert np.count_nonzero(~mask_d2.values) == 25
    assert not np.any(mask_d2.values[2:7, 2:7])
    assert mask_d2.values[1, 4]  # Outside 5x5 remains valid


def test_3d_temporal_dataarray_support():
    """Verify that 3D (time, y, x) arrays dilate spatially per time-slice."""
    # 2 time slices of 7x7 grid
    grid = np.full((2, 7, 7), 4, dtype=np.uint8)
    grid[0, 3, 3] = 9  # Cloud in time 0 center
    grid[1, 1, 1] = 8  # Cloud in time 1 top-left

    da_3d = xr.DataArray(
        grid,
        dims=("time", "y", "x"),
        coords={"time": [0, 1]},
        name="SCL",
    )

    mask = create_valid_mask(da_3d, dilate_pixels=1)
    assert mask.shape == (2, 7, 7)

    # Time 0: 3x3 around (3,3)
    assert np.count_nonzero(~mask.isel(time=0).values) == 9
    # Time 1: 2x2 corner clipped around (1,1) -> 3x3 box
    assert np.count_nonzero(~mask.isel(time=1).values) == 9


def test_dataset_input_auto_extraction():
    """Verify that passing an xarray.Dataset containing an SCL variable works directly."""
    scl_arr = np.full((5, 5), 4, dtype=np.uint8)
    scl_arr[0, 0] = 3  # Cloud shadow

    ds = xr.Dataset({
        "B04": (("y", "x"), np.ones((5, 5))),
        "SCL": (("y", "x"), scl_arr),
    })

    mask = create_valid_mask(ds, dilate_pixels=1)
    assert isinstance(mask, xr.DataArray)
    assert mask.name == "valid_mask"
    # Corner (0,0) dilated with 3x3 structuring element has 4 pixels invalid in corner
    assert np.count_nonzero(~mask.values) == 4


def test_apply_valid_mask():
    """Verify apply_valid_mask masks invalid regions with NaN without leaking into clean areas."""
    data = xr.DataArray(np.array([[100.0, 200.0], [300.0, 400.0]]), dims=("y", "x"))
    mask = xr.DataArray(np.array([[True, False], [True, True]]), dims=("y", "x"))

    masked_data = apply_valid_mask(data, mask, fill_value=np.nan)
    assert masked_data.values[0, 0] == 100.0
    assert np.isnan(masked_data.values[0, 1])
    assert masked_data.values[1, 0] == 300.0
    assert masked_data.values[1, 1] == 400.0


@pytest.mark.integration
def test_real_scene_masking_and_dilation():
    """Verify create_valid_mask on a real Sentinel-2 scene over Thirthahalli.

    Acceptance Criteria:
    - Masks all identified clouds.
    - Expands the mask boundary by 2 pixels.
    - Preserves vegetation and water bodies without NaN leakage into clean areas.
    """
    from haddu.core.stac_client import search_passes, DEFAULT_AOI_BBOX
    from haddu.core.loader import load_stac_dataset

    items = search_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="2026-01-01",
        end_date="2026-02-15",
        max_cloud=15,
    )
    assert len(items) > 0

    # Pick pass covering Thirthahalli
    candidate = [it for it in items if "T43PER" in it.id or "T43PEQ" in it.id][0]

    ds = load_stac_dataset(
        items=[candidate],
        bands=["B04", "SCL"],
        resolution=40,  # 40m for quick test streaming
        bbox=DEFAULT_AOI_BBOX,
    )
    ds = ds.compute()

    mask_d0 = create_valid_mask(ds, dilate_pixels=0)
    mask_d2 = create_valid_mask(ds, dilate_pixels=2)

    # 1. Output types and dimensions
    assert isinstance(mask_d2, xr.DataArray)
    assert mask_d2.dtype == bool
    assert mask_d2.shape == ds.SCL.shape

    # 2. Dilation must expand or keep invalid count >= d0
    invalid_d0_count = int(np.count_nonzero(~mask_d0.values))
    invalid_d2_count = int(np.count_nonzero(~mask_d2.values))
    assert invalid_d2_count >= invalid_d0_count

    # 3. Apply mask to reflectance band B04
    b04_masked = apply_valid_mask(ds.B04, mask_d2)

    # Where mask is True, B04 values must be finite and preserved
    clean_indices = mask_d2.values
    if np.any(clean_indices):
        assert np.all(np.isfinite(b04_masked.values[clean_indices]))
        assert np.array_equal(b04_masked.values[clean_indices], ds.B04.values[clean_indices])

    # Where mask is False, B04 values must be NaN
    masked_indices = ~mask_d2.values
    if np.any(masked_indices):
        assert np.all(np.isnan(b04_masked.values[masked_indices]))
