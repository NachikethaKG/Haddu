"""Cloud, shadow, and invalid pixel masking for Sentinel-2 Scene Classification Layer (SCL).

Provides morphological dilation of cloud and cloud-shadow boundaries to prevent
edge-artifact false positives during spectral change detection.
"""

import logging
from typing import Sequence, Union
import numpy as np
import scipy.ndimage as ndi
import xarray as xr

logger = logging.getLogger(__name__)

# SCL classes deemed invalid for optical surface reflectance analytics
# 0: No Data, 1: Saturated/Defective, 3: Cloud Shadows,
# 8: Cloud Medium Probability, 9: Cloud High Probability, 10: Thin Cirrus, 11: Snow/Ice
DEFAULT_INVALID_SCL_CLASSES = (0, 1, 3, 8, 9, 10, 11)

# SCL classes representing clean, usable ground surface
# 4: Vegetation, 5: Not-vegetated (bare soil), 6: Water, 7: Unclassified
DEFAULT_VALID_SCL_CLASSES = (4, 5, 6, 7)

DEFAULT_DILATE_PIXELS = 2


def create_valid_mask(
    scl: Union[xr.DataArray, xr.Dataset, np.ndarray],
    dilate_pixels: int = DEFAULT_DILATE_PIXELS,
    invalid_classes: Sequence[int] = DEFAULT_INVALID_SCL_CLASSES,
) -> xr.DataArray:
    """Create a boolean validity mask from Sentinel-2 SCL with morphological dilation.

    Identifies invalid pixels (clouds, shadows, defective, no-data) and expands
    the masked invalid region outwards by `dilate_pixels` using a 3x3 structuring
    element to eliminate fringe cloud false positives.

    Args:
        scl: Sentinel-2 Scene Classification Layer (DataArray, Dataset, or ndarray).
        dilate_pixels: Number of dilation iterations on invalid mask (default: 2).
        invalid_classes: Tuple/list of SCL integers to treat as invalid.

    Returns:
        Boolean xarray.DataArray of identical shape and coordinates where `True`
        denotes clean, usable surface reflectance and `False` denotes invalid/masked.
    """
    # If a full Dataset is supplied, extract the SCL variable
    if isinstance(scl, xr.Dataset):
        if "SCL" not in scl:
            raise KeyError("Supplied xarray.Dataset does not contain an 'SCL' data variable.")
        scl_da = scl["SCL"]
    elif isinstance(scl, xr.DataArray):
        scl_da = scl
    else:
        # Wrap raw numpy array in a default DataArray
        scl_da = xr.DataArray(scl)

    # Extract numpy array (handling dask or in-memory)
    scl_values = scl_da.values if hasattr(scl_da, "values") else np.asarray(scl)

    # Check for NaNs
    nan_mask = np.isnan(scl_values)

    # Safe integer conversion for class membership
    scl_int = np.nan_to_num(scl_values, nan=-1).astype(np.int32)
    invalid_mask = np.isin(scl_int, invalid_classes) | nan_mask

    # Morphological dilation on invalid (cloud/shadow) pixels
    if dilate_pixels > 0:
        structuring_element = np.ones((3, 3), dtype=bool)

        if invalid_mask.ndim == 2:
            dilated_invalid = ndi.binary_dilation(
                invalid_mask,
                structure=structuring_element,
                iterations=dilate_pixels,
            )
        elif invalid_mask.ndim == 3:
            # 3D: (time, y, x) -> dilate slice-by-slice across spatial axes
            dilated_invalid = np.empty_like(invalid_mask, dtype=bool)
            for t in range(invalid_mask.shape[0]):
                dilated_invalid[t] = ndi.binary_dilation(
                    invalid_mask[t],
                    structure=structuring_element,
                    iterations=dilate_pixels,
                )
        else:
            dilated_invalid = ndi.binary_dilation(
                invalid_mask,
                iterations=dilate_pixels,
            )
    else:
        dilated_invalid = invalid_mask

    # Valid pixels are those that are NOT in the dilated invalid mask
    valid_mask = ~dilated_invalid

    total_pixels = valid_mask.size
    valid_count = int(np.count_nonzero(valid_mask))
    valid_pct = (valid_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    logger.debug(
        f"create_valid_mask: {valid_count:,}/{total_pixels:,} "
        f"({valid_pct:.2f}%) clean surface pixels retained after {dilate_pixels}px dilation."
    )

    return xr.DataArray(
        valid_mask,
        coords=scl_da.coords,
        dims=scl_da.dims,
        name="valid_mask",
        attrs={
            "description": "Boolean mask of valid, cloud-free surface reflectance",
            "dilated_pixels": dilate_pixels,
            "invalid_scl_classes": list(invalid_classes),
        },
    )


def apply_valid_mask(
    data: Union[xr.DataArray, xr.Dataset],
    mask: xr.DataArray,
    fill_value: float = np.nan,
) -> Union[xr.DataArray, xr.Dataset]:
    """Mask out invalid pixels across an xarray DataArray or Dataset with a fill value."""
    return data.where(mask, fill_value)
