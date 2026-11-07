"""Polarimetric radar disturbance detection engine for Sentinel-1 SAR imagery.

Identifies sudden drops in cross-polarized (VH) volume scattering caused by
monsoon-season tree felling and forest clearing in the Western Ghats.
Applies spatial speckle reduction filtering (median or Lee filter) to reject
inherent radar speckle noise, followed by morphological opening and minimum
cluster area thresholding to isolate genuine clearing events.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import xarray as xr

from haddu.core.change_detector import (
    DEFAULT_MIN_AREA_SQM,
    DEFAULT_TARGET_CRS_EPSG,
    SQM_PER_ACRE,
    SpatialAlignmentError,
    _get_crs_epsg,
)

logger = logging.getLogger(__name__)

# Default radar backscatter drop threshold in dB
# Forest canopy in Western Ghats typically has VH around -11 to -14 dB.
# Clearing reduces volume scattering, dropping VH by 2.5 dB or more.
DEFAULT_VH_DROP_DB = 2.5
DEFAULT_SPECKLE_FILTER = "median"
DEFAULT_FILTER_SIZE = 3


@dataclass
class SARChangeDetectionResult:
    """Encapsulates the outputs and spatial metrics of SAR deforestation detection.

    Supports tuple unpacking:
        disturbance_mask, metrics = detect_sar_deforestation(...)
    """

    disturbance_mask: xr.DataArray
    delta_vh: xr.DataArray
    filtered_delta_vh: xr.DataArray
    labeled_mask: xr.DataArray
    metrics: dict[str, Any]

    def __iter__(self):
        """Enable tuple unpacking: mask, metrics = result."""
        yield self.disturbance_mask
        yield self.metrics

    @property
    def change_mask(self) -> xr.DataArray:
        """Alias for disturbance_mask to match vectorizer and optical change detector."""
        return self.disturbance_mask

    @property
    def total_area_sqm(self) -> float:
        """Total detected disturbance area in square meters."""
        return float(self.metrics.get("total_area_sqm", 0.0))

    @property
    def total_area_acres(self) -> float:
        """Total detected disturbance area in acres."""
        return float(self.metrics.get("total_area_acres", 0.0))

    @property
    def total_pixels(self) -> int:
        """Total number of disturbance pixels detected."""
        return int(self.metrics.get("total_pixels", 0))

    @property
    def num_patches(self) -> int:
        """Number of contiguous disturbance patches detected."""
        return int(self.metrics.get("num_patches", 0))

    def __getitem__(self, key: str) -> Any:
        if key in ("disturbance_mask", "change_mask"):
            return self.disturbance_mask
        if key == "delta_vh":
            return self.delta_vh
        if key == "filtered_delta_vh":
            return self.filtered_delta_vh
        if key == "labeled_mask":
            return self.labeled_mask
        if key in self.metrics:
            return self.metrics[key]
        raise KeyError(f"Key '{key}' not found in SARChangeDetectionResult.")


def _extract_vh_band(scene: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """Extract and validate the cross-polarized VH backscatter band.

    Args:
        scene: Input xarray Dataset or DataArray.

    Returns:
        2D xarray DataArray with (y, x) spatial dimensions.

    Raises:
        KeyError: If no VH backscatter variable is present in the dataset.
        TypeError: If scene is not an xarray Dataset or DataArray.
    """
    if isinstance(scene, xr.DataArray):
        da = scene
    elif isinstance(scene, xr.Dataset):
        if "vh" in scene.data_vars:
            da = scene["vh"]
        elif "VH" in scene.data_vars:
            da = scene["VH"]
        else:
            candidates = [v for v in scene.data_vars if "vh" in v.lower()]
            if candidates:
                da = scene[candidates[0]]
            else:
                raise KeyError(
                    f"No VH backscatter band found in dataset. Available: {list(scene.data_vars.keys())}"
                )
    else:
        raise TypeError(f"Expected xr.Dataset or xr.DataArray, got {type(scene).__name__}")

    # Squeeze out temporal dimension if present
    if "time" in da.dims:
        if da.sizes["time"] == 1:
            da = da.squeeze("time", drop=True)
        else:
            da = da.isel(time=-1)

    return da


def align_sar_rasters(
    pre: xr.Dataset | xr.DataArray,
    post: xr.Dataset | xr.DataArray,
    check_spatial: bool = True,
    expected_epsg: int = DEFAULT_TARGET_CRS_EPSG,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Align pre- and post-event SAR VH rasters and verify spatial compatibility.

    Args:
        pre: Pre-disturbance SAR Dataset or DataArray.
        post: Post-disturbance SAR Dataset or DataArray.
        check_spatial: Whether to strictly verify spatial alignment and CRS.
        expected_epsg: Expected Coordinate Reference System EPSG code (default: 32643).

    Returns:
        Tuple of aligned (vh_pre, vh_post) DataArrays.

    Raises:
        SpatialAlignmentError: If CRS, coordinates, or spatial grids do not align.
    """
    vh_pre = _extract_vh_band(pre)
    vh_post = _extract_vh_band(post)

    if check_spatial:
        pre_epsg = _get_crs_epsg(vh_pre)
        post_epsg = _get_crs_epsg(vh_post)

        if pre_epsg is not None and pre_epsg != expected_epsg:
            raise SpatialAlignmentError(
                f"Pre-scene CRS EPSG:{pre_epsg} does not match expected EPSG:{expected_epsg}."
            )
        if post_epsg is not None and post_epsg != expected_epsg:
            raise SpatialAlignmentError(
                f"Post-scene CRS EPSG:{post_epsg} does not match expected EPSG:{expected_epsg}."
            )
        if pre_epsg is not None and post_epsg is not None and pre_epsg != post_epsg:
            raise SpatialAlignmentError(
                f"CRS mismatch between pre-scene (EPSG:{pre_epsg}) and post-scene (EPSG:{post_epsg})."
            )

        if "x" not in vh_pre.coords or "y" not in vh_pre.coords:
            raise SpatialAlignmentError("Pre-scene is missing spatial coordinate 'x' or 'y'.")
        if "x" not in vh_post.coords or "y" not in vh_post.coords:
            raise SpatialAlignmentError("Post-scene is missing spatial coordinate 'x' or 'y'.")

        # Verify resolution compatibility
        if len(vh_pre.coords["x"]) > 1 and len(vh_post.coords["x"]) > 1:
            res_pre = abs(float(vh_pre.coords["x"][1] - vh_pre.coords["x"][0]))
            res_post = abs(float(vh_post.coords["x"][1] - vh_post.coords["x"][0]))
            if abs(res_pre - res_post) > 1.0:
                raise SpatialAlignmentError(
                    f"Resolution mismatch: pre has {res_pre:.1f}m, post has {res_post:.1f}m."
                )

        if vh_pre.shape != vh_post.shape:
            try:
                vh_pre, vh_post = xr.align(vh_pre, vh_post, join="inner")
                if vh_pre.size == 0 or vh_post.size == 0:
                    raise SpatialAlignmentError("Pre- and post-scenes have no overlapping spatial extent.")
            except Exception as err:
                raise SpatialAlignmentError(
                    f"Spatial coordinate mismatch between pre and post: {err}"
                ) from err
        else:
            if not np.allclose(vh_pre.coords["x"].values, vh_post.coords["x"].values, atol=0.5):
                raise SpatialAlignmentError("Pre- and post-scene 'x' coordinates do not align spatially.")
            if not np.allclose(vh_pre.coords["y"].values, vh_post.coords["y"].values, atol=0.5):
                raise SpatialAlignmentError("Pre- and post-scene 'y' coordinates do not align spatially.")

    return vh_pre, vh_post


def _lee_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply Lee speckle filter over a 2D radar array.

    Args:
        image: 2D numpy array.
        size: Moving window size (default: 3).

    Returns:
        Filtered 2D numpy array.
    """
    img = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(img)
    if not np.any(valid):
        return image.copy()

    safe_img = np.where(valid, img, np.nanmean(img[valid]))

    # Local mean and variance
    mean = ndi.uniform_filter(safe_img, size=size, mode="reflect")
    mean_sq = ndi.uniform_filter(safe_img**2, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean**2, 0.0)

    # Estimate noise variance from local variances
    noise_var = float(np.nanvar(safe_img) * 0.25)
    weights = var / (var + noise_var + 1e-6)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = mean + weights * (safe_img - mean)
    return np.where(valid, filtered, np.nan)


def _apply_speckle_filter(
    arr: np.ndarray,
    method: str = "median",
    size: int = 3,
) -> np.ndarray:
    """Apply speckle reduction filtering over backscatter difference matrix.

    Args:
        arr: 2D numpy array of backscatter difference values.
        method: Filter algorithm ('median', 'lee', or 'none').
        size: Filter window size (default: 3).

    Returns:
        Filtered 2D numpy array.
    """
    if method == "none" or method is None:
        return arr.copy()

    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr.copy()

    if method == "median":
        # Fill NaNs temporarily with median of valid data to prevent boundary distortion
        fill_val = float(np.nanmedian(arr[valid]))
        safe_arr = np.where(valid, arr, fill_val)
        filtered = ndi.median_filter(safe_arr, size=size, mode="reflect")
        return np.where(valid, filtered, np.nan)
    elif method == "lee":
        return _lee_filter(arr, size=size)
    else:
        logger.warning(f"Unknown filter method '{method}', defaulting to 3x3 median filter.")
        fill_val = float(np.nanmedian(arr[valid]))
        safe_arr = np.where(valid, arr, fill_val)
        filtered = ndi.median_filter(safe_arr, size=size, mode="reflect")
        return np.where(valid, filtered, np.nan)


def detect_sar_deforestation(
    pre_sar_ds: xr.Dataset | xr.DataArray,
    post_sar_ds: xr.Dataset | xr.DataArray,
    vh_drop_db: float = DEFAULT_VH_DROP_DB,
    min_area_sqm: float = DEFAULT_MIN_AREA_SQM,
    filter_method: str = DEFAULT_SPECKLE_FILTER,
    filter_size: int = DEFAULT_FILTER_SIZE,
    check_spatial: bool = True,
) -> SARChangeDetectionResult:
    r"""Detect forest clearing events by computing polarimetric VH backscatter drop.

    Calculates the backscatter drop:
        \Delta\text{VH}_{\text{dB}} = \text{VH}_{\text{pre}} - \text{VH}_{\text{post}}

    Applies speckle reduction filtering (e.g. 3x3 median filter or Lee filter),
    flags pixels where \Delta\text{VH}_{\text{dB}} > \text{vh_drop_db}, and filters
    out isolated speckle artifacts smaller than min_area_sqm via morphological opening
    and connected-component labeling.

    Args:
        pre_sar_ds: Baseline pre-event SAR pass (Dataset or DataArray with VH channel in dB).
        post_sar_ds: Post-event SAR pass (Dataset or DataArray with VH channel in dB).
        vh_drop_db: Minimum backscatter drop in decibels to flag tree felling (default: 2.5 dB).
        min_area_sqm: Minimum contiguous patch area in square meters (default: 500 m²).
        filter_method: Speckle reduction filter algorithm ('median', 'lee', or 'none').
        filter_size: Window size for speckle filter (default: 3 for 3x3 window).
        check_spatial: Whether to verify spatial grid alignment and CRS (default: True).

    Returns:
        SARChangeDetectionResult containing:
        - disturbance_mask: Cleaned boolean DataArray (True = detected clearing cluster)
        - delta_vh: Unfiltered difference DataArray (VH_pre - VH_post)
        - filtered_delta_vh: Speckle-filtered difference DataArray
        - labeled_mask: Integer DataArray of connected component IDs
        - metrics: Dictionary of comprehensive spatial metrics and per-patch attributes

    Raises:
        SpatialAlignmentError: If scenes do not align spatially or CRS mismatches.
    """
    # 1. Align pre- and post-scenes spatially
    vh_pre, vh_post = align_sar_rasters(pre_sar_ds, post_sar_ds, check_spatial=check_spatial)

    pre_vals = vh_pre.values if hasattr(vh_pre, "values") else np.asarray(vh_pre)
    post_vals = vh_post.values if hasattr(vh_post, "values") else np.asarray(vh_post)

    # 2. Calculate backscatter difference: delta_vh = VH_pre - VH_post
    valid_mask = np.isfinite(pre_vals) & np.isfinite(post_vals)
    delta_vals = np.where(valid_mask, pre_vals - post_vals, np.nan)
    initial_candidates = (delta_vals > vh_drop_db) & valid_mask

    # 3. Apply speckle reduction filtering
    filtered_delta = _apply_speckle_filter(
        delta_vals,
        method=filter_method,
        size=filter_size,
    )

    # 4. Flag pixels where backscatter drop exceeds threshold
    filtered_candidates = (filtered_delta > vh_drop_db) & valid_mask

    # 5. Morphological cleanup: discard isolated speckle noise smaller than min_area_sqm
    # 5a. Connected component labeling on filtered candidates
    structuring_element = np.ones((3, 3), dtype=bool)
    labeled_patches, num_initial = ndi.label(filtered_candidates, structure=structuring_element)

    # 5b. Determine pixel resolution and area (e.g. 400 m² for 20m SAR, 100 m² for 10m SAR)
    pixel_area_sqm = 400.0  # fallback default for 20m resolution
    if (
        "x" in vh_pre.coords
        and "y" in vh_pre.coords
        and len(vh_pre.coords["x"]) > 1
        and len(vh_pre.coords["y"]) > 1
    ):
        res_x = abs(float(vh_pre.coords["x"][1] - vh_pre.coords["x"][0]))
        res_y = abs(float(vh_pre.coords["y"][1] - vh_pre.coords["y"][0]))
        pixel_area_sqm = res_x * res_y

    min_pixels = max(1, math.ceil(min_area_sqm / pixel_area_sqm))

    if num_initial > 0:
        patch_counts = np.bincount(labeled_patches.ravel())
        valid_patch_ids = np.where(patch_counts >= min_pixels)[0]
        # Remove background label 0
        valid_patch_ids = valid_patch_ids[valid_patch_ids > 0]
        cleaned_binary = np.isin(labeled_patches, valid_patch_ids)
    else:
        cleaned_binary = np.zeros_like(filtered_candidates, dtype=bool)

    # Re-label final patches sequentially
    final_labeled, num_patches = ndi.label(cleaned_binary, structure=structuring_element)

    # 6. Extract spatial patch metrics
    patches_metrics_list: list[dict[str, Any]] = []
    total_change_pixels = int(np.count_nonzero(cleaned_binary))

    for patch_id in range(1, num_patches + 1):
        patch_sel = final_labeled == patch_id
        p_count = int(np.count_nonzero(patch_sel))
        p_area_sqm = float(p_count * pixel_area_sqm)
        p_area_acres = float(p_area_sqm / SQM_PER_ACRE)

        patch_deltas = filtered_delta[patch_sel]
        patch_deltas_clean = patch_deltas[np.isfinite(patch_deltas)]
        mean_delta = float(np.mean(patch_deltas_clean)) if patch_deltas_clean.size > 0 else 0.0
        max_delta = float(np.max(patch_deltas_clean)) if patch_deltas_clean.size > 0 else 0.0

        patches_metrics_list.append(
            {
                "patch_id": patch_id,
                "pixel_count": p_count,
                "area_sqm": round(p_area_sqm, 2),
                "area_acres": round(p_area_acres, 4),
                "mean_delta_vh_db": round(mean_delta, 4),
                "max_delta_vh_db": round(max_delta, 4),
            }
        )

    total_area_sqm = float(total_change_pixels * pixel_area_sqm)
    total_area_acres = float(total_area_sqm / SQM_PER_ACRE)

    metrics_summary: dict[str, Any] = {
        "num_patches": num_patches,
        "total_pixels": total_change_pixels,
        "total_area_sqm": round(total_area_sqm, 2),
        "total_area_acres": round(total_area_acres, 4),
        "vh_drop_db": vh_drop_db,
        "min_area_sqm": min_area_sqm,
        "min_pixels": min_pixels,
        "pixel_area_sqm": pixel_area_sqm,
        "filter_method": filter_method,
        "filter_size": filter_size,
        "initial_candidates_pixels": int(np.count_nonzero(initial_candidates)),
        "speckle_filtered_candidates": int(np.count_nonzero(filtered_candidates)),
        "noise_pixels_filtered": int(np.count_nonzero(initial_candidates) - total_change_pixels),
        "patches": patches_metrics_list,
    }

    logger.info(
        f"SAR deforestation detection completed: {num_patches} patches found, "
        f"{total_area_sqm:,.2f} m² ({total_area_acres:,.2f} acres) total disturbance, "
        f"{metrics_summary['noise_pixels_filtered']} noise pixels filtered."
    )

    # 7. Build output xarray DataArrays preserving spatial coordinates
    spatial_coords = {dim: vh_pre.coords[dim] for dim in ["y", "x"] if dim in vh_pre.coords}
    if "spatial_ref" in vh_pre.coords:
        spatial_coords["spatial_ref"] = vh_pre.coords["spatial_ref"]

    disturbance_mask_da = xr.DataArray(
        cleaned_binary,
        coords=spatial_coords,
        dims=["y", "x"],
        name="sar_disturbance_mask",
        attrs={
            "long_name": "SAR Radar Backscatter Disturbance Mask",
            "vh_drop_db": vh_drop_db,
            "min_area_sqm": min_area_sqm,
            "num_patches": num_patches,
            "total_area_sqm": total_area_sqm,
            "total_area_acres": total_area_acres,
        },
    )

    delta_vh_da = xr.DataArray(
        delta_vals,
        coords=spatial_coords,
        dims=["y", "x"],
        name="delta_vh",
        attrs={
            "long_name": "SAR Bi-temporal VH Drop (Pre - Post)",
            "formula": "vh_pre - vh_post",
            "units": "dB",
            "vh_drop_db": vh_drop_db,
        },
    )

    filtered_delta_vh_da = xr.DataArray(
        filtered_delta,
        coords=spatial_coords,
        dims=["y", "x"],
        name="filtered_delta_vh",
        attrs={
            "long_name": f"Speckle-Filtered SAR VH Drop ({filter_method} {filter_size}x{filter_size})",
            "units": "dB",
            "filter_method": filter_method,
            "filter_size": filter_size,
        },
    )

    labeled_mask_da = xr.DataArray(
        final_labeled,
        coords=spatial_coords,
        dims=["y", "x"],
        name="sar_disturbance_labels",
        attrs={
            "long_name": "SAR Disturbance Cluster Labels",
            "num_patches": num_patches,
        },
    )

    return SARChangeDetectionResult(
        disturbance_mask=disturbance_mask_da,
        delta_vh=delta_vh_da,
        filtered_delta_vh=filtered_delta_vh_da,
        labeled_mask=labeled_mask_da,
        metrics=metrics_summary,
    )
