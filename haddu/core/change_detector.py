"""Bi-temporal change detection and morphological noise cleanup engine.

Detects forest canopy loss and encroachment events between two Sentinel-2 passes
by analyzing NDVI drop matrices, applying morphological filtering to eliminate
isolated single-pixel noise, and filtering connected clearing clusters by physical area.
"""

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
import xarray as xr

from haddu.core.indices import calculate_ndvi
from haddu.core.masking import create_valid_mask

logger = logging.getLogger(__name__)

# Default change detection parameters
DEFAULT_NDVI_THRESHOLD = 0.25  # Minimum NDVI decrease (pre - post) to trigger loss candidate
DEFAULT_MIN_AREA_SQM = 500.0   # Minimum contiguous clearing size in m² (e.g. 5x 10m pixels)
DEFAULT_TARGET_CRS_EPSG = 32643  # UTM Zone 43N
DEFAULT_PIXEL_RESOLUTION = 10.0   # 10 meters ground resolution
SQM_PER_ACRE = 4046.8564224


class SpatialAlignmentError(ValueError):
    """Raised when pre- and post-scene spatial grids, CRS, or resolutions do not align."""
    pass


@dataclass
class ChangeDetectionResult:
    """Encapsulates the outputs and spatial metrics of bi-temporal change detection.

    Supports tuple unpacking:
        change_mask, metrics = detect_canopy_loss(...)
    """
    change_mask: xr.DataArray
    delta_ndvi: xr.DataArray
    labeled_mask: xr.DataArray
    metrics: Dict[str, Any]

    def __iter__(self):
        """Enable tuple unpacking: mask, metrics = result."""
        yield self.change_mask
        yield self.metrics

    @property
    def total_area_sqm(self) -> float:
        return self.metrics.get("total_area_sqm", 0.0)

    @property
    def total_area_acres(self) -> float:
        return self.metrics.get("total_area_acres", 0.0)

    @property
    def num_patches(self) -> int:
        return self.metrics.get("num_patches", 0)

    def __getitem__(self, key: str) -> Any:
        if key == "change_mask":
            return self.change_mask
        if key == "delta_ndvi":
            return self.delta_ndvi
        if key == "labeled_mask":
            return self.labeled_mask
        if key in self.metrics:
            return self.metrics[key]
        raise KeyError(f"Key '{key}' not found in ChangeDetectionResult.")


def _get_crs_epsg(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[int]:
    """Extract EPSG integer code from xarray object CRS/geobox metadata if available."""
    if hasattr(ds, "geobox") and ds.geobox is not None:
        try:
            return ds.geobox.crs.epsg
        except Exception:
            pass

    if hasattr(ds, "rio") and ds.rio.crs is not None:
        try:
            return ds.rio.crs.to_epsg()
        except Exception:
            pass

    if "spatial_ref" in ds.coords:
        try:
            val = ds.spatial_ref.item()
            if isinstance(val, int) and val > 0:
                return val
            wkt = str(ds.spatial_ref.attrs.get("crs_wkt", ""))
            if "32643" in wkt or "UTM zone 43N" in wkt:
                return 32643
        except Exception:
            pass

    return None


def verify_spatial_alignment(
    pre: Union[xr.Dataset, xr.DataArray],
    post: Union[xr.Dataset, xr.DataArray],
    expected_epsg: int = DEFAULT_TARGET_CRS_EPSG,
    expected_res: float = DEFAULT_PIXEL_RESOLUTION,
) -> None:
    """Verify that pre- and post-scenes share identical spatial coordinates, CRS, and resolution.

    Args:
        pre: Pre-disturbance Dataset or DataArray.
        post: Post-disturbance Dataset or DataArray.
        expected_epsg: Target EPSG code (default: 32643 for UTM 43N).
        expected_res: Target pixel resolution in CRS units (default: 10.0m).

    Raises:
        SpatialAlignmentError: If CRS, resolution, or spatial coordinate arrays do not match.
    """
    # 1. Verify CRS if metadata is present
    pre_epsg = _get_crs_epsg(pre)
    post_epsg = _get_crs_epsg(post)

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

    # 2. Verify spatial dimensions exist
    for name, obj in [("pre", pre), ("post", post)]:
        if "x" not in obj.coords or "y" not in obj.coords:
            raise SpatialAlignmentError(
                f"{name.capitalize()} scene is missing spatial coordinate 'x' or 'y'."
            )

    # 3. Verify resolution
    # 3. Verify resolution for both pre- and post-scenes
    for name, obj in [("Pre-scene", pre), ("Post-scene", post)]:
        if len(obj.coords["x"]) > 1:
            res_x = abs(float(obj.coords["x"][1] - obj.coords["x"][0]))
            if abs(res_x - expected_res) > 1.0:
                raise SpatialAlignmentError(
                    f"{name} resolution ({res_x:.2f}m) does not match expected {expected_res}m."
                )
        if len(obj.coords["y"]) > 1:
            res_y = abs(float(obj.coords["y"][1] - obj.coords["y"][0]))
            if abs(res_y - expected_res) > 1.0:
                raise SpatialAlignmentError(
                    f"{name} resolution ({res_y:.2f}m) does not match expected {expected_res}m."
                )

    # 4. Verify coordinate alignment and shape
    if pre.coords["x"].shape != post.coords["x"].shape or pre.coords["y"].shape != post.coords["y"].shape:
        raise SpatialAlignmentError(
            f"Spatial coordinate shape mismatch: pre (y={len(pre.coords['y'])}, x={len(pre.coords['x'])}) "
            f"vs post (y={len(post.coords['y'])}, x={len(post.coords['x'])})."
        )

    if not np.allclose(pre.coords["x"].values, post.coords["x"].values, atol=0.5):
        raise SpatialAlignmentError("Pre- and post-scene 'x' coordinates do not align spatially.")
    if not np.allclose(pre.coords["y"].values, post.coords["y"].values, atol=0.5):
        raise SpatialAlignmentError("Pre- and post-scene 'y' coordinates do not align spatially.")


def detect_canopy_loss(
    pre_ds: Union[xr.Dataset, xr.DataArray],
    post_ds: Union[xr.Dataset, xr.DataArray],
    ndvi_threshold: float = DEFAULT_NDVI_THRESHOLD,
    min_area_sqm: float = DEFAULT_MIN_AREA_SQM,
    dilate_pixels: int = 2,
    check_spatial: bool = True,
) -> ChangeDetectionResult:
    """Compute bi-temporal canopy loss matrix and filter noise clusters.

    Compares two cloud-masked satellite passes, identifies vegetation drop
    candidates (delta_ndvi = ndvi_pre - ndvi_post > ndvi_threshold), and applies
    morphological opening and connected-component filtering to eliminate isolated
    pixel noise while retaining genuine clearing patches.

    Args:
        pre_ds: Baseline pre-clearing Dataset (with bands/SCL) or pre-computed NDVI DataArray.
        post_ds: Subsequent post-clearing Dataset or pre-computed NDVI DataArray.
        ndvi_threshold: Drop threshold for delta_ndvi (default: 0.25).
        min_area_sqm: Minimum contiguous patch area in square meters (default: 500m²).
        dilate_pixels: Dilation iterations for cloud masking if SCL present (default: 2).
        check_spatial: Whether to verify spatial alignment and CRS (default: True).

    Returns:
        ChangeDetectionResult containing:
        - change_mask: Cleaned boolean DataArray (True = detected clearing cluster)
        - delta_ndvi: Continuous difference DataArray (ndvi_pre - ndvi_post)
        - labeled_mask: Integer DataArray of connected component IDs
        - metrics: Dictionary of comprehensive spatial metrics and per-patch attributes

    Raises:
        SpatialAlignmentError: If scenes do not align spatially.
    """
    if check_spatial:
        verify_spatial_alignment(pre_ds, post_ds)

    # 1. Extract or compute pre-scene NDVI & valid mask
    if isinstance(pre_ds, xr.Dataset):
        pre_mask = (
            create_valid_mask(pre_ds["SCL"], dilate_pixels=dilate_pixels)
            if "SCL" in pre_ds and "valid_mask" not in pre_ds
            else pre_ds.get("valid_mask")
        )
        ndvi_pre = calculate_ndvi(pre_ds, valid_mask=pre_mask)
    else:
        ndvi_pre = pre_ds

    # 2. Extract or compute post-scene NDVI & valid mask
    if isinstance(post_ds, xr.Dataset):
        post_mask = (
            create_valid_mask(post_ds["SCL"], dilate_pixels=dilate_pixels)
            if "SCL" in post_ds and "valid_mask" not in post_ds
            else post_ds.get("valid_mask")
        )
        ndvi_post = calculate_ndvi(post_ds, valid_mask=post_mask)
    else:
        ndvi_post = post_ds

    # Squeeze out temporal dimension if present to prevent xarray bi-temporal outer-join collision
    if "time" in ndvi_pre.dims:
        ndvi_pre = ndvi_pre.squeeze("time", drop=True)
    if "time" in ndvi_post.dims:
        ndvi_post = ndvi_post.squeeze("time", drop=True)

    # 3. Compute continuous bi-temporal difference: delta_ndvi = ndvi_pre - ndvi_post
    delta_ndvi = ndvi_pre - ndvi_post

    # Determine mutual pixel validity
    valid_pre_arr = ~np.isnan(ndvi_pre.values)
    valid_post_arr = ~np.isnan(ndvi_post.values)
    mutual_valid = valid_pre_arr & valid_post_arr

    delta_vals = np.where(mutual_valid, delta_ndvi.values, np.nan)

    # 4. Initial binary thresholding on mutually valid clean pixels
    raw_candidates = (delta_vals > ndvi_threshold) & mutual_valid

    # 5. Morphological cleanup
    # 5a. Binary opening using 3x3 structuring element to eliminate isolated 1-pixel spikes
    structuring_element = np.ones((3, 3), dtype=bool)
    opened_mask = ndi.binary_opening(raw_candidates, structure=structuring_element)

    # 5b. Connected components labeling
    labeled_patches, num_initial = ndi.label(opened_mask, structure=structuring_element)

    # 5c. Filter out clusters smaller than min_area_sqm
    # Determine pixel area (default: 100m² for 10m Sentinel-2)
    pixel_area_sqm = 100.0
    if "x" in ndvi_pre.coords and "y" in ndvi_pre.coords and len(ndvi_pre.coords["x"]) > 1 and len(ndvi_pre.coords["y"]) > 1:
        res_x = abs(float(ndvi_pre.coords["x"][1] - ndvi_pre.coords["x"][0]))
        res_y = abs(float(ndvi_pre.coords["y"][1] - ndvi_pre.coords["y"][0]))
        pixel_area_sqm = res_x * res_y

    min_pixels = max(1, int(math.ceil(min_area_sqm / pixel_area_sqm)))

    if num_initial > 0:
        patch_counts = np.bincount(labeled_patches.ravel())
        valid_patch_ids = np.where(patch_counts >= min_pixels)[0]
        # Remove background label 0 from valid component ids
        valid_patch_ids = valid_patch_ids[valid_patch_ids > 0]
        cleaned_binary = np.isin(labeled_patches, valid_patch_ids)
    else:
        cleaned_binary = np.zeros_like(opened_mask, dtype=bool)
        valid_patch_ids = np.array([], dtype=int)

    # Re-label final patches sequentially
    final_labeled, num_patches = ndi.label(cleaned_binary, structure=structuring_element)

    # 6. Extract spatial patch metrics
    patches_metrics_list: List[Dict[str, Any]] = []
    total_change_pixels = int(np.count_nonzero(cleaned_binary))

    for patch_id in range(1, num_patches + 1):
        patch_sel = (final_labeled == patch_id)
        p_count = int(np.count_nonzero(patch_sel))
        p_area_sqm = float(p_count * pixel_area_sqm)
        p_area_acres = float(p_area_sqm / SQM_PER_ACRE)

        patch_deltas = delta_vals[patch_sel]
        patch_deltas_clean = patch_deltas[~np.isnan(patch_deltas)]
        mean_delta = float(np.mean(patch_deltas_clean)) if patch_deltas_clean.size > 0 else 0.0
        max_delta = float(np.max(patch_deltas_clean)) if patch_deltas_clean.size > 0 else 0.0

        patches_metrics_list.append({
            "patch_id": patch_id,
            "pixel_count": p_count,
            "area_sqm": round(p_area_sqm, 2),
            "area_acres": round(p_area_acres, 4),
            "mean_delta_ndvi": round(mean_delta, 4),
            "max_delta_ndvi": round(max_delta, 4),
        })

    total_area_sqm = float(total_change_pixels * pixel_area_sqm)
    total_area_acres = float(total_area_sqm / SQM_PER_ACRE)

    metrics_summary: Dict[str, Any] = {
        "num_patches": num_patches,
        "total_pixels": total_change_pixels,
        "total_area_sqm": round(total_area_sqm, 2),
        "total_area_acres": round(total_area_acres, 4),
        "ndvi_threshold": ndvi_threshold,
        "min_area_sqm": min_area_sqm,
        "min_pixels": min_pixels,
        "pixel_area_sqm": pixel_area_sqm,
        "initial_candidates_pixels": int(np.count_nonzero(raw_candidates)),
        "opened_pixels": int(np.count_nonzero(opened_mask)),
        "noise_pixels_filtered": int(np.count_nonzero(raw_candidates) - total_change_pixels),
        "patches": patches_metrics_list,
    }

    logger.info(
        f"Canopy loss detection completed: {num_patches} patches found, "
        f"{total_area_sqm:,.2f} m² ({total_area_acres:,.2f} acres) total clearing, "
        f"{metrics_summary['noise_pixels_filtered']} noise pixels filtered."
    )

    # 7. Build output xarray DataArrays preserving spatial coordinates
    spatial_coords = {
        dim: ndvi_pre.coords[dim]
        for dim in ["y", "x"]
        if dim in ndvi_pre.coords
    }
    if "spatial_ref" in ndvi_pre.coords:
        spatial_coords["spatial_ref"] = ndvi_pre.coords["spatial_ref"]

    change_mask_da = xr.DataArray(
        cleaned_binary,
        coords=spatial_coords,
        dims=["y", "x"],
        name="canopy_loss_mask",
        attrs={
            "long_name": "Canopy Loss Detected Mask",
            "ndvi_threshold": ndvi_threshold,
            "min_area_sqm": min_area_sqm,
            "num_patches": num_patches,
            "total_area_sqm": total_area_sqm,
            "total_area_acres": total_area_acres,
        },
    )

    delta_ndvi_da = xr.DataArray(
        delta_vals,
        coords=spatial_coords,
        dims=["y", "x"],
        name="delta_ndvi",
        attrs={
            "long_name": "Bi-temporal NDVI Difference (Pre - Post)",
            "formula": "ndvi_pre - ndvi_post",
            "ndvi_threshold": ndvi_threshold,
        },
    )

    labeled_mask_da = xr.DataArray(
        final_labeled,
        coords=spatial_coords,
        dims=["y", "x"],
        name="canopy_loss_labels",
        attrs={
            "long_name": "Canopy Loss Cluster Labels",
            "num_patches": num_patches,
        },
    )

    return ChangeDetectionResult(
        change_mask=change_mask_da,
        delta_ndvi=delta_ndvi_da,
        labeled_mask=labeled_mask_da,
        metrics=metrics_summary,
    )
