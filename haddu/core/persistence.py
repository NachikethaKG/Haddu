"""Persistent Breakpoint & Bare Soil Transition Detector.

Develops the multi-pass persistence engine that confirms canopy loss only when negative
residuals exceed statistical thresholds across consecutive passes and correspond with
bare-soil signatures (Delta NDBI > 0), filtering out temporary haze or seasonal dips.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
import logging
import math
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import xarray as xr

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from haddu.core.harmonic import HarmonicModel
from haddu.core.indices import calculate_ndbi

logger = logging.getLogger(__name__)

# Default number of consecutive anomalous passes required to confirm persistence
DEFAULT_CONSECUTIVE_PASSES: int = 2

# Standard z-score threshold (z_t < -2.5 indicates drop beyond 99.4% seasonal baseline)
DEFAULT_Z_THRESHOLD: float = 2.5

# Minimum clearing patch area in square meters (500 sqm = 5 Sentinel-2 10m pixels)
DEFAULT_MIN_AREA_SQM: float = 500.0

# Minimum residual standard deviation floor to avoid division by zero
DEFAULT_MIN_SIGMA: float = 0.01

# Minimum Delta NDBI increase threshold to confirm bare soil / non-vegetation transition
DEFAULT_NDBI_THRESHOLD: float = 0.0

# Standard Western Ghats undisturbed evergreen canopy NDBI baseline fallback
DEFAULT_FOREST_NDBI_BASELINE: float = -0.15

# Square meters per acre constant
SQM_PER_ACRE: float = 4046.8564224


@dataclass
class PersistentBreakResult:
    """Container for persistent breakpoint detection outputs and spatial metrics.

    Attributes:
        change_mask: 2D boolean DataArray (y, x) where True indicates confirmed canopy loss.
        break_dates: 2D datetime64 DataArray (y, x) containing estimated inception date of break (or NaT).
        break_indices: 2D integer DataArray (y, x) containing time index of break (or -1).
        z_scores: 3D float DataArray (time, y, x) of standardized residuals.
        residuals: 3D float DataArray (time, y, x) of observed minus predicted NDVI.
        delta_ndbi: 3D or 2D float DataArray of NDBI change confirming bare soil transition.
        num_patches: Count of contiguous verified clearing patches.
        total_area_sqm: Total verified clearing area in square meters.
        total_area_acres: Total verified clearing area in acres.
        patch_metrics: List of summary dictionaries for each connected patch.
        attrs: Metadata dictionary.
    """

    change_mask: xr.DataArray
    break_dates: xr.DataArray
    break_indices: xr.DataArray
    z_scores: xr.DataArray
    residuals: xr.DataArray
    delta_ndbi: Optional[xr.DataArray] = None
    num_patches: int = 0
    total_area_sqm: float = 0.0
    total_area_acres: float = 0.0
    patch_metrics: List[Dict[str, Any]] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[xr.DataArray]:
        """Support standard 2-tuple unpacking: change_mask, break_dates = result."""
        return iter((self.change_mask, self.break_dates))

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like indexing access to variables."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"PersistentBreakResult has no attribute or variable '{key}'.")

    @property
    def earliest_break_date(self) -> Optional[pd.Timestamp]:
        """Earliest confirmed break timestamp across all detected patches."""
        valid_dates = self.break_dates.values[~pd.isna(self.break_dates.values)]
        if len(valid_dates) == 0:
            return None
        return pd.Timestamp(np.min(valid_dates))

    @property
    def latest_break_date(self) -> Optional[pd.Timestamp]:
        """Latest confirmed break timestamp across all detected patches."""
        valid_dates = self.break_dates.values[~pd.isna(self.break_dates.values)]
        if len(valid_dates) == 0:
            return None
        return pd.Timestamp(np.max(valid_dates))

    def to_dataset(self) -> xr.Dataset:
        """Export primary result arrays and metadata into an xarray.Dataset."""
        data_vars: Dict[str, xr.DataArray] = {
            "change_mask": self.change_mask,
            "break_dates": self.break_dates,
            "break_indices": self.break_indices,
            "z_scores": self.z_scores,
            "residuals": self.residuals,
        }
        if self.delta_ndbi is not None:
            data_vars["delta_ndbi"] = self.delta_ndbi

        return xr.Dataset(data_vars=data_vars, attrs=self.attrs)


def detect_persistent_breaks(
    timeseries_cube: xr.Dataset,
    harmonic_model: HarmonicModel,
    consecutive_passes: int = DEFAULT_CONSECUTIVE_PASSES,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    min_area_sqm: float = DEFAULT_MIN_AREA_SQM,
    confirm_ndbi: bool = True,
    ndbi_threshold: float = DEFAULT_NDBI_THRESHOLD,
    ndbi_baseline: Optional[Union[xr.DataArray, float]] = None,
    min_sigma: float = DEFAULT_MIN_SIGMA,
) -> PersistentBreakResult:
    """Detect persistent canopy loss breakpoints verified by spectral bare-soil transition.

    Identifies significant negative NDVI anomalies relative to the seasonal Fourier baseline:
        z_t = (NDVI_t - hat{NDVI}_t) / sigma < -z_threshold

    Verifies temporal persistence across consecutive valid (cloud-free) observations,
    confirms spectral shift toward bare-soil / non-vegetated surface (Delta NDBI > 0),
    and filters out single-pixel noise and sub-threshold clusters using morphological opening.

    Args:
        timeseries_cube: xarray.Dataset containing 'ndvi' and optionally 'ndbi' or 'B11'/'B08'
            with dimensions (time, y, x).
        harmonic_model: Fitted HarmonicModel instance from fit_harmonic_baseline.
        consecutive_passes: Number of consecutive valid observations that must show anomaly
            (default: 2 passes).
        z_threshold: Positive z-score threshold; anomalies trigger when z_t < -z_threshold
            (default: 2.5).
        min_area_sqm: Minimum contiguous patch area in square meters (default: 500 sqm).
        confirm_ndbi: If True, requires Delta NDBI > ndbi_threshold to eliminate water inundation
            and shadow artifacts (default: True).
        ndbi_threshold: Minimum Delta NDBI increase to confirm bare soil transition (default: 0.0).
        ndbi_baseline: Optional reference NDBI array or scalar. If None, derived per-pixel
            from the median of pre-break or non-anomalous valid observations.
        min_sigma: Lower floor on residual standard deviation to prevent runaway z-scores.

    Returns:
        PersistentBreakResult containing change_mask, break_dates, break_indices, z_scores,
        residuals, and spatial patch metrics.

    Raises:
        ValueError: If input parameters or required dataset variables are invalid.
    """
    if consecutive_passes < 1:
        raise ValueError(f"consecutive_passes must be >= 1, received {consecutive_passes}.")
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be positive, received {z_threshold}.")
    if not isinstance(timeseries_cube, xr.Dataset):
        raise TypeError(
            f"Expected xarray.Dataset for timeseries_cube, received {type(timeseries_cube).__name__}."
        )
    if "ndvi" not in timeseries_cube.data_vars:
        raise ValueError("Input timeseries_cube must contain an 'ndvi' data variable.")
    if "time" not in timeseries_cube.dims:
        raise ValueError("Input timeseries_cube must have a 'time' dimension.")

    da_ndvi = timeseries_cube["ndvi"]
    time_coords = da_ndvi.time
    n_times = len(time_coords)

    # Spatial dimensions and shapes
    spatial_dims = [d for d in da_ndvi.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(
            f"Expected 2 spatial dimensions in timeseries_cube, found {spatial_dims}."
        )
    dim_y, dim_x = spatial_dims[0], spatial_dims[1]
    h_y = da_ndvi.sizes[dim_y]
    w_x = da_ndvi.sizes[dim_x]
    p_pixels = h_y * w_x

    # Verify spatial alignment with harmonic model
    if harmonic_model.c0.shape != (h_y, w_x):
        raise ValueError(
            f"Harmonic model spatial grid {harmonic_model.c0.shape} does not match "
            f"timeseries_cube spatial grid ({h_y}, {w_x})."
        )

    # 1. Compute residuals and standardized z-scores
    residuals = harmonic_model.compute_residuals(timeseries_cube)
    z_scores = harmonic_model.compute_zscores(timeseries_cube, min_sigma=min_sigma)

    # 2. Extract or compute NDBI array if spectral confirmation is enabled
    da_ndbi: Optional[xr.DataArray] = None
    if confirm_ndbi:
        if "ndbi" in timeseries_cube.data_vars:
            da_ndbi = timeseries_cube["ndbi"]
        elif "B11" in timeseries_cube.data_vars and "B08" in timeseries_cube.data_vars:
            logger.info("Computing NDBI array from B11 and B08 bands for spectral confirmation...")
            valid_m = timeseries_cube.get("valid_mask")
            da_ndbi = calculate_ndbi(timeseries_cube, valid_mask=valid_m)
        else:
            logger.warning(
                "Spectral confirmation requested (confirm_ndbi=True) but neither 'ndbi' nor "
                "'B11'/'B08' bands were found in timeseries_cube. Proceeding with NDVI persistence only."
            )

    # 3. Determine pixel-level observation validity
    ndvi_2d = da_ndvi.values.reshape((n_times, p_pixels))
    valid_obs = ~np.isnan(ndvi_2d)

    if "valid_mask" in timeseries_cube.data_vars:
        scl_valid = timeseries_cube["valid_mask"].values.reshape((n_times, p_pixels)).astype(bool)
        valid_obs = valid_obs & scl_valid

    if da_ndbi is not None:
        ndbi_2d = da_ndbi.values.reshape((n_times, p_pixels))
        valid_obs = valid_obs & (~np.isnan(ndbi_2d))
    else:
        ndbi_2d = None

    # 4. Standardized negative anomaly identification
    z_2d = z_scores.values.reshape((n_times, p_pixels))
    ndvi_anomalous = (z_2d < -z_threshold) & valid_obs  # (T, P)

    # 5. Spectral bare-soil confirmation (Delta NDBI > ndbi_threshold)
    delta_ndbi_2d = np.zeros((n_times, p_pixels), dtype=np.float32)
    if da_ndbi is not None and ndbi_2d is not None:
        # Determine baseline NDBI per pixel
        if ndbi_baseline is not None:
            if isinstance(ndbi_baseline, xr.DataArray):
                base_ndbi_flat = ndbi_baseline.values.ravel()
            else:
                base_ndbi_flat = np.full(p_pixels, float(ndbi_baseline))
        else:
            # Per-pixel median across non-anomalous valid passes
            base_ndbi_flat = np.full(p_pixels, DEFAULT_FOREST_NDBI_BASELINE, dtype=np.float32)
            non_anom_valid = (~ndvi_anomalous) & valid_obs  # (T, P)

            # Fast computation of per-pixel baseline NDBI
            has_norm_obs = np.any(non_anom_valid, axis=0)
            norm_indices = np.where(has_norm_obs)[0]
            for p in norm_indices:
                t_norm = np.where(non_anom_valid[:, p])[0]
                base_ndbi_flat[p] = np.median(ndbi_2d[t_norm, p])

        delta_ndbi_2d = ndbi_2d - base_ndbi_flat[None, :]
        bare_soil_confirmed = (delta_ndbi_2d > ndbi_threshold) & valid_obs
        joint_anomalous = ndvi_anomalous & bare_soil_confirmed
    else:
        joint_anomalous = ndvi_anomalous

    # 6. Multi-pass persistence evaluation
    # Pre-filter candidate pixels that have at least `consecutive_passes` total anomalies
    cand_pixels = np.where(np.sum(joint_anomalous, axis=0) >= consecutive_passes)[0]

    raw_change_mask = np.zeros(p_pixels, dtype=bool)
    break_indices = np.full(p_pixels, -1, dtype=np.int32)

    for p in cand_pixels:
        valid_t = np.where(valid_obs[:, p])[0]
        if len(valid_t) < consecutive_passes:
            continue

        anom_seq = joint_anomalous[valid_t, p]
        run_len = 0
        for i, is_anom in enumerate(anom_seq):
            if is_anom:
                run_len += 1
                if run_len == consecutive_passes:
                    # Inception pass: the first pass of the consecutive anomalous run
                    break_t = valid_t[i - consecutive_passes + 1]
                    raw_change_mask[p] = True
                    break_indices[p] = break_t
                    break
            else:
                run_len = 0

    # 7. Morphological opening and minimum area filtering
    raw_2d = raw_change_mask.reshape((h_y, w_x))
    structuring_element = np.ones((3, 3), dtype=bool)

    # 7a. Binary opening to remove isolated 1-pixel spikes
    opened_mask = ndi.binary_opening(raw_2d, structure=structuring_element)

    # 7b. Connected components labeling
    labeled_patches, num_initial = ndi.label(opened_mask, structure=structuring_element)

    # 7c. Compute pixel area from spatial coordinates
    pixel_area_sqm = 100.0  # Default 10m x 10m
    if dim_x in da_ndvi.coords and dim_y in da_ndvi.coords:
        if len(da_ndvi.coords[dim_x]) > 1 and len(da_ndvi.coords[dim_y]) > 1:
            res_x = abs(float(da_ndvi.coords[dim_x][1] - da_ndvi.coords[dim_x][0]))
            res_y = abs(float(da_ndvi.coords[dim_y][1] - da_ndvi.coords[dim_y][0]))
            pixel_area_sqm = res_x * res_y

    min_pixels = max(1, int(math.ceil(min_area_sqm / pixel_area_sqm)))

    if num_initial > 0:
        patch_counts = np.bincount(labeled_patches.ravel())
        valid_ids = np.where(patch_counts >= min_pixels)[0]
        valid_ids = valid_ids[valid_ids > 0]
        cleaned_mask = np.isin(labeled_patches, valid_ids)
    else:
        cleaned_mask = np.zeros((h_y, w_x), dtype=bool)

    # Re-label final patches sequentially
    final_labeled, num_patches = ndi.label(cleaned_mask, structure=structuring_element)

    # 8. Reset break indices for filtered out pixels
    break_indices_2d = break_indices.reshape((h_y, w_x))
    break_indices_2d[~cleaned_mask] = -1

    # 9. Format break dates array
    time_vals = time_coords.values
    break_dates_arr = np.full((h_y, w_x), np.datetime64("NaT"), dtype="datetime64[ns]")

    changed_y, changed_x = np.where(cleaned_mask)
    for y_idx, x_idx in zip(changed_y, changed_x):
        b_idx = break_indices_2d[y_idx, x_idx]
        if 0 <= b_idx < n_times:
            break_dates_arr[y_idx, x_idx] = pd.Timestamp(time_vals[b_idx]).to_datetime64()

    # 10. Extract patch metrics
    spatial_coords = {dim_y: da_ndvi.coords[dim_y], dim_x: da_ndvi.coords[dim_x]}
    if "spatial_ref" in da_ndvi.coords:
        spatial_coords["spatial_ref"] = da_ndvi.coords["spatial_ref"]

    src_crs_wkt = None
    if "spatial_ref" in da_ndvi.coords:
        spatial_ref_var = da_ndvi.coords["spatial_ref"]
        src_crs_wkt = getattr(spatial_ref_var, "crs_wkt", None) or getattr(
            spatial_ref_var, "spatial_ref", None
        )

    transformer = None
    if HAS_PYPROJ and src_crs_wkt:
        try:
            crs_from = pyproj.CRS.from_user_input(src_crs_wkt)
            crs_to = pyproj.CRS.from_epsg(4326)
            transformer = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
        except Exception as e:
            logger.debug(f"Could not build coordinate transformer: {e}")

    patch_metrics_list: List[Dict[str, Any]] = []
    total_change_pixels = int(np.count_nonzero(cleaned_mask))
    total_area_sqm = float(total_change_pixels * pixel_area_sqm)
    total_area_acres = float(total_area_sqm / SQM_PER_ACRE)

    y_coords = da_ndvi.coords[dim_y].values
    x_coords = da_ndvi.coords[dim_x].values

    for patch_id in range(1, num_patches + 1):
        patch_sel = final_labeled == patch_id
        p_count = int(np.count_nonzero(patch_sel))
        p_area_sqm = float(p_count * pixel_area_sqm)
        p_area_acres = float(p_area_sqm / SQM_PER_ACRE)

        # Centroid in projected coordinates
        p_ys, p_xs = np.where(patch_sel)
        cent_y = float(np.mean(y_coords[p_ys]))
        cent_x = float(np.mean(x_coords[p_xs]))

        # Earliest and latest break dates in the patch
        patch_dates = break_dates_arr[patch_sel]
        valid_p_dates = patch_dates[~pd.isna(patch_dates)]
        earliest_p_date = (
            str(pd.Timestamp(np.min(valid_p_dates)).strftime("%Y-%m-%d"))
            if len(valid_p_dates) > 0
            else None
        )
        latest_p_date = (
            str(pd.Timestamp(np.max(valid_p_dates)).strftime("%Y-%m-%d"))
            if len(valid_p_dates) > 0
            else None
        )

        # Z-score severity
        patch_z = z_2d[:, patch_sel.ravel()]
        min_z = float(np.nanmin(patch_z)) if patch_z.size > 0 else 0.0

        p_info: Dict[str, Any] = {
            "patch_id": patch_id,
            "pixel_count": p_count,
            "area_sqm": round(p_area_sqm, 2),
            "area_acres": round(p_area_acres, 4),
            "earliest_break_date": earliest_p_date,
            "latest_break_date": latest_p_date,
            "min_zscore": round(min_z, 2),
            "centroid_y": round(cent_y, 2),
            "centroid_x": round(cent_x, 2),
        }

        # Reproject centroid to WGS84 lat/lon if transformer available
        if transformer is not None:
            try:
                lon, lat = transformer.transform(cent_x, cent_y)
                p_info["centroid_lat"] = round(float(lat), 6)
                p_info["centroid_lon"] = round(float(lon), 6)
            except Exception:
                pass

        patch_metrics_list.append(p_info)

    # 11. Wrap arrays into xarray DataArrays
    da_change_mask = xr.DataArray(
        cleaned_mask,
        dims=[dim_y, dim_x],
        coords=spatial_coords,
        name="persistent_change_mask",
        attrs={"long_name": "Confirmed Persistent Canopy Loss Mask"},
    )

    da_break_dates = xr.DataArray(
        break_dates_arr,
        dims=[dim_y, dim_x],
        coords=spatial_coords,
        name="break_date",
        attrs={"long_name": "Estimated Inception Date of Canopy Loss"},
    )

    da_break_indices = xr.DataArray(
        break_indices_2d,
        dims=[dim_y, dim_x],
        coords=spatial_coords,
        name="break_index",
        attrs={"long_name": "Time Index of Inception Pass (-1 for Unchanged)"},
    )

    da_delta_ndbi = None
    if da_ndbi is not None:
        da_delta_ndbi = xr.DataArray(
            delta_ndbi_2d.reshape((n_times, h_y, w_x)),
            dims=["time", dim_y, dim_x],
            coords=da_ndbi.coords,
            name="delta_ndbi",
            attrs={"long_name": "Delta NDBI (Relative to Forest Baseline)"},
        )

    logger.info(
        f"Detected {num_patches} persistent clearing patch(es) totaling "
        f"{total_area_sqm:.1f} m² ({total_area_acres:.2f} acres) across {n_times} passes."
    )

    result_attrs = {
        "title": "Haddu Persistent Breakpoint & Bare Soil Detection",
        "consecutive_passes": int(consecutive_passes),
        "z_threshold": float(z_threshold),
        "min_area_sqm": float(min_area_sqm),
        "confirm_ndbi": bool(confirm_ndbi),
        "num_patches": int(num_patches),
        "total_area_sqm": float(total_area_sqm),
        "total_area_acres": float(total_area_acres),
    }

    return PersistentBreakResult(
        change_mask=da_change_mask,
        break_dates=da_break_dates,
        break_indices=da_break_indices,
        z_scores=z_scores,
        residuals=residuals,
        delta_ndbi=da_delta_ndbi,
        num_patches=num_patches,
        total_area_sqm=total_area_sqm,
        total_area_acres=total_area_acres,
        patch_metrics=patch_metrics_list,
        attrs=result_attrs,
    )
