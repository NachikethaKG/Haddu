"""Raster-to-vector polygon extraction and GeoJSON generation module.

Converts raster change detection masks into structured geopandas GeoDataFrames
with UTM projected area measurements, geographic WGS84 centroids, and spectral metrics.
"""

import logging
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import rasterio.features
from rasterio.transform import from_origin
import rioxarray  # noqa: F401 - activates .rio accessor on xarray objects
from shapely.geometry import Polygon, shape
import xarray as xr

from haddu.core.change_detector import ChangeDetectionResult, SQM_PER_ACRE
from haddu.core.persistence import PersistentBreakResult
from haddu.core.sar_change import SARChangeDetectionResult

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_CRS = "EPSG:32643"  # UTM Zone 43N
DEFAULT_TARGET_CRS = "EPSG:4326"   # WGS84 Latitude/Longitude


def extract_affine_transform(da: xr.DataArray) -> rasterio.Affine:
    """Extract affine transform from xarray DataArray via rioxarray or 1D coordinates."""
    if hasattr(da, "rio") and da.rio.crs:
        try:
            return da.rio.transform()
        except Exception as exc:
            logger.debug(f"da.rio.transform() failed ({exc}), falling back to coordinate math.")

    if "x" in da.coords and "y" in da.coords and len(da.coords["x"]) > 1 and len(da.coords["y"]) > 1:
        x_vals = da.coords["x"].values
        y_vals = da.coords["y"].values
        res_x = float(abs(x_vals[1] - x_vals[0]))
        res_y = float(abs(y_vals[1] - y_vals[0]))

        # In standard north-up rasters, y is descending
        top = float(y_vals[0]) + (res_y / 2.0 if y_vals[0] > y_vals[1] else -res_y / 2.0)
        left = float(x_vals[0]) - res_x / 2.0
        return from_origin(left, top, res_x, res_y)

    return rasterio.Affine.identity()


def vectorize_change_mask(
    change_input: Union[ChangeDetectionResult, xr.DataArray],
    source_crs: str = DEFAULT_SOURCE_CRS,
    target_crs: str = DEFAULT_TARGET_CRS,
) -> gpd.GeoDataFrame:
    """Convert raster canopy loss mask into a projected vector GeoDataFrame.

    Uses rasterio.features.shapes to trace polygon outlines around contiguous
    loss patches, computes area in square meters and acres, calculates WGS84
    centroids, and reprojects vector boundaries to EPSG:4326 (WGS84).

    Args:
        change_input: ChangeDetectionResult instance or boolean/integer DataArray.
        source_crs: Coordinate Reference System of the raster grid (default: EPSG:32643).
        target_crs: Destination CRS for polygon output (default: EPSG:4326).

    Returns:
        geopandas.GeoDataFrame in target_crs with columns:
        - patch_id (int)
        - area_sqm (float)
        - area_acres (float)
        - centroid_lon (float)
        - centroid_lat (float)
        - mean_delta_ndvi (Optional[float])
        - max_delta_ndvi (Optional[float])
        - geometry (shapely.geometry.Polygon)
    """
    empty_cols = [
        "patch_id",
        "area_sqm",
        "area_acres",
        "centroid_lon",
        "centroid_lat",
        "mean_delta_ndvi",
        "max_delta_ndvi",
        "geometry",
    ]

    # Resolve DataArrays from ChangeDetectionResult, PersistentBreakResult, or raw DataArray
    delta_ndvi_da: Optional[xr.DataArray] = None
    if isinstance(change_input, PersistentBreakResult):
        return vectorize_persistent_breaks(
            change_input,
            source_crs=source_crs,
            target_crs=target_crs,
        )
    elif isinstance(change_input, ChangeDetectionResult):
        labeled_da = change_input.labeled_mask
        mask_da = change_input.change_mask
        delta_ndvi_da = change_input.delta_ndvi
    elif isinstance(change_input, SARChangeDetectionResult):
        labeled_da = change_input.labeled_mask
        mask_da = change_input.disturbance_mask
        delta_ndvi_da = change_input.delta_vh
    elif isinstance(change_input, xr.DataArray):
        labeled_da = change_input
        mask_da = change_input > 0
    else:
        raise TypeError(f"Unsupported change input type: {type(change_input)}")

    labeled_arr = labeled_da.values.astype(np.int32)
    binary_mask = (labeled_arr > 0)

    if not np.any(binary_mask):
        logger.info("No change detected in raster mask. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(
            columns=empty_cols,
            geometry="geometry",
            crs=target_crs,
        )

    # Derive affine transform
    transform = extract_affine_transform(labeled_da)

    # Extract polygon geometries using rasterio.features.shapes
    shapes_generator = rasterio.features.shapes(
        labeled_arr,
        mask=binary_mask,
        transform=transform,
        connectivity=8,
    )

    records = []
    delta_vals = delta_ndvi_da.values if delta_ndvi_da is not None else None

    for geom_dict, label_val in shapes_generator:
        patch_id = int(label_val)
        if patch_id == 0:
            continue

        poly = shape(geom_dict)
        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty or poly.area <= 0:
            continue

        patch_mask = (labeled_arr == patch_id)
        mean_delta = None
        max_delta = None
        if delta_vals is not None:
            sel_deltas = delta_vals[patch_mask]
            clean_deltas = sel_deltas[~np.isnan(sel_deltas)]
            if clean_deltas.size > 0:
                mean_delta = round(float(np.mean(clean_deltas)), 4)
                max_delta = round(float(np.max(clean_deltas)), 4)

        records.append({
            "patch_id": patch_id,
            "area_sqm": round(float(poly.area), 2),
            "area_acres": round(float(poly.area / SQM_PER_ACRE), 4),
            "mean_delta_ndvi": mean_delta,
            "max_delta_ndvi": max_delta,
            "geometry": poly,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=empty_cols,
            geometry="geometry",
            crs=target_crs,
        )

    # Build GeoDataFrame in native projected CRS (meters)
    gdf_proj = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)

    # Calculate exact geographic centroids by reprojecting projected centroids
    proj_centroids = gdf_proj.geometry.centroid
    wgs84_centroids = proj_centroids.to_crs(target_crs)

    # Reproject polygons to target CRS (EPSG:4326)
    gdf_wgs84 = gdf_proj.to_crs(target_crs)
    gdf_wgs84["centroid_lon"] = wgs84_centroids.x
    gdf_wgs84["centroid_lat"] = wgs84_centroids.y

    # Reorder columns cleanly
    gdf_wgs84 = gdf_wgs84[empty_cols]
    gdf_wgs84 = gdf_wgs84.sort_values(by="area_sqm", ascending=False).reset_index(drop=True)

    logger.info(
        f"Vectorized {len(gdf_wgs84)} polygon(s). Total loss area: "
        f"{gdf_wgs84['area_acres'].sum():.2f} acres."
    )
    return gdf_wgs84


def vectorize_persistent_breaks(
    result: PersistentBreakResult,
    source_crs: str = DEFAULT_SOURCE_CRS,
    target_crs: str = DEFAULT_TARGET_CRS,
) -> gpd.GeoDataFrame:
    """Convert PersistentBreakResult into a projected vector GeoDataFrame.

    Traces polygon outlines around contiguous persistent clearing patches,
    computes area in square meters and acres, derives WGS84 centroids,
    and attaches break_date and confidence_zscore attributes.

    Args:
        result: PersistentBreakResult instance from detect_persistent_breaks.
        source_crs: Coordinate Reference System of the raster grid (default: EPSG:32643).
        target_crs: Destination CRS for polygon output (default: EPSG:4326).

    Returns:
        geopandas.GeoDataFrame in target_crs with columns:
        - patch_id (int)
        - break_date (Optional[str])
        - confidence_zscore (Optional[float])
        - area_acres (float)
        - area_sqm (float)
        - centroid_lat (float)
        - centroid_lon (float)
        - geometry (shapely.geometry.Polygon)
    """
    import scipy.ndimage as ndi
    import pandas as pd

    empty_cols = [
        "patch_id",
        "break_date",
        "confidence_zscore",
        "area_acres",
        "area_sqm",
        "centroid_lat",
        "centroid_lon",
        "geometry",
    ]

    mask_da = result.change_mask
    mask_arr = mask_da.values.astype(bool)

    if not np.any(mask_arr):
        logger.info("No persistent change detected. Returning empty GeoDataFrame.")
        return gpd.GeoDataFrame(
            columns=empty_cols,
            geometry="geometry",
            crs=target_crs,
        )

    # Derive connected component labels
    labeled_arr, num_patches = ndi.label(mask_arr, structure=np.ones((3, 3), dtype=bool))

    transform = extract_affine_transform(mask_da)
    shapes_generator = rasterio.features.shapes(
        labeled_arr.astype(np.int32),
        mask=mask_arr,
        transform=transform,
        connectivity=8,
    )

    records = []
    break_dates_arr = result.break_dates.values
    z_scores_arr = result.z_scores.values if result.z_scores is not None else None

    # Pre-map patch_metrics if available
    metrics_map = {m["patch_id"]: m for m in (result.patch_metrics or [])}

    for geom_dict, label_val in shapes_generator:
        patch_id = int(label_val)
        if patch_id == 0:
            continue

        poly = shape(geom_dict)
        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty or poly.area <= 0:
            continue

        patch_mask = labeled_arr == patch_id

        # Break date
        break_date_str = None
        if patch_id in metrics_map and metrics_map[patch_id].get("earliest_break_date"):
            break_date_str = str(metrics_map[patch_id]["earliest_break_date"])
        else:
            dates_in_patch = break_dates_arr[patch_mask]
            valid_dates = dates_in_patch[~pd.isna(dates_in_patch)]
            if len(valid_dates) > 0:
                break_date_str = pd.Timestamp(np.min(valid_dates)).strftime("%Y-%m-%d")

        # Confidence z-score
        confidence_zscore = None
        if patch_id in metrics_map and metrics_map[patch_id].get("min_zscore") is not None:
            confidence_zscore = abs(float(metrics_map[patch_id]["min_zscore"]))
        elif z_scores_arr is not None:
            sel_z = z_scores_arr[:, patch_mask]
            clean_z = sel_z[~np.isnan(sel_z)]
            if clean_z.size > 0:
                confidence_zscore = round(abs(float(np.min(clean_z))), 2)

        records.append({
            "patch_id": patch_id,
            "break_date": break_date_str,
            "confidence_zscore": confidence_zscore,
            "area_sqm": round(float(poly.area), 2),
            "area_acres": round(float(poly.area / SQM_PER_ACRE), 4),
            "geometry": poly,
        })

    if not records:
        return gpd.GeoDataFrame(
            columns=empty_cols,
            geometry="geometry",
            crs=target_crs,
        )

    gdf_proj = gpd.GeoDataFrame(records, geometry="geometry", crs=source_crs)
    proj_centroids = gdf_proj.geometry.centroid
    wgs84_centroids = proj_centroids.to_crs(target_crs)

    gdf_wgs84 = gdf_proj.to_crs(target_crs)
    gdf_wgs84["centroid_lat"] = wgs84_centroids.y
    gdf_wgs84["centroid_lon"] = wgs84_centroids.x

    gdf_wgs84 = gdf_wgs84[empty_cols]
    gdf_wgs84 = gdf_wgs84.sort_values(by="area_sqm", ascending=False).reset_index(drop=True)

    logger.info(
        f"Vectorized {len(gdf_wgs84)} persistent breakpoint polygon(s). Total loss area: "
        f"{gdf_wgs84['area_acres'].sum():.2f} acres."
    )
    return gdf_wgs84
