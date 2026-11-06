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
from shapely.geometry import shape, Polygon
import xarray as xr

from haddu.core.change_detector import ChangeDetectionResult, SQM_PER_ACRE

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

    # Resolve DataArrays from ChangeDetectionResult or raw DataArray
    delta_ndvi_da: Optional[xr.DataArray] = None
    if isinstance(change_input, ChangeDetectionResult):
        labeled_da = change_input.labeled_mask
        mask_da = change_input.change_mask
        delta_ndvi_da = change_input.delta_ndvi
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
