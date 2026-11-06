"""Multi-temporal time-series cube builder for Sentinel-2 optical imagery.

Aggregates cloud-masked satellite passes across multi-year or annual monitoring windows
into regularized xarray time-series cubes with clean NDVI and NDBI arrays.
"""

from datetime import date, datetime
import logging
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pystac
import xarray as xr

from haddu.core.stac_client import (
    DEFAULT_AOI_BBOX,
    search_passes,
    STACError,
)
from haddu.core.loader import (
    DEFAULT_CRS,
    DEFAULT_RESOLUTION,
    DEFAULT_CHUNKS,
    load_stac_dataset,
)
from haddu.core.masking import (
    DEFAULT_DILATE_PIXELS,
    create_valid_mask,
)
from haddu.core.indices import (
    calculate_ndvi,
    calculate_ndbi,
)

logger = logging.getLogger(__name__)

# Standard multispectral bands required for cloud-masked NDVI and NDBI cubes
DEFAULT_CUBE_BANDS = ["B04", "B08", "B11", "SCL"]


def format_date_str(d: Union[str, date, datetime]) -> str:
    """Format input date object or string into standardized YYYY-MM-DD."""
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y-%m-%d")
    return str(d).strip()


def build_timeseries_cube(
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
    start_date: Union[str, date] = "2024-01-01",
    end_date: Union[str, date] = "2024-12-31",
    max_cloud: float = 25.0,
    resolution: Union[int, float] = DEFAULT_RESOLUTION,
    crs: str = DEFAULT_CRS,
    dilate_pixels: int = DEFAULT_DILATE_PIXELS,
    preferred_tile: Optional[str] = "T43PER",
    max_passes: Optional[int] = None,
    items: Optional[Sequence[pystac.Item]] = None,
    chunks: Optional[Dict[str, int]] = None,
    compute: bool = False,
) -> xr.Dataset:
    """Construct a multi-temporal time-series cube with cloud-masked NDVI and NDBI.

    Queries Planetary Computer STAC for Sentinel-2 L2A passes over the requested
    spatio-temporal window, streams required spectral bands (B04, B08, B11, SCL),
    applies morphologically dilated cloud/shadow masking, computes NDVI and NDBI,
    and returns a regularized xarray.Dataset with dimensions (time, y, x).

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
        start_date: Start of monitoring window (YYYY-MM-DD).
        end_date: End of monitoring window (YYYY-MM-DD).
        max_cloud: Maximum cloud cover percentage allowed (default: 25.0).
        resolution: Spatial pixel resolution in CRS units (default: 10 meters).
        crs: Target Coordinate Reference System (default: 'EPSG:32643').
        dilate_pixels: Cloud boundary dilation radius in pixels (default: 2).
        preferred_tile: Preferred MGRS tile ID (default: 'T43PER' for full Taluk).
        max_passes: Optional maximum number of passes to include in the cube.
        items: Optional pre-queried or synthetic sequence of pystac.Item objects.
        chunks: Dask spatial chunk mapping (default: {'x': 2048, 'y': 2048}).
        compute: If True, computes the full Dask graph into memory (default: False).

    Returns:
        xarray.Dataset with dimensions (time, y, x) containing:
        - ndvi: Clean vegetation index array with invalid pixels masked to NaN.
        - ndbi: Clean built-up / bare soil index with invalid pixels masked to NaN.
        - valid_mask: Boolean array indicating clean, cloud-free surface reflectance.

    Raises:
        ValueError: If no candidate Sentinel-2 passes are found in the window.
    """
    start_str = format_date_str(start_date)
    end_str = format_date_str(end_date)

    # 1. Query STAC catalog if items not pre-supplied
    if items is None:
        logger.info(
            f"Querying Sentinel-2 passes for AOI {list(bbox)} between {start_str} "
            f"and {end_str} (max_cloud={max_cloud}%)..."
        )
        found_items = search_passes(
            bbox=bbox,
            start_date=start_str,
            end_date=end_str,
            max_cloud=max_cloud,
        )
    else:
        found_items = list(items)

    if not found_items:
        raise ValueError(
            f"No Sentinel-2 passes found between {start_str} and {end_str} "
            f"for bounding box {list(bbox)} with max_cloud<={max_cloud}%."
        )

    # 2. Filter by preferred tile if requested and present
    if preferred_tile:
        per_items = [i for i in found_items if preferred_tile in i.id]
        if per_items:
            selected_items = per_items
            logger.info(
                f"Filtered {len(found_items)} item(s) down to {len(selected_items)} "
                f"matching preferred tile '{preferred_tile}'."
            )
        else:
            selected_items = found_items
            logger.info(
                f"Preferred tile '{preferred_tile}' not found in items; "
                f"using all {len(found_items)} available pass(es)."
            )
    else:
        selected_items = found_items

    # 3. Deduplicate / sort items by acquisition time
    # If multiple items share the same solar day, sort by cloud cover to prioritize cleanest
    selected_items.sort(
        key=lambda item: (
            item.datetime or datetime.min,
            float(item.properties.get("eo:cloud_cover", 100.0)),
        )
    )

    if max_passes is not None and max_passes > 0:
        selected_items = selected_items[:max_passes]
        logger.info(f"Capped timeseries cube to {len(selected_items)} pass(es).")

    logger.info(
        f"Loading timeseries cube from {len(selected_items)} STAC item(s) "
        f"at {resolution}m resolution into {crs}..."
    )

    # 4. Stream multispectral bands lazily into xarray Dataset
    target_chunks = chunks if chunks is not None else DEFAULT_CHUNKS
    raw_ds = load_stac_dataset(
        items=selected_items,
        bands=DEFAULT_CUBE_BANDS,
        crs=crs,
        resolution=resolution,
        bbox=bbox,
        chunks=target_chunks,
        groupby="solar_day",
    )

    # 5. Generate dilated validity mask across 3D SCL array
    logger.info(f"Applying morphological cloud & shadow masking (dilation={dilate_pixels}px)...")
    valid_mask = create_valid_mask(raw_ds["SCL"], dilate_pixels=dilate_pixels)

    # 6. Calculate vectorized spectral indices (NDVI and NDBI) with mask applied
    logger.info("Computing vectorized NDVI and NDBI time-series arrays...")
    ndvi = calculate_ndvi(raw_ds, valid_mask=valid_mask)
    ndbi = calculate_ndbi(raw_ds, valid_mask=valid_mask)

    # 7. Construct final time-series cube Dataset
    cube = xr.Dataset(
        data_vars={
            "ndvi": ndvi,
            "ndbi": ndbi,
            "valid_mask": valid_mask,
        },
        coords=ndvi.coords,
        attrs={
            "title": "Haddu Multi-Temporal Sentinel-2 Indices Cube",
            "crs": crs,
            "resolution": resolution,
            "bbox": list(bbox),
            "start_date": start_str,
            "end_date": end_str,
            "num_passes": int(len(raw_ds.time)),
            "indices": ["ndvi", "ndbi"],
        },
    )

    # Ensure chronological ordering along time dimension
    cube = cube.sortby("time")

    if compute:
        logger.info("Materializing lazy timeseries cube into memory...")
        cube = cube.compute()

    logger.info(
        f"Successfully constructed timeseries cube: dimensions={dict(cube.sizes)}, "
        f"time_steps={len(cube.time)} passes."
    )
    return cube
