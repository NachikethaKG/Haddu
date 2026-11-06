"""Windowed cloud-native raster ingestion engine for Sentinel-2 imagery.

Uses odc-stac to stream multispectral bands on-demand via HTTP byte-range
requests directly into dask-backed xarray Datasets without downloading full scenes.
"""

import logging
from typing import Dict, List, Optional, Sequence, Union
import pystac
import xarray as xr
import odc.stac
from rasterio.errors import RasterioIOError

from haddu.core.stac_client import DEFAULT_AOI_BBOX

logger = logging.getLogger(__name__)

# Standard target bands for forest change detection & cloud masking
DEFAULT_TARGET_BANDS = [
    "B02",  # Blue (10m)
    "B03",  # Green (10m)
    "B04",  # Red (10m)
    "B08",  # NIR (10m)
    "B11",  # SWIR1 (20m -> resampled to 10m)
    "B12",  # SWIR2 (20m -> resampled to 10m)
    "SCL",  # Scene Classification Layer (20m -> resampled to 10m)
]

DEFAULT_CRS = "EPSG:32643"  # UTM Zone 43N
DEFAULT_RESOLUTION = 10     # 10 meters ground resolution
DEFAULT_CHUNKS = {"x": 2048, "y": 2048}


class LoaderError(Exception):
    """Base exception for raster loading and ingestion errors."""
    pass


class RasterLoadError(LoaderError):
    """Raised when odc-stac fails to stream or decode target bands via HTTP."""
    pass


def load_stac_dataset(
    items: Sequence[pystac.Item],
    bands: Optional[Sequence[str]] = None,
    crs: str = DEFAULT_CRS,
    resolution: Union[int, float] = DEFAULT_RESOLUTION,
    bbox: Optional[Sequence[float]] = DEFAULT_AOI_BBOX,
    chunks: Optional[Dict[str, int]] = None,
    groupby: Optional[str] = "solar_day",
    **kwargs,
) -> xr.Dataset:
    """Lazily load target bands from signed STAC items into an xarray Dataset.

    Streams data via HTTP range requests projected into UTM Zone 43N.

    Args:
        items: Sequence of signed pystac.Item instances.
        bands: List of band identifiers to load (default: B02, B03, B04, B08, B11, B12, SCL).
        crs: Target Coordinate Reference System (default: 'EPSG:32643').
        resolution: Target pixel resolution in CRS units (default: 10 meters).
        bbox: Target bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
        chunks: Dask chunk size mapping (default: {'x': 2048, 'y': 2048}).
        groupby: Temporal grouping strategy ('solar_day' or None).
        **kwargs: Extra parameters forwarded to odc.stac.load.

    Returns:
        Lazy dask-backed xarray.Dataset with spatial coordinates in target CRS.

    Raises:
        ValueError: If items list is empty or invalid.
        RasterLoadError: If HTTP byte-range request or raster decode fails.
    """
    if not items:
        raise ValueError("Cannot load dataset: items list is empty.")

    target_bands = list(bands or DEFAULT_TARGET_BANDS)
    target_chunks = chunks if chunks is not None else DEFAULT_CHUNKS

    logger.info(
        f"Loading {len(items)} STAC item(s) with bands={target_bands}, "
        f"crs={crs}, resolution={resolution}m, chunks={target_chunks}"
    )

    try:
        ds = odc.stac.load(
            items=items,
            bands=target_bands,
            crs=crs,
            resolution=resolution,
            bbox=list(bbox) if bbox else None,
            chunks=target_chunks,
            groupby=groupby,
            **kwargs,
        )

        # Ensure spatial_ref coordinate exists
        logger.info(
            f"Successfully constructed lazy xarray.Dataset with dimensions: {dict(ds.sizes)}"
        )
        return ds

    except (RasterioIOError, RuntimeError, OSError) as io_err:
        raise RasterLoadError(
            f"Failed to read raster data via HTTP byte-range request: {io_err}"
        ) from io_err
    except Exception as err:
        raise RasterLoadError(
            f"Unexpected error while loading STAC raster bands: {err}"
        ) from err
