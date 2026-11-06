"""Core satellite STAC client and raster transformation modules."""

from haddu.core.stac_client import (
    PLANETARY_COMPUTER_STAC_URL,
    DEFAULT_COLLECTION,
    DEFAULT_AOI_BBOX,
    STACError,
    STACQueryError,
    get_stac_client,
    search_passes,
)
from haddu.core.loader import (
    DEFAULT_TARGET_BANDS,
    DEFAULT_CRS,
    DEFAULT_RESOLUTION,
    DEFAULT_CHUNKS,
    LoaderError,
    RasterLoadError,
    load_stac_dataset,
)

__all__ = [
    "PLANETARY_COMPUTER_STAC_URL",
    "DEFAULT_COLLECTION",
    "DEFAULT_AOI_BBOX",
    "STACError",
    "STACQueryError",
    "get_stac_client",
    "search_passes",
    "DEFAULT_TARGET_BANDS",
    "DEFAULT_CRS",
    "DEFAULT_RESOLUTION",
    "DEFAULT_CHUNKS",
    "LoaderError",
    "RasterLoadError",
    "load_stac_dataset",
]
