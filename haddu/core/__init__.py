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
from haddu.core.masking import (
    DEFAULT_INVALID_SCL_CLASSES,
    DEFAULT_VALID_SCL_CLASSES,
    DEFAULT_DILATE_PIXELS,
    create_valid_mask,
    apply_valid_mask,
)
from haddu.core.indices import (
    DEFAULT_EPSILON,
    align_to_grid,
    calculate_ndvi,
    calculate_ndre,
    calculate_nbr,
    calculate_dnbr,
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
    "DEFAULT_INVALID_SCL_CLASSES",
    "DEFAULT_VALID_SCL_CLASSES",
    "DEFAULT_DILATE_PIXELS",
    "create_valid_mask",
    "apply_valid_mask",
    "DEFAULT_EPSILON",
    "align_to_grid",
    "calculate_ndvi",
    "calculate_ndre",
    "calculate_nbr",
    "calculate_dnbr",
]
