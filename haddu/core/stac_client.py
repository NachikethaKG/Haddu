"""STAC client for querying Sentinel-2 scenes via Microsoft Planetary Computer.

Uses pystac-client with planetary_computer.sign_inplace for authenticated,
on-demand access to cloud-native Sentinel-2 L2A data.
"""

import logging
from datetime import date, datetime
from typing import List, Sequence, Union
import pystac
import pystac_client
import planetary_computer
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

PLANETARY_COMPUTER_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"
DEFAULT_AOI_BBOX = (75.05, 13.55, 75.40, 13.85)  # Thirthahalli Taluk


class STACError(Exception):
    """Base exception for STAC query and network operations."""
    pass


class STACQueryError(STACError):
    """Raised when STAC query fails or encounters an unrecoverable API error."""
    pass


def get_stac_client(
    catalog_url: str = PLANETARY_COMPUTER_STAC_URL,
) -> pystac_client.Client:
    """Connect to Microsoft Planetary Computer STAC catalog with sign_inplace modifier."""
    try:
        return pystac_client.Client.open(
            catalog_url,
            modifier=planetary_computer.sign_inplace,
        )
    except Exception as err:
        raise STACQueryError(
            f"Failed to connect to STAC catalog at {catalog_url}: {err}"
        ) from err


def format_date(d: Union[str, date, datetime]) -> str:
    """Convert input date or datetime to ISO YYYY-MM-DD string."""
    if isinstance(d, (datetime, date)):
        return d.strftime("%Y-%m-%d")
    return str(d).strip()


def search_passes(
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
    start_date: Union[str, date, datetime] = "2024-01-01",
    end_date: Union[str, date, datetime] = "2024-02-15",
    max_cloud: Union[int, float] = 20,
    collection: str = DEFAULT_COLLECTION,
    catalog_url: str = PLANETARY_COMPUTER_STAC_URL,
    limit: int | None = None,
) -> List[pystac.Item]:
    """Search for Sentinel-2 satellite passes intersecting the target AOI bounding box.

    Args:
        bbox: Target bounding box (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        start_date: Start date (YYYY-MM-DD, date, or datetime).
        end_date: End date (YYYY-MM-DD, date, or datetime).
        max_cloud: Maximum allowable cloud cover percentage (0-100).
        collection: STAC collection ID (default: "sentinel-2-l2a").
        catalog_url: Planetary Computer STAC endpoint.
        limit: Optional maximum number of items to return.

    Returns:
        List of signed pystac.Item objects sorted chronologically.

    Raises:
        STACQueryError: If the STAC catalog cannot be reached or returns an error.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain exactly 4 coordinates (min_lon, min_lat, max_lon, max_lat), got: {bbox}")

    start_str = format_date(start_date)
    end_str = format_date(end_date)
    date_range = f"{start_str}/{end_str}"

    client = get_stac_client(catalog_url)

    logger.info(
        f"Searching STAC collection '{collection}' for bbox={bbox}, "
        f"dates={date_range}, max_cloud={max_cloud}%"
    )

    try:
        search = client.search(
            collections=[collection],
            bbox=list(bbox),
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": float(max_cloud)}},
            limit=limit,
        )
        items = list(search.items())

        # Sort items chronologically by capture datetime
        items.sort(key=lambda item: item.datetime or datetime.min)

        if not items:
            logger.warning(
                f"No Sentinel-2 passes found for bbox {bbox} between {date_range} "
                f"with cloud cover < {max_cloud}%."
            )
        else:
            logger.info(f"Found {len(items)} matching Sentinel-2 passes.")

        return items

    except RequestException as net_err:
        raise STACQueryError(f"Network error while querying Planetary Computer STAC: {net_err}") from net_err
    except Exception as err:
        raise STACQueryError(f"Unexpected error executing STAC search: {err}") from err
