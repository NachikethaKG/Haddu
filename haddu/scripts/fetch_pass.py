"""CLI script to fetch Sentinel-2 passes for Thirthahalli and export cloud-masked true-color preview.

Pulls windowed scenes via Microsoft Planetary Computer STAC, evaluates SCL cloud masks,
saves true-color composite PNG, and logs metadata into PostGIS.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, Optional, Sequence
import dask
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from geoalchemy2.shape import from_shape
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from haddu.core.stac_client import (
    DEFAULT_AOI_BBOX,
    search_passes,
    STACError,
)
from haddu.core.loader import (
    load_stac_dataset,
    LoaderError,
)
from haddu.db.session import engine, init_db, get_db
from haddu.db.models import AOI, SatellitePass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("haddu.fetch_pass")


def fetch_and_export_pass(
    start_date: str = "2026-01-01",
    end_date: str = "2026-02-15",
    max_cloud: int = 15,
    output_dir: str = "./output",
    resolution: int = 20,
    aoi_name: str = "Thirthahalli Taluk",
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
) -> Dict[str, Any]:
    """Execute end-to-end extraction, SCL cloud masking, image export, and database logging."""
    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("Haddu - Sentinel-2 Extraction & Visual Sanity Engine")
    logger.info(f"Target AOI     : {aoi_name} (bbox: {list(bbox)})")
    logger.info(f"Date Range     : {start_date} to {end_date} (Max Cloud: {max_cloud}%)")
    logger.info(f"Resolution     : {resolution}m | Output Directory: {out_path.resolve()}")
    logger.info("=" * 65)

    # 1. Search for available passes in STAC catalog
    logger.info("Querying Microsoft Planetary Computer STAC catalog...")
    items = search_passes(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud=max_cloud,
    )

    if not items:
        raise RuntimeError(
            f"No Sentinel-2 passes found matching AOI {bbox} between {start_date} and {end_date}."
        )

    # Prioritize items whose footprint covers the centroid of the target AOI
    center_lon = (bbox[0] + bbox[2]) / 2.0
    center_lat = (bbox[1] + bbox[3]) / 2.0
    centroid_items = [
        it for it in items
        if it.bbox
        and it.bbox[0] <= center_lon <= it.bbox[2]
        and it.bbox[1] <= center_lat <= it.bbox[3]
    ]

    best_item = min(
        centroid_items or items,
        key=lambda it: it.properties.get("eo:cloud_cover", 100.0),
    )

    cloud_cover = float(best_item.properties.get("eo:cloud_cover", 0.0))
    platform = best_item.properties.get("platform", "Sentinel-2")
    item_date = best_item.datetime.strftime("%Y-%m-%d") if best_item.datetime else start_date

    logger.info(
        f"Selected pass  : {best_item.id}\n"
        f"  Date         : {item_date}\n"
        f"  Cloud Cover  : {cloud_cover:.2f}%\n"
        f"  Platform     : {platform}"
    )

    # 2. Windowed raster load using odc-stac (Red, Green, Blue, SCL)
    logger.info("Streaming windowed bands via HTTP byte-range requests...")
    t_stream = time.time()

    ds = load_stac_dataset(
        items=[best_item],
        bands=["B04", "B03", "B02", "SCL"],
        crs="EPSG:32643",
        resolution=resolution,
        bbox=bbox,
        chunks={"x": 2048, "y": 2048},
    )

    # Parallel chunk compute using multithreading
    with dask.config.set(num_workers=8):
        ds_computed = ds.compute()

    logger.info(f"Stream and projection complete in {time.time() - t_stream:.2f}s")

    r = ds_computed.B04.isel(time=0).values.astype(np.float32)
    g = ds_computed.B03.isel(time=0).values.astype(np.float32)
    b = ds_computed.B02.isel(time=0).values.astype(np.float32)
    scl = ds_computed.SCL.isel(time=0).values

    # 3. SCL mask evaluation: 3 (cloud shadow), 8 (cloud med), 9 (cloud high), 10 (cirrus)
    invalid_mask = np.isin(scl, [3, 8, 9, 10])
    total_pixels = scl.size
    valid_pixels = int(np.count_nonzero(~invalid_mask))
    valid_pct = (valid_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    logger.info(
        f"SCL Evaluation : {valid_pct:.2f}% valid cloud-free pixels "
        f"({valid_pixels:,}/{total_pixels:,})"
    )

    # 4. Normalize RGB (0-10000 -> 0-1) and enhance contrast for vivid preview
    rgb = np.stack([r, g, b], axis=-1) / 10000.0
    rgb = np.clip(rgb, 0.0, 1.0)

    # Apply 3.2x reflectance stretch for crisp hill contours and river visibility
    rgb_enhanced = np.clip(rgb * 3.2, 0.0, 1.0)

    # Apply cloud mask overlay (clouds rendered in light neutral tone)
    rgb_enhanced[invalid_mask] = 1.0

    # Save true-color preview PNG
    preview_filename = f"thirthahalli_rgb_{item_date}.png"
    preview_filepath = out_path / preview_filename
    plt.imsave(preview_filepath, rgb_enhanced)

    logger.info(f"Preview Image  : Saved composite to {preview_filepath.resolve()}")

    # 5. Record metadata into PostGIS database
    db_recorded = False
    try:
        logger.info("Attempting PostGIS metadata registration...")
        init_db(engine)

        with get_db() as session:
            # Verify or register AOI
            aoi = session.scalar(select(AOI).where(AOI.name == aoi_name))
            if not aoi:
                poly = Polygon([
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[0], bbox[3]),
                    (bbox[0], bbox[1]),
                ])
                aoi = AOI(
                    name=aoi_name,
                    category="Taluk",
                    geom=from_shape(poly, srid=4326),
                )
                session.add(aoi)
                session.flush()

            # Record or update SatellitePass
            existing_pass = session.scalar(
                select(SatellitePass).where(SatellitePass.stac_item_id == best_item.id)
            )

            pass_date = (
                best_item.datetime.date()
                if best_item.datetime
                else datetime.strptime(item_date, "%Y-%m-%d").date()
            )

            metadata_dict = {
                "stac_item_id": best_item.id,
                "platform": platform,
                "datetime": best_item.datetime.isoformat() if best_item.datetime else None,
                "cloud_cover_pct": cloud_cover,
                "valid_cloud_free_pct": round(valid_pct, 2),
                "preview_image": str(preview_filepath.resolve()),
                "bbox": list(bbox),
                "resolution": resolution,
                "epsg": 32643,
            }

            if existing_pass:
                existing_pass.cloud_cover_pct = cloud_cover
                existing_pass.metadata_json = metadata_dict
                logger.info(f"Updated existing SatellitePass record (ID: {existing_pass.id})")
            else:
                sat_pass = SatellitePass(
                    aoi_id=aoi.id,
                    platform=platform,
                    pass_date=pass_date,
                    cloud_cover_pct=cloud_cover,
                    stac_item_id=best_item.id,
                    metadata_json=metadata_dict,
                )
                session.add(sat_pass)
                session.flush()
                logger.info(f"Inserted new SatellitePass record (ID: {sat_pass.id}) into PostGIS.")

            db_recorded = True

    except (OperationalError, Exception) as db_err:
        logger.warning(
            f"Database logging skipped or failed: {db_err}. "
            "To enable database logging, start PostGIS: docker compose up -d"
        )

    elapsed_time = time.time() - t_start
    logger.info("=" * 65)
    logger.info(f"Pipeline Completed in {elapsed_time:.2f} seconds.")
    logger.info("=" * 65)

    return {
        "stac_item_id": best_item.id,
        "date": item_date,
        "cloud_cover_pct": cloud_cover,
        "valid_cloud_free_pct": valid_pct,
        "preview_filepath": str(preview_filepath),
        "db_recorded": db_recorded,
        "elapsed_seconds": elapsed_time,
    }


def main() -> None:
    """CLI entrypoint for fetching Sentinel-2 passes and exporting visual preview."""
    parser = argparse.ArgumentParser(
        description="Haddu - Fetch Sentinel-2 satellite pass and generate cloud-masked true-color preview."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-01-01",
        help="Start date in YYYY-MM-DD format (default: 2026-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-02-15",
        help="End date in YYYY-MM-DD format (default: 2026-02-15)",
    )
    parser.add_argument(
        "--max-cloud",
        type=int,
        default=15,
        help="Maximum cloud cover percentage filter (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save preview PNG (default: ./output)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=20,
        help="Target ground resolution in meters (default: 20)",
    )

    args = parser.parse_args()

    try:
        fetch_and_export_pass(
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud=args.max_cloud,
            output_dir=args.output_dir,
            resolution=args.resolution,
        )
    except Exception as err:
        logger.error(f"Execution failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
