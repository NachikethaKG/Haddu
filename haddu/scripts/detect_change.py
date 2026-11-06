"""CLI utility for bi-temporal satellite change detection and encroachment vectorization.

Compares two Sentinel-2 passes over Thirthahalli Taluk, identifies forest canopy
clearing events using morphological differencing, vectorizes change clusters to GeoJSON,
renders an annotated difference heatmap, and logs incidents to PostGIS.
"""

import argparse
from datetime import datetime, date, timedelta
import logging
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pystac
from shapely.geometry import Polygon
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from geoalchemy2.shape import from_shape

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
    RasterLoadError,
)
from haddu.core.change_detector import (
    DEFAULT_NDVI_THRESHOLD,
    DEFAULT_MIN_AREA_SQM,
    detect_canopy_loss,
    ChangeDetectionResult,
)
from haddu.core.vectorizer import vectorize_change_mask
from haddu.db.session import get_db, init_db, engine
from haddu.db.models import AOI, EncroachmentIncident

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("haddu.detect_change")


def parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")


def find_best_pass_near_date(
    target_date: date,
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
    search_window_days: int = 5,
    max_cloud: float = 20.0,
    preferred_tile: str = "T43PER",
) -> pystac.Item:
    """Find the clearest Sentinel-2 STAC item near a target date prioritizing T43PER.

    Args:
        target_date: Date around which to search.
        bbox: Target AOI bounding box [min_lon, min_lat, max_lon, max_lat].
        search_window_days: Days before and after target_date to include in query.
        max_cloud: Maximum cloud cover percentage allowed.
        preferred_tile: Preferred MGRS tile ID (T43PER covers full Thirthahalli Taluk).

    Returns:
        Best matched pystac.Item.

    Raises:
        ValueError: If no candidate passes are found within the search window.
    """
    start_date = (target_date - timedelta(days=search_window_days)).strftime("%Y-%m-%d")
    end_date = (target_date + timedelta(days=search_window_days)).strftime("%Y-%m-%d")

    logger.info(
        f"Searching Sentinel-2 passes for target date {target_date} "
        f"between {start_date} and {end_date} (max_cloud={max_cloud}%)..."
    )

    items = search_passes(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud=max_cloud,
    )

    if not items:
        # Retry with wider 14-day window if needed
        logger.warning(f"No passes found within {search_window_days} days. Retrying with 14-day window...")
        start_date_wide = (target_date - timedelta(days=14)).strftime("%Y-%m-%d")
        end_date_wide = (target_date + timedelta(days=14)).strftime("%Y-%m-%d")
        items = search_passes(
            bbox=bbox,
            start_date=start_date_wide,
            end_date=end_date_wide,
            max_cloud=max_cloud + 10.0,
        )

    if not items:
        raise ValueError(
            f"No suitable Sentinel-2 passes found near {target_date} "
            f"within AOI {bbox} and cloud threshold."
        )

    # Filter items containing preferred MGRS tile
    per_items = [item for item in items if preferred_tile in item.id]
    candidates = per_items if per_items else items

    # Sort candidates by temporal proximity to target_date and cloud cover
    def _rank_key(item: pystac.Item) -> Tuple[int, float]:
        item_dt = item.datetime.date() if item.datetime else target_date
        days_diff = abs((item_dt - target_date).days)
        cloud_pct = float(item.properties.get("eo:cloud_cover", 100.0))
        return (days_diff, cloud_pct)

    sorted_candidates = sorted(candidates, key=_rank_key)
    selected_item = sorted_candidates[0]

    sel_date = selected_item.datetime.strftime("%Y-%m-%d") if selected_item.datetime else "Unknown"
    sel_cloud = selected_item.properties.get("eo:cloud_cover", "N/A")
    logger.info(f"Selected pass: {selected_item.id} (Date: {sel_date}, Cloud: {sel_cloud}%)")
    return selected_item


def render_difference_heatmap(
    delta_ndvi: np.ndarray,
    change_mask: np.ndarray,
    pre_date_str: str,
    post_date_str: str,
    threshold: float,
    output_path: Path,
    num_patches: int,
    total_acres: float,
) -> Path:
    """Render and save an annotated difference heatmap PNG."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot delta NDVI with high contrast colormap
    # Positive delta (vegetation drop) rendered in warm oranges/reds
    masked_delta = np.ma.masked_invalid(delta_ndvi)
    im = ax.imshow(
        masked_delta,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.8,
        interpolation="bilinear",
    )

    # Highlight detected clearing patches with cyan/blue boundary contours
    if np.any(change_mask):
        ax.contour(
            change_mask,
            levels=[0.5],
            colors=["#00f0ff"],
            linewidths=1.5,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NDVI Drop (ΔNDVI = Pre - Post)", fontsize=11, fontweight="bold")

    ax.set_title(
        f"Haddu Forest Encroachment Heatmap: Thirthahalli Taluk\n"
        f"Pre: {pre_date_str} | Post: {post_date_str} | Threshold: ΔNDVI > {threshold}\n"
        f"Detected Clearing: {total_acres:.2f} acres across {num_patches} cluster(s)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info(f"Saved difference heatmap to: {output_path.resolve()}")
    return output_path


def record_incidents_to_db(
    gdf,
    pre_date: date,
    post_date: date,
    aoi_name: str,
    bbox: Sequence[float],
) -> int:
    """Commit detected vector encroachment incidents to PostGIS database."""
    if gdf.empty:
        logger.info("No detected incidents to record in PostGIS.")
        return 0

    try:
        logger.info("Connecting to PostGIS database to record incidents...")
        init_db(engine)

        with get_db() as session:
            # Ensure AOI exists
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

            inserted_count = 0
            for _, row in gdf.iterrows():
                poly_geom = row.geometry
                incident = EncroachmentIncident(
                    aoi_id=aoi.id,
                    pre_date=pre_date,
                    post_date=post_date,
                    area_sqm=float(row["area_sqm"]),
                    area_acres=float(row["area_acres"]),
                    centroid_lat=float(row["centroid_lat"]),
                    centroid_lon=float(row["centroid_lon"]),
                    mean_delta_ndvi=float(row["mean_delta_ndvi"]) if row["mean_delta_ndvi"] is not None else None,
                    max_delta_ndvi=float(row["max_delta_ndvi"]) if row["max_delta_ndvi"] is not None else None,
                    geom=from_shape(poly_geom, srid=4326),
                    properties_json={
                        "patch_id": int(row["patch_id"]),
                        "pre_date": pre_date.isoformat(),
                        "post_date": post_date.isoformat(),
                    },
                )
                session.add(incident)
                inserted_count += 1

            session.commit()
            logger.info(f"Successfully committed {inserted_count} incident records to PostGIS.")
            return inserted_count

    except OperationalError as exc:
        logger.warning(
            f"PostGIS database unreachable ({exc.orig}). "
            "Proceeding without database persistence. "
            "To persist to PostGIS, run: 'docker compose up -d db'"
        )
        return 0
    except Exception as exc:
        logger.warning(f"Database insertion failed: {exc}. Proceeding.")
        return 0


def run_change_detection_pipeline(
    pre_date: date,
    post_date: date,
    threshold: float = DEFAULT_NDVI_THRESHOLD,
    min_area_sqm: float = DEFAULT_MIN_AREA_SQM,
    output_dir: Path = Path("./output"),
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
    max_cloud: float = 20.0,
    aoi_name: str = "Thirthahalli Taluk",
    skip_db: bool = False,
) -> Tuple[Path, Path, int]:
    """Execute end-to-end change detection, vectorization, and persistence pipeline.

    Returns:
        Tuple of (geojson_path, heatmap_path, num_incidents_recorded).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_date_str = pre_date.strftime("%Y-%m-%d")
    post_date_str = post_date.strftime("%Y-%m-%d")

    logger.info("=" * 65)
    logger.info("HADDU SATELLITE CHANGE DETECTION ENGINE")
    logger.info(f"Target AOI       : {aoi_name} {list(bbox)}")
    logger.info(f"Pre-Scene Date   : {pre_date_str}")
    logger.info(f"Post-Scene Date  : {post_date_str}")
    logger.info(f"NDVI Threshold   : {threshold}")
    logger.info(f"Min Cluster Area : {min_area_sqm} m²")
    logger.info("=" * 65)

    # 1. Search for best Sentinel-2 passes
    pre_item = find_best_pass_near_date(pre_date, bbox=bbox, max_cloud=max_cloud)
    post_item = find_best_pass_near_date(post_date, bbox=bbox, max_cloud=max_cloud)

    actual_pre_date = pre_item.datetime.strftime("%Y-%m-%d") if pre_item.datetime else pre_date_str
    actual_post_date = post_item.datetime.strftime("%Y-%m-%d") if post_item.datetime else post_date_str

    # 2. Lazily stream multispectral bands (B04, B08, SCL)
    target_bands = ["B04", "B08", "SCL"]
    logger.info(f"Streaming pre-scene bands {target_bands} (CRS: {DEFAULT_CRS})...")
    pre_ds = load_stac_dataset(
        items=[pre_item],
        bands=target_bands,
        crs=DEFAULT_CRS,
        resolution=DEFAULT_RESOLUTION,
        bbox=bbox,
        chunks=DEFAULT_CHUNKS,
    ).compute()

    logger.info(f"Streaming post-scene bands {target_bands} (CRS: {DEFAULT_CRS})...")
    post_ds = load_stac_dataset(
        items=[post_item],
        bands=target_bands,
        crs=DEFAULT_CRS,
        resolution=DEFAULT_RESOLUTION,
        bbox=bbox,
        chunks=DEFAULT_CHUNKS,
    ).compute()

    # 3. Detect canopy loss with morphological cleanup
    logger.info("Running bi-temporal difference detection and morphological opening...")
    change_result: ChangeDetectionResult = detect_canopy_loss(
        pre_ds=pre_ds,
        post_ds=post_ds,
        ndvi_threshold=threshold,
        min_area_sqm=min_area_sqm,
        dilate_pixels=2,
        check_spatial=True,
    )

    num_patches = change_result.num_patches
    total_acres = change_result.total_area_acres
    total_sqm = change_result.total_area_sqm

    logger.info(
        f"Detection Result : {num_patches} cluster(s) detected, "
        f"{total_sqm:,.2f} m² ({total_acres:.2f} acres) total canopy loss."
    )

    # 4. Vectorize change mask to GeoDataFrame (EPSG:4326)
    logger.info("Vectorizing change clusters into GeoDataFrame...")
    gdf = vectorize_change_mask(
        change_input=change_result,
        source_crs=DEFAULT_CRS,
        target_crs="EPSG:4326",
    )

    # 5. Export GeoJSON
    geojson_filename = f"change_{pre_date_str}_{post_date_str}.geojson"
    geojson_path = out_dir / geojson_filename
    gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info(f"Saved GeoJSON polygons to : {geojson_path.resolve()}")

    # 6. Render Difference Heatmap PNG
    heatmap_filename = f"change_heatmap_{pre_date_str}_{post_date_str}.png"
    heatmap_path = out_dir / heatmap_filename
    render_difference_heatmap(
        delta_ndvi=change_result.delta_ndvi.values,
        change_mask=change_result.change_mask.values,
        pre_date_str=actual_pre_date,
        post_date_str=actual_post_date,
        threshold=threshold,
        output_path=heatmap_path,
        num_patches=num_patches,
        total_acres=total_acres,
    )

    # 7. Record incidents to PostGIS database
    db_count = 0
    if not skip_db:
        db_count = record_incidents_to_db(
            gdf=gdf,
            pre_date=pre_date,
            post_date=post_date,
            aoi_name=aoi_name,
            bbox=bbox,
        )

    # Print summary
    print("\n" + "=" * 65)
    print("HADDU CHANGE DETECTION PIPELINE COMPLETE")
    print("=" * 65)
    print(f"Pre-Scene Pass    : {pre_item.id} ({actual_pre_date})")
    print(f"Post-Scene Pass   : {post_item.id} ({actual_post_date})")
    print(f"Clusters Detected : {num_patches}")
    print(f"Total Loss Area   : {total_sqm:,.2f} m² ({total_acres:.2f} acres)")
    print(f"GeoJSON Output    : {geojson_path.resolve()}")
    print(f"Heatmap Output    : {heatmap_path.resolve()}")
    print(f"Database Records  : {db_count} incident(s) committed")
    print("=" * 65 + "\n")

    return geojson_path, heatmap_path, db_count


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="haddu.scripts.detect_change",
        description="Bi-temporal satellite change detection and forest encroachment vectorizer for Haddu.",
    )
    parser.add_argument(
        "--pre-date",
        type=parse_date,
        required=True,
        help="Pre-clearing baseline date in YYYY-MM-DD format (e.g. 2025-12-15).",
    )
    parser.add_argument(
        "--post-date",
        type=parse_date,
        required=True,
        help="Post-clearing monitoring date in YYYY-MM-DD format (e.g. 2026-03-15).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_NDVI_THRESHOLD,
        help=f"NDVI drop threshold to flag canopy loss (default: {DEFAULT_NDVI_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-area-sqm",
        type=float,
        default=DEFAULT_MIN_AREA_SQM,
        help=f"Minimum clearing patch area in m² (default: {DEFAULT_MIN_AREA_SQM}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory to save output GeoJSON and heatmap PNG (default: ./output).",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20.0,
        help="Maximum cloud cover percentage allowed in Sentinel-2 search (default: 20.0).",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        default=False,
        help="Skip recording incident records to PostGIS database.",
    )
    return parser


def main():
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        run_change_detection_pipeline(
            pre_date=args.pre_date,
            post_date=args.post_date,
            threshold=args.threshold,
            min_area_sqm=args.min_area_sqm,
            output_dir=args.output_dir,
            max_cloud=args.max_cloud,
            skip_db=args.skip_db,
        )
    except Exception as exc:
        logger.error(f"Change detection failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
