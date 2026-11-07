"""Integrated multi-sensor legal audit CLI and violation detection pipeline.

Executes optical (Sentinel-2) or all-weather radar (Sentinel-1 SAR) canopy disturbance
detection, cross-references detected clearing polygons against gazetted Reserve Forests
and statutory 50m riparian buffer corridors in PostGIS, categorizes violations into
legal enforcement risk tiers (CRITICAL, HIGH, MODERATE), outputs an annotated GeoJSON,
and persists incident violations to PostGIS.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import pystac
from geoalchemy2.shape import from_shape
from shapely.geometry import Polygon
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from haddu.core.change_detector import (
    DEFAULT_MIN_AREA_SQM,
    DEFAULT_NDVI_THRESHOLD,
    detect_canopy_loss,
)
from haddu.core.classifier import (
    ViolationTier,
    classify_incident_violations,
    summarize_violations,
)
from haddu.core.loader import (
    DEFAULT_CHUNKS,
    DEFAULT_CRS,
    load_stac_dataset,
)
from haddu.core.sar_change import (
    DEFAULT_FILTER_SIZE,
    DEFAULT_SPECKLE_FILTER,
    DEFAULT_VH_DROP_DB,
    detect_sar_deforestation,
)
from haddu.core.sar_loader import (
    DEFAULT_SAR_ORBIT,
    DEFAULT_SAR_RESOLUTION,
    fetch_sar_passes,
    search_sar_passes,
)
from haddu.core.stac_client import DEFAULT_AOI_BBOX
from haddu.core.vectorizer import vectorize_change_mask
from haddu.db.models import AOI, EncroachmentIncident
from haddu.db.session import engine, get_db, init_db
from haddu.scripts.detect_change import find_best_pass_near_date

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("haddu.audit")


def parse_date(date_str: str) -> date:
    """Parse YYYY-MM-DD date string."""
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")


def find_best_sar_pass_near_date(
    target_date: date,
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
    search_window_days: int = 15,
    orbit_direction: str | None = DEFAULT_SAR_ORBIT,
) -> pystac.Item:
    """Find the closest Sentinel-1 SAR STAC item near a target date.

    Args:
        target_date: Center date around which to search.
        bbox: Target AOI bounding box [min_lon, min_lat, max_lon, max_lat].
        search_window_days: Days before and after target_date to include in query.
        orbit_direction: Orbit direction filter ("DESCENDING", "ASCENDING", or "ANY").

    Returns:
        Best matched pystac.Item.

    Raises:
        ValueError: If no candidate passes are found within the search window.
    """
    start_date = (target_date - timedelta(days=search_window_days)).strftime("%Y-%m-%d")
    end_date = (target_date + timedelta(days=search_window_days)).strftime("%Y-%m-%d")

    logger.info(
        f"Searching Sentinel-1 SAR passes near {target_date} "
        f"between {start_date} and {end_date} (orbit={orbit_direction})..."
    )

    items = search_sar_passes(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        orbit_direction=orbit_direction,
    )

    if not items and orbit_direction != "ANY":
        logger.warning(f"No passes found with orbit={orbit_direction}. Retrying with any orbit...")
        items = search_sar_passes(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            orbit_direction="ANY",
        )

    if not items:
        # Retry with wider 30-day window
        logger.warning(f"No SAR passes found within {search_window_days} days. Retrying with 30-day window...")
        start_wide = (target_date - timedelta(days=30)).strftime("%Y-%m-%d")
        end_wide = (target_date + timedelta(days=30)).strftime("%Y-%m-%d")
        items = search_sar_passes(
            bbox=bbox,
            start_date=start_wide,
            end_date=end_wide,
            orbit_direction="ANY",
        )

    if not items:
        raise ValueError(
            f"No suitable Sentinel-1 SAR passes found near {target_date} within AOI {bbox}."
        )

    def _rank_key(item: pystac.Item) -> int:
        item_dt = item.datetime.date() if item.datetime else target_date
        return abs((item_dt - target_date).days)

    sorted_items = sorted(items, key=_rank_key)
    selected = sorted_items[0]
    sel_date = selected.datetime.strftime("%Y-%m-%d") if selected.datetime else "Unknown"
    logger.info(f"Selected SAR pass: {selected.id} (Capture Date: {sel_date})")
    return selected


def record_audit_violations_to_db(
    classified_gdf: gpd.GeoDataFrame,
    sensor: str,
    pre_date: date,
    post_date: date,
    aoi_name: str,
    bbox: Sequence[float],
) -> int:
    """Persist classified legal violation incidents to PostGIS database."""
    if classified_gdf.empty:
        logger.info("No detected incidents to commit to database.")
        return 0

    try:
        logger.info("Connecting to PostGIS database to persist legal violation incidents...")
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
            for idx, row in classified_gdf.iterrows():
                poly_geom = row.geometry
                reasons_val = row.get("violation_reasons", [])
                if isinstance(reasons_val, (list, tuple)):
                    reasons_list = list(reasons_val)
                else:
                    reasons_list = [str(reasons_val)]

                incident = EncroachmentIncident(
                    aoi_id=aoi.id,
                    pre_date=pre_date,
                    post_date=post_date,
                    area_sqm=float(row.get("area_sqm", 0.0)),
                    area_acres=float(row.get("area_acres", 0.0)),
                    centroid_lat=float(row.get("centroid_lat", 0.0)),
                    centroid_lon=float(row.get("centroid_lon", 0.0)),
                    mean_delta_ndvi=(
                        float(row["mean_delta_ndvi"])
                        if row.get("mean_delta_ndvi") is not None
                        else None
                    ),
                    max_delta_ndvi=(
                        float(row["max_delta_ndvi"])
                        if row.get("max_delta_ndvi") is not None
                        else None
                    ),
                    geom=from_shape(poly_geom, srid=4326),
                    violation_tier=str(row.get("violation_tier", ViolationTier.MODERATE)),
                    encroached_rf_name=row.get("encroached_rf_name"),
                    rf_area_acres=(
                        float(row["rf_area_acres"])
                        if row.get("rf_area_acres") is not None
                        else 0.0
                    ),
                    riparian_breach=bool(row.get("riparian_breach", False)),
                    sensor=sensor,
                    properties_json={
                        "sensor": sensor,
                        "patch_id": int(row.get("patch_id", idx)),
                        "violation_tier": row.get("violation_tier"),
                        "esz_breach": bool(row.get("esz_breach", False)),
                        "distance_to_rf_m": float(row.get("distance_to_rf_m", 0.0)),
                        "nearest_rf_name": row.get("nearest_rf_name"),
                        "violation_reasons": reasons_list,
                        "pre_date": pre_date.isoformat(),
                        "post_date": post_date.isoformat(),
                    },
                )
                session.add(incident)
                inserted_count += 1

            session.commit()
            logger.info(f"Successfully committed {inserted_count} incident records with legal tiers to PostGIS.")
            return inserted_count

    except OperationalError as exc:
        logger.warning(
            f"PostGIS database unreachable ({exc.orig}). "
            "Proceeding without database persistence. "
            "To persist to PostGIS, ensure Docker is running: 'docker compose up -d db'"
        )
        return 0
    except (KeyError, TypeError, ValueError, OSError) as exc:
        logger.warning(f"Database insertion failed: {exc}. Proceeding.")
        return 0


def run_audit_pipeline(
    sensor: str,
    pre_date: date,
    post_date: date,
    output_dir: Path = Path("./output/audit"),
    aoi_name: str = "Thirthahalli Taluk",
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
    threshold: float | None = None,
    min_area_sqm: float = DEFAULT_MIN_AREA_SQM,
    resolution: float | None = None,
    max_cloud: float = 20.0,
    speckle_filter: str = DEFAULT_SPECKLE_FILTER,
    filter_size: int = DEFAULT_FILTER_SIZE,
    skip_db: bool = False,
    session_override: Any | None = None,
    boundaries_dir: Path | None = None,
) -> tuple[Path, gpd.GeoDataFrame, dict[str, Any]]:
    """Execute end-to-end multi-sensor audit and legal violation classification pipeline.

    Args:
        sensor: Sensor mode ('s1'/'sar' or 's2'/'optical').
        pre_date: Baseline target date.
        post_date: Monitoring target date.
        output_dir: Directory where GeoJSON and summary reports will be stored.
        aoi_name: Target AOI description.
        bbox: Target AOI bounding box [min_lon, min_lat, max_lon, max_lat].
        threshold: Backscatter drop threshold in dB (s1) or NDVI drop threshold (s2).
        min_area_sqm: Minimum clearing patch area in m² (default: 500.0).
        resolution: Spatial sampling resolution in meters.
        max_cloud: Maximum allowed cloud cover pct for optical search (s2).
        speckle_filter: Speckle filter algorithm for SAR ('median' or 'lee').
        filter_size: Spatial window kernel size for SAR speckle filtering.
        skip_db: If True, skips database persistence.
        session_override: Optional SQLAlchemy session for testing or dependency injection.
        boundaries_dir: Optional custom path to boundary GeoJSON files.

    Returns:
        Tuple of (geojson_path, classified_gdf, summary_metrics).
    """
    sensor_norm = sensor.strip().lower()
    if sensor_norm in ("sar", "s1", "sentinel-1"):
        sensor_type = "s1"
        active_threshold = threshold if threshold is not None else DEFAULT_VH_DROP_DB
        active_res = resolution if resolution is not None else DEFAULT_SAR_RESOLUTION
    elif sensor_norm in ("optical", "s2", "sentinel-2"):
        sensor_type = "s2"
        active_threshold = threshold if threshold is not None else DEFAULT_NDVI_THRESHOLD
        active_res = resolution if resolution is not None else 10.0
    else:
        raise ValueError(f"Unsupported sensor '{sensor}'. Choose 's1' (SAR) or 's2' (Optical).")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_date_str = pre_date.strftime("%Y-%m-%d")
    post_date_str = post_date.strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"HADDU LEGAL VIOLATION AUDIT ENGINE [{sensor_type.upper()}]")
    logger.info(f"Target AOI       : {aoi_name} {list(bbox)}")
    logger.info(f"Sensor Platform  : {sensor_type.upper()} ({'Sentinel-1 SAR Radar' if sensor_type == 's1' else 'Sentinel-2 Optical'})")
    logger.info(f"Pre-Scene Date   : {pre_date_str}")
    logger.info(f"Post-Scene Date  : {post_date_str}")
    logger.info(f"Change Threshold : {active_threshold} ({'dB drop' if sensor_type == 's1' else 'ΔNDVI'})")
    logger.info(f"Min Patch Area   : {min_area_sqm} m²")
    logger.info(f"Resolution       : {active_res} m")
    logger.info("=" * 70)

    # 1. Acquire Scenes and Execute Change Detection
    if sensor_type == "s1":
        logger.info("Step 1/4: Ingesting Sentinel-1 SAR radar passes through monsoon cloud cover...")
        pre_item = find_best_sar_pass_near_date(pre_date, bbox=bbox)
        post_item = find_best_sar_pass_near_date(post_date, bbox=bbox)

        actual_pre_date = pre_item.datetime.strftime("%Y-%m-%d") if pre_item.datetime else pre_date_str
        actual_post_date = post_item.datetime.strftime("%Y-%m-%d") if post_item.datetime else post_date_str

        logger.info(f"Streaming pre-scene SAR {pre_item.id}...")
        pre_ds = fetch_sar_passes(
            items=[pre_item],
            bbox=bbox,
            resolution=active_res,
            chunks=DEFAULT_CHUNKS,
        ).compute()

        logger.info(f"Streaming post-scene SAR {post_item.id}...")
        post_ds = fetch_sar_passes(
            items=[post_item],
            bbox=bbox,
            resolution=active_res,
            chunks=DEFAULT_CHUNKS,
        ).compute()

        logger.info("Executing polarimetric radar deforestation detection...")
        detection_res = detect_sar_deforestation(
            pre_sar_ds=pre_ds,
            post_sar_ds=post_ds,
            vh_drop_db=active_threshold,
            min_area_sqm=min_area_sqm,
            filter_method=speckle_filter,
            filter_size=filter_size,
        )
    else:
        logger.info("Step 1/4: Ingesting Sentinel-2 optical multispectral passes...")
        pre_item = find_best_pass_near_date(pre_date, bbox=bbox, max_cloud=max_cloud)
        post_item = find_best_pass_near_date(post_date, bbox=bbox, max_cloud=max_cloud)

        actual_pre_date = pre_item.datetime.strftime("%Y-%m-%d") if pre_item.datetime else pre_date_str
        actual_post_date = post_item.datetime.strftime("%Y-%m-%d") if post_item.datetime else post_date_str

        target_bands = ["B04", "B08", "SCL"]
        logger.info(f"Streaming pre-scene optical {pre_item.id}...")
        pre_ds = load_stac_dataset(
            items=[pre_item],
            bands=target_bands,
            crs=DEFAULT_CRS,
            resolution=active_res,
            bbox=bbox,
            chunks=DEFAULT_CHUNKS,
        ).compute()

        logger.info(f"Streaming post-scene optical {post_item.id}...")
        post_ds = load_stac_dataset(
            items=[post_item],
            bands=target_bands,
            crs=DEFAULT_CRS,
            resolution=active_res,
            bbox=bbox,
            chunks=DEFAULT_CHUNKS,
        ).compute()

        logger.info("Executing optical canopy loss detection...")
        detection_res = detect_canopy_loss(
            pre_ds=pre_ds,
            post_ds=post_ds,
            ndvi_threshold=active_threshold,
            min_area_sqm=min_area_sqm,
        )

    # 2. Vectorize Canopy Disturbance Rasters
    logger.info("Step 2/4: Vectorizing disturbance clusters into vector polygons...")
    raw_gdf = vectorize_change_mask(
        detection_res,
        source_crs=DEFAULT_CRS,
        target_crs="EPSG:4326",
    )
    logger.info(f"Vectorized {len(raw_gdf)} disturbance patch polygon(s).")

    # 3. Classify Legal Violation Tiers via PostGIS Spatial Intersection
    logger.info("Step 3/4: Cross-referencing Reserve Forest and riparian boundaries in PostGIS...")
    classified_gdf = classify_incident_violations(
        change_gdf=raw_gdf,
        db_session=session_override,
        boundaries_dir=boundaries_dir,
    )

    classified_gdf["sensor"] = sensor_type
    classified_gdf["pre_date"] = actual_pre_date
    classified_gdf["post_date"] = actual_post_date
    classified_gdf["aoi_name"] = aoi_name

    summary = summarize_violations(classified_gdf)
    summary["sensor"] = sensor_type
    summary["pre_date"] = actual_pre_date
    summary["post_date"] = actual_post_date
    summary["aoi_name"] = aoi_name

    # 4. Save Output GeoJSON and Summary JSON
    logger.info("Step 4/4: Exporting annotated legal violation GeoJSON and summary report...")
    geojson_path = out_dir / f"audit_{sensor_type}_{pre_date_str}_{post_date_str}.geojson"
    summary_path = out_dir / f"audit_summary_{sensor_type}_{pre_date_str}_{post_date_str}.json"

    # Convert list column to string representation for clean GeoJSON export compatibility
    export_gdf = classified_gdf.copy()
    if not export_gdf.empty and "violation_reasons" in export_gdf.columns:
        export_gdf["violation_reasons"] = export_gdf["violation_reasons"].apply(
            lambda r: "; ".join(r) if isinstance(r, (list, tuple)) else str(r)
        )

    export_gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info(f"Saved annotated legal violation GeoJSON to: {geojson_path.resolve()}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved audit summary metrics to: {summary_path.resolve()}")

    # 5. Database Persistence
    if not skip_db:
        if session_override is not None:
            # When session_override is supplied in tests
            pass
        else:
            record_audit_violations_to_db(
                classified_gdf=classified_gdf,
                sensor=sensor_type,
                pre_date=pre_date,
                post_date=post_date,
                aoi_name=aoi_name,
                bbox=bbox,
            )

    return geojson_path, classified_gdf, summary


def print_audit_report(summary: dict[str, Any], classified_gdf: gpd.GeoDataFrame) -> None:
    """Print a structured, readable console audit report table."""
    print("\n" + "=" * 76)
    print("                HADDU SATELLITE LEGAL AUDIT REPORT")
    print("=" * 76)
    print(f"  Target AOI           : {summary.get('aoi_name', 'Thirthahalli Taluk')}")
    print(f"  Sensor Platform      : {summary.get('sensor', 's1').upper()}")
    print(f"  Audit Period         : {summary.get('pre_date')} to {summary.get('post_date')}")
    print(f"  Total Incidents      : {summary['total_incidents']} cluster(s)")
    print(f"  Total Disturbance    : {summary['total_area_acres']:,.2f} acres")
    print("-" * 76)
    print("  LEGAL VIOLATION RISK TIERS:")
    print(
        f"    [CRITICAL] Reserve Forest Intrusions : {summary['critical_count']:3d} incident(s) "
        f"({summary['critical_acres']:6.2f} acres)"
    )
    print(
        f"    [HIGH]     Riparian & ESZ Breaches   : {summary['high_count']:3d} incident(s) "
        f"({summary['high_acres']:6.2f} acres)"
    )
    print(
        f"    [MODERATE] Private Agricultural Land : {summary['moderate_count']:3d} incident(s) "
        f"({summary['moderate_acres']:6.2f} acres)"
    )
    print("-" * 76)
    print(f"  Riparian Buffer (50m) Breaches : {summary['riparian_breaches']}")
    print(f"  Eco-Sensitive Zone (ESZ) Alerts: {summary['esz_breaches']}")

    if summary.get("encroached_forest_tracts"):
        print("\n  RESERVE FOREST TRACTS COMPROMISED:")
        for tract in summary["encroached_forest_tracts"]:
            print(
                f"    - {tract['name']:35s}: {tract['incidents']} incident(s), "
                f"{tract['rf_area_acres']:,.2f} acres breached"
            )

    if not classified_gdf.empty:
        print("\n  TOP DETECTED VIOLATION INCIDENTS:")
        print(f"  {'ID':<5} {'Tier':<10} {'Acres':<8} {'Breached RF / Nearest RF':<32} {'Reasons'}")
        print("  " + "-" * 72)
        top_incidents = classified_gdf.sort_values(
            by=["violation_tier", "area_acres"],
            ascending=[True, False],
        ).head(10)
        for _, row in top_incidents.iterrows():
            patch_id = str(row.get("patch_id", ""))
            tier = str(row.get("violation_tier", ""))
            acres = f"{float(row.get('area_acres', 0.0)):.2f}"
            rf_str = (
                str(row.get("encroached_rf_name"))
                if row.get("encroached_rf_name")
                else str(row.get("nearest_rf_name", "N/A"))
            )
            reasons = row.get("violation_reasons", [])
            reason_str = reasons[0] if isinstance(reasons, list) and reasons else str(reasons)
            print(f"  {patch_id:<5} {tier:<10} {acres:<8} {rf_str[:30]:<32} {reason_str}")

    print("=" * 76 + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for run_audit CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m haddu.scripts.run_audit",
        description="Run multi-sensor change detection and legal violation tier classification in PostGIS.",
    )
    parser.add_argument(
        "--sensor",
        type=str,
        required=True,
        choices=["s1", "s2", "sar", "optical"],
        help="Sensor platform to use: 's1' (SAR Radar) or 's2' (Multispectral Optical).",
    )
    parser.add_argument(
        "--pre-date",
        type=parse_date,
        required=True,
        help="Pre-clearing baseline date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--post-date",
        type=parse_date,
        required=True,
        help="Post-clearing monitoring date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output/audit"),
        help="Directory to save audit GeoJSON and reports (default: ./output/audit).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Change threshold: VH drop in dB for S1 (default: 2.5), or ΔNDVI for S2 (default: 0.35).",
    )
    parser.add_argument(
        "--min-area-sqm",
        type=float,
        default=DEFAULT_MIN_AREA_SQM,
        help=f"Minimum cluster area in m² (default: {DEFAULT_MIN_AREA_SQM}).",
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


def main() -> None:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        _geojson_path, classified_gdf, summary = run_audit_pipeline(
            sensor=args.sensor,
            pre_date=args.pre_date,
            post_date=args.post_date,
            output_dir=args.output_dir,
            threshold=args.threshold,
            min_area_sqm=args.min_area_sqm,
            max_cloud=args.max_cloud,
            skip_db=args.skip_db,
        )
        print_audit_report(summary, classified_gdf)
    except Exception:
        logger.exception("Audit pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
