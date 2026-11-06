"""Time-Series Seasonal Monitoring and Persistent Breakpoint CLI.

Executes the full Haddu Phase 3 monitoring engine over a specified target year:
1. Ingests cloud-masked Sentinel-2 cubes over multi-year baseline + target windows.
2. Fits Fourier OLS harmonic regression over the historical baseline to model phenology.
3. Detects persistent negative residuals verified by spectral bare-soil emergence (Delta NDBI > 0).
4. Vectorizes confirmed persistent canopy loss into GeoJSON polygons with break dates and z-scores.
5. Generates time-series verification profile plots (output/pixel_profile_<lat>_<lon>.png).
6. Persists verified encroachment incidents into PostGIS / SQLite.
"""

import argparse
from datetime import date, datetime
import logging
from pathlib import Path
import sys
from typing import List, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from geoalchemy2.shape import from_shape

from haddu.core.stac_client import DEFAULT_AOI_BBOX
from haddu.core.loader import DEFAULT_CRS, DEFAULT_RESOLUTION
from haddu.core.timeseries import build_timeseries_cube
from haddu.core.harmonic import fit_harmonic_baseline, HarmonicModel
from haddu.core.persistence import (
    DEFAULT_CONSECUTIVE_PASSES,
    DEFAULT_Z_THRESHOLD,
    DEFAULT_MIN_AREA_SQM,
    PersistentBreakResult,
    detect_persistent_breaks,
    plot_pixel_profile,
)
from haddu.core.vectorizer import vectorize_persistent_breaks
from haddu.db.session import engine, init_db, get_db
from haddu.db.models import AOI, EncroachmentIncident

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("haddu.monitor_timeseries")


def record_persistent_incidents_to_db(
    gdf: gpd.GeoDataFrame,
    baseline_start: date,
    target_end: date,
    aoi_name: str,
    bbox: Sequence[float],
) -> int:
    """Commit verified persistent encroachment incidents to PostGIS database."""
    if gdf.empty:
        logger.info("No persistent incidents to record in database.")
        return 0

    try:
        logger.info("Connecting to PostGIS database to record persistent incidents...")
        init_db(engine)

        with get_db() as session:
            # Check or create AOI
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
                b_date = None
                if row.get("break_date") and not pd.isna(row["break_date"]):
                    try:
                        b_date = pd.Timestamp(row["break_date"]).date()
                    except Exception:
                        b_date = None

                conf_val = None
                if row.get("confidence_zscore") is not None and not pd.isna(row["confidence_zscore"]):
                    conf_val = float(row["confidence_zscore"])

                incident = EncroachmentIncident(
                    aoi_id=aoi.id if aoi else None,
                    pre_date=baseline_start,
                    post_date=target_end,
                    break_date=b_date,
                    confidence_score=conf_val,
                    area_sqm=float(row["area_sqm"]),
                    area_acres=float(row["area_acres"]),
                    centroid_lat=float(row["centroid_lat"]),
                    centroid_lon=float(row["centroid_lon"]),
                    geom=from_shape(poly_geom, srid=4326),
                    properties_json={
                        "patch_id": int(row["patch_id"]),
                        "break_date": str(row["break_date"]),
                        "confidence_zscore": conf_val,
                        "monitoring_engine": "fourier_ols_persistence",
                    },
                )
                session.add(incident)
                inserted_count += 1

            session.commit()
            logger.info(f"Successfully committed {inserted_count} persistent incident records to PostGIS.")
            return inserted_count

    except OperationalError as exc:
        logger.warning(
            f"PostGIS database unreachable ({exc.orig}). "
            "Proceeding without database persistence. "
            "To persist to PostGIS, run: 'docker compose up -d db'"
        )
        return 0


def run_monitoring_pipeline(
    target_year: int = 2026,
    baseline_years: int = 2,
    consecutive_passes: int = DEFAULT_CONSECUTIVE_PASSES,
    output_dir: str = "./output",
    bbox: Sequence[float] = DEFAULT_AOI_BBOX,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    min_area_sqm: float = DEFAULT_MIN_AREA_SQM,
    resolution: float = DEFAULT_RESOLUTION,
    max_cloud: float = 25.0,
    preferred_tile: Optional[str] = "T43PER",
    plot_top_n: int = 3,
    persist_db: bool = False,
    aoi_name: str = "Thirthahalli Taluk",
    max_passes: Optional[int] = None,
) -> PersistentBreakResult:
    """Run full time-series seasonal monitoring, persistence detection, and export.

    Args:
        target_year: Evaluation year to monitor for new persistent clearings.
        baseline_years: Number of historical years to fit Fourier seasonal baseline.
        consecutive_passes: Consecutive anomalous passes required to confirm break.
        output_dir: Directory where GeoJSON and plot artifacts are stored.
        bbox: Bounding box coordinates in EPSG:4326.
        z_threshold: Z-score threshold for negative residual anomaly.
        min_area_sqm: Minimum contiguous patch area in square meters.
        resolution: Spatial resolution in meters (default: 10m).
        max_cloud: Maximum cloud cover percentage allowed per pass.
        preferred_tile: MGRS tile filter (default: 'T43PER').
        plot_top_n: Number of top loss patches to generate verification plots for.
        persist_db: Whether to record detected incidents to PostGIS.
        aoi_name: Name of Area of Interest.
        max_passes: Optional limit on total passes to stream.

    Returns:
        PersistentBreakResult containing detection arrays and patch metrics.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Establish temporal boundaries
    start_baseline_str = f"{target_year - baseline_years}-01-01"
    end_baseline_str = f"{target_year - 1}-12-31"
    start_target_str = f"{target_year}-01-01"
    end_target_str = f"{target_year}-12-31"

    logger.info("=" * 70)
    logger.info("HADDU TIME-SERIES CANOPY MONITORING ENGINE")
    logger.info("=" * 70)
    logger.info(f"Target Year          : {target_year} ({start_target_str} to {end_target_str})")
    logger.info(f"Baseline Window      : {baseline_years} year(s) ({start_baseline_str} to {end_baseline_str})")
    logger.info(f"Consecutive Passes   : {consecutive_passes}")
    logger.info(f"Z-Score Threshold    : {z_threshold}σ drop below seasonal baseline")
    logger.info(f"Min Patch Area       : {min_area_sqm} m²")
    logger.info(f"Spatial Resolution   : {resolution} meters")
    logger.info(f"AOI Bounding Box     : {list(bbox)}")
    logger.info(f"Output Directory     : {out_path.resolve()}")
    logger.info("=" * 70)

    # 2. Build multi-temporal Sentinel-2 cube across entire window
    logger.info("Step 1/5: Ingesting multi-temporal Sentinel-2 cube from STAC...")
    full_cube = build_timeseries_cube(
        bbox=bbox,
        start_date=start_baseline_str,
        end_date=end_target_str,
        max_cloud=max_cloud,
        resolution=resolution,
        preferred_tile=preferred_tile,
        max_passes=max_passes,
        compute=True,
    )

    total_passes = len(full_cube.time)
    logger.info(f"Ingested {total_passes} cloud-masked pass(es) spanning {start_baseline_str} to {end_target_str}.")

    # 3. Fit Harmonic Baseline Decomposition
    logger.info("Step 2/5: Fitting Fourier OLS Harmonic Seasonal Baseline...")
    # Select baseline passes (or use full cube if baseline period has few passes)
    baseline_slice = full_cube.sel(time=slice(start_baseline_str, end_baseline_str))
    if len(baseline_slice.time) >= 4:
        fit_cube = baseline_slice
        logger.info(f"Fitting harmonic model over {len(baseline_slice.time)} baseline pass(es).")
    else:
        fit_cube = full_cube
        logger.warning(
            f"Only {len(baseline_slice.time)} pass(es) in baseline slice. "
            f"Fitting harmonic model over all {len(full_cube.time)} pass(es)."
        )

    harmonic_model = fit_harmonic_baseline(fit_cube, min_valid_obs=4)
    mean_sigma = float(harmonic_model.sigma.mean())
    logger.info(f"Fitted harmonic baseline: mean residual RMSE σ = {mean_sigma:.4f}")

    # 4. Detect Persistent Breakpoints & Bare Soil Transitions
    logger.info("Step 3/5: Detecting persistent breakpoints and bare-soil emergence...")
    # Evaluate over target year passes if available, or full cube
    target_slice = full_cube.sel(time=slice(start_target_str, end_target_str))
    eval_cube = target_slice if len(target_slice.time) >= consecutive_passes else full_cube

    result = detect_persistent_breaks(
        timeseries_cube=eval_cube,
        harmonic_model=harmonic_model,
        consecutive_passes=consecutive_passes,
        z_threshold=z_threshold,
        min_area_sqm=min_area_sqm,
        confirm_ndbi=True,
    )

    logger.info(
        f"Detected {result.num_patches} persistent clearing cluster(s) totaling "
        f"{result.total_area_acres:.2f} acres ({result.total_area_sqm:.1f} m²)."
    )

    # 5. Vectorize into GeoJSON polygons
    logger.info("Step 4/5: Vectorizing confirmed breakpoint masks into GeoJSON...")
    gdf = vectorize_persistent_breaks(result, source_crs=DEFAULT_CRS, target_crs="EPSG:4326")

    geojson_file = out_path / f"persistent_change_{target_year}.geojson"
    gdf.to_file(geojson_file, driver="GeoJSON")
    logger.info(f"Saved persistent encroachment GeoJSON to: {geojson_file.resolve()}")

    # 6. Generate time-series verification profile plots
    logger.info("Step 5/5: Generating time-series trajectory verification plots...")
    plots_generated: List[Path] = []
    patches_to_plot = result.patch_metrics[:plot_top_n]

    for p in patches_to_plot:
        p_id = p["patch_id"]
        cent_y = p["centroid_y"]
        cent_x = p["centroid_x"]
        cent_lat = p.get("centroid_lat")
        cent_lon = p.get("centroid_lon")
        b_date = p.get("earliest_break_date")

        if cent_lat is not None and cent_lon is not None:
            plot_filename = f"pixel_profile_{cent_lat:.5f}_{cent_lon:.5f}.png"
        else:
            plot_filename = f"pixel_profile_patch{p_id}_{cent_y:.0f}_{cent_x:.0f}.png"

        plot_dest = out_path / plot_filename
        plot_pixel_profile(
            cube=full_cube,
            harmonic_model=harmonic_model,
            y=cent_y,
            x=cent_x,
            break_date=b_date,
            lat=cent_lat,
            lon=cent_lon,
            z_threshold=z_threshold,
            output_path=plot_dest,
        )
        plots_generated.append(plot_dest)

    # 7. Database Persistence
    if persist_db:
        logger.info("Recording persistent incidents in database...")
        b_start_date = datetime.strptime(start_baseline_str, "%Y-%m-%d").date()
        t_end_date = datetime.strptime(end_target_str, "%Y-%m-%d").date()
        record_persistent_incidents_to_db(
            gdf=gdf,
            baseline_start=b_start_date,
            target_end=t_end_date,
            aoi_name=aoi_name,
            bbox=bbox,
        )

    # 8. Print terminal summary
    print("\n" + "=" * 65)
    print("HADDU PERSISTENT MONITORING SUMMARY REPORT")
    print("=" * 65)
    print(f"Target Year               : {target_year}")
    print(f"Baseline Window           : {baseline_years} year(s) ({start_baseline_str} -> {end_baseline_str})")
    print(f"Total Passes Analyzed     : {total_passes}")
    print(f"Verified Clearing Patches : {result.num_patches}")
    print(f"Total Loss Area (Acres)   : {result.total_area_acres:.2f} acres")
    print(f"Total Loss Area (Sq. m)   : {result.total_area_sqm:,.1f} m²")
    if result.earliest_break_date:
        print(f"Earliest Break Date       : {result.earliest_break_date.strftime('%Y-%m-%d')}")
    if result.latest_break_date:
        print(f"Latest Break Date         : {result.latest_break_date.strftime('%Y-%m-%d')}")
    print(f"Vector GeoJSON Export     : {geojson_file.resolve()}")
    for plot_p in plots_generated:
        print(f"Verification Plot         : {plot_p.resolve()}")
    print("=" * 65 + "\n")

    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Haddu Time-Series Seasonal Monitoring & Persistent Breakpoint CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2026,
        help="Evaluation year to monitor for persistent canopy loss.",
    )
    parser.add_argument(
        "--baseline-years",
        type=int,
        default=2,
        help="Number of historical years prior to target-year to fit seasonal baseline.",
    )
    parser.add_argument(
        "--consecutive-passes",
        type=int,
        default=DEFAULT_CONSECUTIVE_PASSES,
        help="Minimum consecutive anomalous passes required to confirm persistence.",
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=DEFAULT_Z_THRESHOLD,
        help="Z-score threshold for negative NDVI residual anomaly.",
    )
    parser.add_argument(
        "--min-area-sqm",
        type=float,
        default=DEFAULT_MIN_AREA_SQM,
        help="Minimum contiguous clearing patch area in square meters.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save vector GeoJSON and verification plots.",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=DEFAULT_AOI_BBOX,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box coordinates in EPSG:4326.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help="Pixel resolution in meters.",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=25.0,
        help="Maximum cloud cover percentage allowed per pass.",
    )
    parser.add_argument(
        "--preferred-tile",
        type=str,
        default="T43PER",
        help="Preferred MGRS tile filter.",
    )
    parser.add_argument(
        "--plot-top-n",
        type=int,
        default=3,
        help="Number of top loss patches to render verification plots for.",
    )
    parser.add_argument(
        "--persist-db",
        action="store_true",
        default=False,
        help="Record verified incidents in PostGIS database.",
    )
    parser.add_argument(
        "--aoi-name",
        type=str,
        default="Thirthahalli Taluk",
        help="Name of Area of Interest in database.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=None,
        help="Cap total number of satellite passes to ingest.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        run_monitoring_pipeline(
            target_year=args.target_year,
            baseline_years=args.baseline_years,
            consecutive_passes=args.consecutive_passes,
            output_dir=args.output_dir,
            bbox=args.bbox,
            z_threshold=args.z_threshold,
            min_area_sqm=args.min_area_sqm,
            resolution=args.resolution,
            max_cloud=args.max_cloud,
            preferred_tile=args.preferred_tile,
            plot_top_n=args.plot_top_n,
            persist_db=args.persist_db,
            aoi_name=args.aoi_name,
            max_passes=args.max_passes,
        )
        return 0
    except Exception as exc:
        logger.error(f"Monitoring pipeline failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
