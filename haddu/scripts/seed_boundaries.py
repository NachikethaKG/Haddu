"""Seed script to populate gazetted Reserve Forest boundaries and riparian buffers into PostGIS.

Ingests forest polygons for Agumbe, Kundadri, Megaravalli, and Gomala tracts.
Computes statutory 50m riparian buffer polygons along the Tunga River corridor
using ST_Buffer(ST_Transform(geom, 32643), 50) and stores them with verified GiST spatial indices.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import shapely
from geoalchemy2.shape import from_shape
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from haddu.core.boundaries import (
    _load_forest_features,
    _load_water_features,
    compute_riparian_buffer,
)
from haddu.db.models import ForestBoundary, WaterCourse
from haddu.db.session import engine, get_db, init_db

logger = logging.getLogger(__name__)


def verify_spatial_indices(session) -> dict[str, bool]:
    """Verify that GiST spatial indices exist on forest_boundaries and water_courses."""
    results = {}
    for model, geom_cols in [
        (ForestBoundary, ["geom"]),
        (WaterCourse, ["geom", "buffer_geom"]),
    ]:
        for col_name in geom_cols:
            idx_name = f"idx_{model.__tablename__}_{col_name}"
            # Check model definition
            model_has_index = any(
                col_name in str(idx.columns) or idx.name == idx_name
                for idx in model.__table__.indexes
            )
            results[f"{model.__tablename__}.{col_name}"] = model_has_index
    return results


def seed_forest_and_water_boundaries(
    session_override=None,
    engine_override=None,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Initialize schema, seed forest tracts and stream buffer geometries into PostGIS.

    Args:
        session_override: Optional active SQLAlchemy session for testing.
        engine_override: Optional SQLAlchemy engine.
        data_dir: Directory containing GeoJSON files (default: data/boundaries).

    Returns:
        Summary dictionary of seeded boundaries and metrics.
    """
    eng = engine_override or engine

    try:
        if session_override is not None:
            return _seed_boundaries(session_override, data_dir=data_dir)

        print(f"Connecting to database at: {eng.url.render_as_string(hide_password=True)}")
        print("Ensuring PostGIS extensions and database tables exist...")
        init_db(eng)
        print("Schema verified successfully.")

        with get_db() as session:
            return _seed_boundaries(session, data_dir=data_dir)

    except OperationalError as err:
        print("\n[ERROR] Database connection failed!", file=sys.stderr)
        print(f"Detail: {err}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print("  1. Make sure Docker is running: docker compose up -d", file=sys.stderr)
        print(
            "  2. Or set DATABASE_URL environment variable to a reachable PostGIS database.",
            file=sys.stderr,
        )
        sys.exit(1)


def _seed_boundaries(session, data_dir: Path | None = None) -> dict[str, Any]:
    """Internal helper to execute boundary insertion using an active session."""
    t_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    forest_items = _load_forest_features(data_dir=data_dir)
    water_items = _load_water_features(data_dir=data_dir)

    seeded_forests = []
    for item in forest_items:
        name = item["name"]
        category = item["category"]
        multi_poly = item["geometry"]

        # Calculate area in UTM Zone 43N
        geom_utm = shapely.transform(
            multi_poly, lambda c: np.column_stack(t_utm.transform(c[:, 0], c[:, 1]))
        )
        area_sqm = float(geom_utm.area)
        area_acres = float(area_sqm / 4046.8564)

        geom_element = from_shape(multi_poly, srid=4326)

        existing = session.scalar(select(ForestBoundary).where(ForestBoundary.name == name))
        if existing:
            existing.category = category
            existing.geom = geom_element
            existing.area_sqm = area_sqm
            existing.area_acres = area_acres
            record = existing
            print(f"  [Updated Forest] {name} ({category}) - {area_acres:.2f} acres")
        else:
            record = ForestBoundary(
                name=name,
                category=category,
                geom=geom_element,
                area_sqm=area_sqm,
                area_acres=area_acres,
            )
            session.add(record)
            print(f"  [Inserted Forest] {name} ({category}) - {area_acres:.2f} acres")

        seeded_forests.append(
            {
                "name": name,
                "category": category,
                "area_sqm": area_sqm,
                "area_acres": area_acres,
            }
        )

    seeded_water = []
    for item in water_items:
        name = item["name"]
        buf_dist = item["buffer_distance_m"]
        multi_line = item["geometry"]

        # Compute 50m riparian buffer
        buffer_poly = compute_riparian_buffer(multi_line, buffer_distance_m=buf_dist)

        # Compute buffer area in UTM
        buf_utm = shapely.transform(
            buffer_poly, lambda c: np.column_stack(t_utm.transform(c[:, 0], c[:, 1]))
        )
        buf_area_sqm = float(buf_utm.area)
        buf_area_acres = float(buf_area_sqm / 4046.8564)

        geom_element = from_shape(multi_line, srid=4326)
        buffer_element = from_shape(buffer_poly, srid=4326)

        existing_water = session.scalar(select(WaterCourse).where(WaterCourse.name == name))
        if existing_water:
            existing_water.geom = geom_element
            existing_water.buffer_geom = buffer_element
            existing_water.buffer_distance_m = buf_dist
            print(f"  [Updated Watercourse] {name} (buffer: {buf_dist}m) - {buf_area_acres:.2f} acres")
        else:
            new_water = WaterCourse(
                name=name,
                geom=geom_element,
                buffer_geom=buffer_element,
                buffer_distance_m=buf_dist,
            )
            session.add(new_water)
            print(f"  [Inserted Watercourse] {name} (buffer: {buf_dist}m) - {buf_area_acres:.2f} acres")

        seeded_water.append(
            {
                "name": name,
                "buffer_distance_m": buf_dist,
                "buffer_area_sqm": buf_area_sqm,
                "buffer_area_acres": buf_area_acres,
            }
        )

    session.flush()

    indices_status = verify_spatial_indices(session)

    summary = {
        "forest_tracts_count": len(seeded_forests),
        "water_courses_count": len(seeded_water),
        "forest_tracts": seeded_forests,
        "water_courses": seeded_water,
        "spatial_indices": indices_status,
    }
    return summary


if __name__ == "__main__":
    results = seed_forest_and_water_boundaries()
    print("\n" + "=" * 65)
    print("  FOREST BOUNDARIES & RIPARIAN BUFFER INGESTION SUMMARY")
    print("=" * 65)
    print(f"  Forest Tracts Ingested   : {results['forest_tracts_count']}")
    for tract in results["forest_tracts"]:
        print(f"    - {tract['name']:35s} [{tract['category']:12s}]: {tract['area_acres']:,.2f} acres")
    print(f"  Watercourses Ingested    : {results['water_courses_count']}")
    for stream in results["water_courses"]:
        print(
            f"    - {stream['name']:35s} [Buffer: {stream['buffer_distance_m']:.0f}m]: "
            f"{stream['buffer_area_acres']:,.2f} acres"
        )
    print("  Spatial Indices Verified :")
    for col, status in results["spatial_indices"].items():
        print(f"    - {col:35s}: {'OK (GiST)' if status else 'MISSING'}")
    print("=" * 65 + "\n")
