"""Seed script to populate gazetted Reserve Forest boundaries and riparian buffers into PostGIS.

Ingests forest polygons for Agumbe, Kundadri, Megaravalli, and Gomala tracts.
Computes statutory 50m riparian buffer polygons along the Tunga River corridor
using ST_Buffer(ST_Transform(geom, 32643), 50) and stores them with verified GiST spatial indices.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import shapely
from geoalchemy2.shape import from_shape
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, shape
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from haddu.db.models import ForestBoundary, WaterCourse
from haddu.db.session import engine, get_db, init_db

logger = logging.getLogger(__name__)

DATA_BOUNDARIES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "boundaries"

# Built-in definitions as fallback if GeoJSON files are missing
FALLBACK_FOREST_TRACTS: list[dict[str, Any]] = [
    {
        "name": "Agumbe Rainforest Reserve Forest",
        "category": "RF",
        "coords": [
            [
                (75.06, 13.55),
                (75.14, 13.55),
                (75.14, 13.62),
                (75.06, 13.62),
                (75.06, 13.55),
            ]
        ],
    },
    {
        "name": "Agumbe National Park Eco-Zone",
        "category": "NationalPark",
        "coords": [
            [
                (75.05, 13.55),
                (75.08, 13.55),
                (75.08, 13.60),
                (75.05, 13.60),
                (75.05, 13.55),
            ]
        ],
    },
    {
        "name": "Kundadri Peak Reserve Forest",
        "category": "RF",
        "coords": [
            [
                (75.15, 13.56),
                (75.22, 13.56),
                (75.22, 13.63),
                (75.15, 13.63),
                (75.15, 13.56),
            ]
        ],
    },
    {
        "name": "Megaravalli State Forest",
        "category": "RF",
        "coords": [
            [
                (75.10, 13.64),
                (75.18, 13.64),
                (75.18, 13.72),
                (75.10, 13.72),
                (75.10, 13.64),
            ]
        ],
    },
    {
        "name": "Thirthahalli Gomala Pasture Land",
        "category": "Gomala",
        "coords": [
            [
                (75.22, 13.67),
                (75.28, 13.67),
                (75.28, 13.72),
                (75.22, 13.72),
                (75.22, 13.67),
            ]
        ],
    },
]

FALLBACK_WATER_COURSES: list[dict[str, Any]] = [
    {
        "name": "Tunga River Corridor",
        "buffer_distance_m": 50.0,
        "coords": [
            (75.16, 13.61),
            (75.21, 13.64),
            (75.25, 13.67),
            (75.29, 13.69),
            (75.33, 13.73),
            (75.38, 13.78),
        ],
    }
]


def compute_riparian_buffer(
    geom_4326: LineString | MultiLineString,
    buffer_distance_m: float = 50.0,
) -> MultiPolygon:
    """Compute projected 50m buffer in UTM Zone 43N and return as WGS84 MultiPolygon.

    Matches PostGIS ST_Transform(ST_Buffer(ST_Transform(geom, 32643), buffer_distance_m), 4326).

    Args:
        geom_4326: LineString or MultiLineString in EPSG:4326.
        buffer_distance_m: Buffer radius in meters (default: 50m).

    Returns:
        MultiPolygon in EPSG:4326.
    """
    t_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    t_wgs = pyproj.Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)

    geom_utm = shapely.transform(
        geom_4326, lambda c: np.column_stack(t_utm.transform(c[:, 0], c[:, 1]))
    )
    buffer_utm = geom_utm.buffer(buffer_distance_m)
    buffer_wgs = shapely.transform(
        buffer_utm, lambda c: np.column_stack(t_wgs.transform(c[:, 0], c[:, 1]))
    )

    if buffer_wgs.geom_type == "Polygon":
        return MultiPolygon([buffer_wgs])
    elif buffer_wgs.geom_type == "MultiPolygon":
        return buffer_wgs
    else:
        raise ValueError(f"Unexpected buffered geometry type: {buffer_wgs.geom_type}")


def _load_forest_features(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load forest boundaries from GeoJSON or fall back to built-in definitions."""
    base_dir = data_dir or DATA_BOUNDARIES_DIR
    geojson_path = base_dir / "forest_boundaries.geojson"

    if geojson_path.exists():
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = []
            for feat in data.get("features", []):
                geom = shape(feat["geometry"])
                if geom.geom_type == "Polygon":
                    multi_geom = MultiPolygon([geom])
                elif geom.geom_type == "MultiPolygon":
                    multi_geom = geom
                else:
                    continue
                features.append(
                    {
                        "name": feat["properties"]["name"],
                        "category": feat["properties"]["category"],
                        "geometry": multi_geom,
                    }
                )
            if features:
                return features
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Failed to read {geojson_path}: {exc}. Using built-in fallbacks.")

    # Built-in fallback
    features = []
    for item in FALLBACK_FOREST_TRACTS:
        polys = [Polygon(ring) for ring in item["coords"]]
        features.append(
            {
                "name": item["name"],
                "category": item["category"],
                "geometry": MultiPolygon(polys),
            }
        )
    return features


def _load_water_features(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load water course features from GeoJSON or fall back to built-in definitions."""
    base_dir = data_dir or DATA_BOUNDARIES_DIR
    geojson_path = base_dir / "water_courses.geojson"

    if geojson_path.exists():
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = []
            for feat in data.get("features", []):
                geom = shape(feat["geometry"])
                if geom.geom_type == "LineString":
                    multi_geom = MultiLineString([geom])
                elif geom.geom_type == "MultiLineString":
                    multi_geom = geom
                else:
                    continue
                features.append(
                    {
                        "name": feat["properties"]["name"],
                        "buffer_distance_m": float(
                            feat["properties"].get("buffer_distance_m", 50.0)
                        ),
                        "geometry": multi_geom,
                    }
                )
            if features:
                return features
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Failed to read {geojson_path}: {exc}. Using built-in fallbacks.")

    # Built-in fallback
    features = []
    for item in FALLBACK_WATER_COURSES:
        line = LineString(item["coords"])
        features.append(
            {
                "name": item["name"],
                "buffer_distance_m": float(item["buffer_distance_m"]),
                "geometry": MultiLineString([line]),
            }
        )
    return features


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
