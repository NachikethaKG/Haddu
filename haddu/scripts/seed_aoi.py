"""Seed script to populate Thirthahalli taluk and key reserve forest AOIs into PostGIS.

Executes database migrations/DDL, seeds the bounding box and corridors,
and queries the polygon area transformed to EPSG:32643 (UTM Zone 43N).
"""

import sys
from typing import Dict, Any, List
from shapely.geometry import Polygon
from geoalchemy2.shape import from_shape
from sqlalchemy import select, func
from sqlalchemy.exc import OperationalError

from haddu.db.session import engine, init_db, get_db, DATABASE_URL
from haddu.db.models import AOI

# Defined AOIs for Phase 1
SEED_AOIS: List[Dict[str, Any]] = [
    {
        "name": "Thirthahalli Taluk",
        "category": "Taluk",
        # Thirthahalli bounding box: [75.05, 13.55, 75.40, 13.85] (lon, lat)
        "coords": [
            (75.05, 13.55),
            (75.40, 13.55),
            (75.40, 13.85),
            (75.05, 13.85),
            (75.05, 13.55),
        ],
    },
    {
        "name": "Agumbe Foothills Corridor",
        "category": "RF",
        # Agumbe rainforest foothills corridor
        "coords": [
            (75.05, 13.55),
            (75.15, 13.55),
            (75.15, 13.63),
            (75.05, 13.63),
            (75.05, 13.55),
        ],
    },
    {
        "name": "Kundadri Reserve Forest Corridor",
        "category": "RF",
        # Kundadri hill & reserve forest corridor
        "coords": [
            (75.14, 13.55),
            (75.22, 13.55),
            (75.22, 13.63),
            (75.14, 13.63),
            (75.14, 13.55),
        ],
    },
]


def seed_areas_of_interest(session_override=None, engine_override=None) -> float:
    """Initialize schema, seed AOIs, and calculate Thirthahalli area in EPSG:32643."""
    eng = engine_override or engine
    print(f"Connecting to database at: {eng.url.render_as_string(hide_password=True)}")

    try:
        if session_override is not None:
            return _seed_and_query(session_override)

        # 1. Apply DDL / Migrations
        print("Ensuring PostGIS extension and database tables exist...")
        init_db(eng)
        print("Schema verified successfully.")

        # 2. Seed AOIs and run verification query
        with get_db() as session:
            return _seed_and_query(session)

    except OperationalError as err:
        print("\n[ERROR] Database connection failed!", file=sys.stderr)
        print(f"Detail: {err}", file=sys.stderr)
        print("\nTroubleshooting tips:", file=sys.stderr)
        print("  1. Make sure Docker is running and run: docker compose up -d", file=sys.stderr)
        print(f"  2. Or set DATABASE_URL environment variable to a reachable PostGIS database.", file=sys.stderr)
        sys.exit(1)


def _seed_and_query(session) -> float:
    """Internal helper to seed AOIs and query projected area using an active session."""
    for item in SEED_AOIS:
        existing = session.scalar(select(AOI).where(AOI.name == item["name"]))
        poly = Polygon(item["coords"])
        geom_element = from_shape(poly, srid=4326)

        if existing:
            print(f"  [Existing] '{item['name']}' ({item['category']}) already present.")
            existing.geom = geom_element
            existing.category = item["category"]
        else:
            aoi = AOI(
                name=item["name"],
                category=item["category"],
                geom=geom_element,
            )
            session.add(aoi)
            print(f"  [Inserted] '{item['name']}' ({item['category']}) registered.")

    session.flush()

    # 3. Query Thirthahalli polygon area calculated in meters via ST_Transform(geom, 32643)
    query = select(
        AOI.name,
        AOI.category,
        func.ST_Area(func.ST_Transform(AOI.geom, 32643)).label("area_sqm"),
    ).where(AOI.name == "Thirthahalli Taluk")

    result = session.execute(query).first()
    if not result or result.area_sqm is None:
        raise RuntimeError("Failed to calculate polygon area for Thirthahalli Taluk.")

    name, category, area_sqm = result.name, result.category, float(result.area_sqm)
    area_sqkm = area_sqm / 1_000_000.0

    print("\n" + "=" * 60)
    print("PostGIS Verification Query Successful:")
    print(f"  AOI Name        : {name}")
    print(f"  Category        : {category}")
    print(f"  Target CRS      : EPSG:32643 (UTM Zone 43N)")
    print(f"  Projected Area  : {area_sqm:,.2f} m² (~{area_sqkm:,.2f} km²)")
    print("=" * 60 + "\n")

    return area_sqm


if __name__ == "__main__":
    seed_areas_of_interest()
