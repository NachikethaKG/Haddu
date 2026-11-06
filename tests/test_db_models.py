"""Unit and integration tests for database models, spatial schemas, and AOI definitions."""

import uuid
from datetime import date
from unittest.mock import MagicMock
from shapely.geometry import Polygon
import pyproj
from sqlalchemy import select, func
from geoalchemy2.shape import from_shape

from haddu.db.models import Base, AOI, SatellitePass
from haddu.scripts.seed_aoi import SEED_AOIS, seed_areas_of_interest


def test_model_tables_and_columns():
    """Verify table names and essential column configurations on models."""
    assert AOI.__tablename__ == "aoi"
    assert SatellitePass.__tablename__ == "satellite_pass"

    aoi_cols = {col.name for col in AOI.__table__.columns}
    assert {"id", "name", "category", "geom", "created_at"}.issubset(aoi_cols)

    pass_cols = {col.name for col in SatellitePass.__table__.columns}
    assert {
        "id",
        "aoi_id",
        "platform",
        "pass_date",
        "cloud_cover_pct",
        "stac_item_id",
        "metadata_json",
        "created_at",
    }.issubset(pass_cols)


def test_seed_aoi_geometries_are_valid():
    """Verify all predefined seed AOIs form valid closed polygons within bounds."""
    # Thirthahalli bounding box limits
    min_lon, min_lat, max_lon, max_lat = 75.05, 13.55, 75.40, 13.85

    for item in SEED_AOIS:
        coords = item["coords"]
        assert len(coords) >= 4, f"AOI {item['name']} must have at least 4 coordinates"
        assert coords[0] == coords[-1], f"AOI {item['name']} polygon must be closed"

        poly = Polygon(coords)
        assert poly.is_valid, f"Polygon for {item['name']} must be geometrically valid"
        assert poly.area > 0, f"Polygon for {item['name']} must have non-zero area"

        # Check that coordinates lie within the taluk bounds (or equal)
        for lon, lat in coords:
            assert min_lon <= lon <= max_lon, f"{lon} outside lon bounds for {item['name']}"
            assert min_lat <= lat <= max_lat, f"{lat} outside lat bounds for {item['name']}"


def test_thirthahalli_projected_area_calculation():
    """Verify UTM Zone 43N (EPSG:32643) area calculation for Thirthahalli bounding box."""
    thirthahalli = next(a for a in SEED_AOIS if a["name"] == "Thirthahalli Taluk")
    poly = Polygon(thirthahalli["coords"])

    # Transform to UTM Zone 43N
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    coords_proj = [transformer.transform(x, y) for x, y in poly.exterior.coords]
    proj_poly = Polygon(coords_proj)

    # Thirthahalli bbox is ~38km x 33km ≈ 1.25 billion m² (~1255 km²)
    area_sqm = proj_poly.area
    area_sqkm = area_sqm / 1e6

    assert 1200 <= area_sqkm <= 1300, f"Expected ~1255 km², got {area_sqkm:.2f} km²"


def test_model_instantiation():
    """Verify ORM model objects can be instantiated with valid geometries."""
    sample_coords = [(75.05, 13.55), (75.40, 13.55), (75.40, 13.85), (75.05, 13.85), (75.05, 13.55)]
    geom = from_shape(Polygon(sample_coords), srid=4326)

    aoi = AOI(
        name="Test Taluk",
        category="Taluk",
        geom=geom,
    )
    assert aoi.name == "Test Taluk"
    assert aoi.category == "Taluk"
    assert "Test Taluk" in repr(aoi)

    sat_pass = SatellitePass(
        aoi_id=uuid.uuid4(),
        platform="Sentinel-2A",
        pass_date=date(2026, 1, 15),
        cloud_cover_pct=12.5,
        stac_item_id="S2A_MSIL2A_20260115_TEST",
        metadata_json={"epsg": 32643, "clouds": 12.5},
    )
    assert sat_pass.platform == "Sentinel-2A"
    assert sat_pass.cloud_cover_pct == 12.5
    assert "Sentinel-2A" in repr(sat_pass)


def test_seed_aoi_execution_flow():
    """Verify seed_areas_of_interest execution flow and query processing."""
    mock_session = MagicMock()
    mock_session.scalar.return_value = None  # Simulates first-time insert

    mock_row = MagicMock()
    mock_row.name = "Thirthahalli Taluk"
    mock_row.category = "Taluk"
    mock_row.area_sqm = 1255640731.29

    mock_session.execute.return_value.first.return_value = mock_row

    area = seed_areas_of_interest(session_override=mock_session)
    assert area == 1255640731.29
    assert mock_session.add.call_count == len(SEED_AOIS)
    assert mock_session.flush.called
    assert mock_session.execute.called
