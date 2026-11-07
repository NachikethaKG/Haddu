"""Unit and integration tests for database models, spatial schemas, and AOI definitions."""

import uuid
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pyproj
import shapely
from geoalchemy2.shape import from_shape
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from haddu.db.models import (
    AOI,
    EncroachmentIncident,
    ForestBoundary,
    SatellitePass,
    WaterCourse,
)
from haddu.scripts.seed_aoi import SEED_AOIS, seed_areas_of_interest
from haddu.scripts.seed_boundaries import (
    _load_forest_features,
    _load_water_features,
    compute_riparian_buffer,
    seed_forest_and_water_boundaries,
    verify_spatial_indices,
)


def test_model_tables_and_columns():
    """Verify table names and essential column configurations on models."""
    assert AOI.__tablename__ == "aoi"
    assert SatellitePass.__tablename__ == "satellite_pass"
    assert EncroachmentIncident.__tablename__ == "encroachment_incidents"
    assert ForestBoundary.__tablename__ == "forest_boundaries"
    assert WaterCourse.__tablename__ == "water_courses"

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

    incident_cols = {col.name for col in EncroachmentIncident.__table__.columns}
    assert {
        "id",
        "aoi_id",
        "pre_date",
        "post_date",
        "area_sqm",
        "area_acres",
        "centroid_lat",
        "centroid_lon",
        "geom",
        "violation_tier",
        "encroached_rf_name",
        "rf_area_acres",
        "riparian_breach",
        "sensor",
        "created_at",
    }.issubset(incident_cols)

    forest_cols = {col.name for col in ForestBoundary.__table__.columns}
    assert {
        "id",
        "name",
        "category",
        "geom",
        "area_sqm",
        "area_acres",
        "created_at",
    }.issubset(forest_cols)

    water_cols = {col.name for col in WaterCourse.__table__.columns}
    assert {
        "id",
        "name",
        "geom",
        "buffer_geom",
        "buffer_distance_m",
        "created_at",
    }.issubset(water_cols)


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

    incident = EncroachmentIncident(
        aoi_id=uuid.uuid4(),
        pre_date=date(2025, 12, 15),
        post_date=date(2026, 3, 15),
        area_sqm=5000.0,
        area_acres=1.235,
        centroid_lat=13.70,
        centroid_lon=75.25,
        mean_delta_ndvi=0.45,
        max_delta_ndvi=0.65,
        violation_tier="CRITICAL",
        encroached_rf_name="Agumbe Rainforest Reserve Forest",
        rf_area_acres=1.235,
        riparian_breach=False,
        sensor="s1",
        geom=geom,
    )
    assert incident.area_sqm == 5000.0
    assert incident.area_acres == 1.235
    assert incident.violation_tier == "CRITICAL"
    assert incident.encroached_rf_name == "Agumbe Rainforest Reserve Forest"
    assert "1.24 acres" in repr(incident)
    assert "tier=CRITICAL" in repr(incident)

    forest = ForestBoundary(
        name="Agumbe Rainforest Reserve Forest",
        category="RF",
        geom=from_shape(MultiPolygon([Polygon(sample_coords)]), srid=4326),
        area_sqm=50000.0,
        area_acres=12.35,
    )
    assert forest.name == "Agumbe Rainforest Reserve Forest"
    assert forest.category == "RF"
    assert "Agumbe Rainforest Reserve Forest" in repr(forest)

    water = WaterCourse(
        name="Tunga River Corridor",
        geom=from_shape(
            MultiLineString([LineString([(75.16, 13.61), (75.21, 13.64)])]),
            srid=4326,
        ),
        buffer_geom=from_shape(MultiPolygon([Polygon(sample_coords)]), srid=4326),
        buffer_distance_m=50.0,
    )
    assert water.name == "Tunga River Corridor"
    assert water.buffer_distance_m == 50.0
    assert "Tunga River Corridor" in repr(water)


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


def test_compute_riparian_buffer():
    """Verify compute_riparian_buffer creates a valid 50m buffer in UTM Zone 43N reprojected to EPSG:4326."""
    line = LineString([(75.16, 13.61), (75.21, 13.64), (75.25, 13.67)])
    buf = compute_riparian_buffer(line, buffer_distance_m=50.0)

    assert isinstance(buf, MultiPolygon)
    assert buf.is_valid
    assert not buf.is_empty

    # Transform back to UTM 32643 to verify buffer width is ~50m
    t_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    line_utm = shapely.transform(line, lambda c: np.column_stack(t_utm.transform(c[:, 0], c[:, 1])))
    buf_utm = shapely.transform(buf, lambda c: np.column_stack(t_utm.transform(c[:, 0], c[:, 1])))

    # Distance from points on buffer exterior to center line should be approx 50m
    exterior = buf_utm.geoms[0].exterior
    sample_pt = exterior.interpolate(100)
    dist = line_utm.distance(sample_pt)
    assert abs(dist - 50.0) < 1.0, f"Expected ~50m buffer width, got {dist:.2f}m"


def test_load_boundary_features_from_data():
    """Verify loading boundary features from GeoJSON in data/boundaries."""
    forests = _load_forest_features()
    assert len(forests) >= 4
    forest_names = {f["name"] for f in forests}
    assert any("Agumbe" in name for name in forest_names)
    assert any("Kundadri" in name for name in forest_names)
    assert any("Megaravalli" in name for name in forest_names)
    assert any("Gomala" in name for name in forest_names)

    for item in forests:
        assert isinstance(item["geometry"], MultiPolygon)
        assert item["geometry"].is_valid
        assert item["category"] in ("RF", "NationalPark", "Gomala")

    watercourses = _load_water_features()
    assert len(watercourses) >= 1
    tunga = next(w for w in watercourses if "Tunga" in w["name"])
    assert isinstance(tunga["geometry"], MultiLineString)
    assert tunga["geometry"].is_valid
    assert tunga["buffer_distance_m"] == 50.0


def test_spatial_indices_verification():
    """Verify verify_spatial_indices checks spatial indices on forest and water tables."""
    mock_session = MagicMock()
    indices = verify_spatial_indices(mock_session)
    assert "forest_boundaries.geom" in indices
    assert "water_courses.geom" in indices
    assert "water_courses.buffer_geom" in indices
    assert all(indices.values()), f"Expected all spatial indices to be True, got {indices}"


def test_seed_forest_and_water_boundaries_flow():
    """Verify seed_forest_and_water_boundaries execution flow with mock session."""
    mock_session = MagicMock()
    mock_session.scalar.return_value = None  # Simulates first-time insert

    summary = seed_forest_and_water_boundaries(session_override=mock_session)

    assert summary["forest_tracts_count"] >= 4
    assert summary["water_courses_count"] >= 1
    assert mock_session.add.call_count >= 5
    assert mock_session.flush.called
    assert all(summary["spatial_indices"].values())
