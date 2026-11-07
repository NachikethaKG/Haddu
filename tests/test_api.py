"""Unit and integration tests for FastAPI backend, routes, and TiTiler tile service."""

import uuid
from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from geoalchemy2.shape import from_shape
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from haddu.api.main import app
from haddu.db.models import AOI, EncroachmentIncident, ForestBoundary, WaterCourse
from haddu.db.session import get_db_session


@pytest.fixture
def mock_session():
    """Create a mock SQLAlchemy session for dependency injection."""
    session = MagicMock()
    return session


@pytest.fixture
def client(mock_session):
    """Create a TestClient with overridden get_db dependency."""
    app.dependency_overrides[get_db_session] = lambda: mock_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_incidents():
    """Create sample incident model objects for testing."""
    sample_poly = Polygon([(75.08, 13.57), (75.10, 13.57), (75.10, 13.59), (75.08, 13.59), (75.08, 13.57)])
    geom = from_shape(sample_poly, srid=4326)

    inc1 = EncroachmentIncident(
        id=uuid.uuid4(),
        pre_date=date(2026, 6, 28),
        post_date=date(2026, 8, 15),
        area_sqm=40468.56,
        area_acres=10.0,
        centroid_lat=13.58,
        centroid_lon=75.09,
        mean_delta_ndvi=0.45,
        max_delta_ndvi=0.65,
        violation_tier="CRITICAL",
        encroached_rf_name="Agumbe Rainforest Reserve Forest",
        rf_area_acres=10.0,
        riparian_breach=False,
        sensor="s1",
        geom=geom,
        created_at=datetime.now(UTC),
    )

    inc2 = EncroachmentIncident(
        id=uuid.uuid4(),
        pre_date=date(2026, 6, 28),
        post_date=date(2026, 8, 15),
        area_sqm=12140.57,
        area_acres=3.0,
        centroid_lat=13.68,
        centroid_lon=75.25,
        mean_delta_ndvi=0.35,
        max_delta_ndvi=0.48,
        violation_tier="HIGH",
        encroached_rf_name=None,
        rf_area_acres=0.0,
        riparian_breach=True,
        sensor="s1",
        geom=geom,
        created_at=datetime.now(UTC),
    )

    inc3 = EncroachmentIncident(
        id=uuid.uuid4(),
        pre_date=date(2026, 1, 15),
        post_date=date(2026, 3, 15),
        area_sqm=4046.86,
        area_acres=1.0,
        centroid_lat=13.70,
        centroid_lon=75.24,
        mean_delta_ndvi=0.38,
        max_delta_ndvi=0.50,
        violation_tier="MODERATE",
        encroached_rf_name=None,
        rf_area_acres=0.0,
        riparian_breach=False,
        sensor="s2",
        geom=geom,
        created_at=datetime.now(UTC),
    )

    return [inc1, inc2, inc3]


def test_root_and_health_endpoints(client):
    """Verify root information and health check endpoints respond with 200."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    data_root = res_root.json()
    assert "Haddu Satellite Encroachment API" in data_root["service"]
    assert data_root["status"] == "healthy"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}


def test_docs_and_openapi(client):
    """Verify interactive Swagger UI docs and OpenAPI JSON schemas are accessible."""
    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    assert "swagger" in res_docs.text.lower() or "html" in res_docs.headers["content-type"]

    res_openapi = client.get("/openapi.json")
    assert res_openapi.status_code == 200
    schema = res_openapi.json()
    assert "/api/v1/incidents" in schema["paths"]
    assert "/api/v1/incidents/geojson" in schema["paths"]
    assert "/api/v1/aoi/bounds" in schema["paths"]


def test_titiler_tile_endpoints_mounted(client):
    """Verify TiTiler dynamic raster tile endpoints are registered under /api/v1/tiles."""
    res_openapi = client.get("/openapi.json")
    schema = res_openapi.json()

    # Verify tile routes exist in openapi
    tile_paths = [p for p in schema["paths"] if p.startswith("/api/v1/tiles")]
    assert len(tile_paths) > 0, "Expected TiTiler endpoints mounted under /api/v1/tiles"

    # Calling tilejson without required url query parameter should return 422 validation error
    res_tilejson = client.get("/api/v1/tiles/WebMercatorQuad/tilejson.json")
    assert res_tilejson.status_code == 422, "Expected 422 missing required 'url' parameter"


def test_list_incidents(client, mock_session, sample_incidents):
    """Verify GET /api/v1/incidents returns paginated incidents."""
    mock_session.scalar.return_value = len(sample_incidents)
    mock_session.scalars.return_value.all.return_value = sample_incidents

    res = client.get("/api/v1/incidents?limit=10&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert len(data["incidents"]) == 3
    assert data["incidents"][0]["violation_tier"] == "CRITICAL"
    assert data["incidents"][0]["area_acres"] == 10.0


def test_list_incidents_with_filters(client, mock_session, sample_incidents):
    """Verify query filters (tier, acres, date, sensor) are handled."""
    filtered_incidents = [sample_incidents[0]]  # Only CRITICAL
    mock_session.scalar.return_value = 1
    mock_session.scalars.return_value.all.return_value = filtered_incidents

    res = client.get("/api/v1/incidents?violation_tier=CRITICAL&min_acres=5.0&sensor=s1")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    assert data["incidents"][0]["violation_tier"] == "CRITICAL"


def test_get_incident_by_id_success(client, mock_session, sample_incidents):
    """Verify GET /api/v1/incidents/{id} returns matching incident."""
    target_inc = sample_incidents[0]
    mock_session.scalar.return_value = target_inc

    res = client.get(f"/api/v1/incidents/{target_inc.id}")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == str(target_inc.id)
    assert data["violation_tier"] == "CRITICAL"
    assert data["encroached_rf_name"] == "Agumbe Rainforest Reserve Forest"


def test_get_incident_by_id_not_found(client, mock_session):
    """Verify GET /api/v1/incidents/{id} returns 404 for unknown UUID."""
    mock_session.scalar.return_value = None
    random_id = uuid.uuid4()

    res = client.get(f"/api/v1/incidents/{random_id}")
    assert res.status_code == 404
    assert f"Incident '{random_id}' not found" in res.json()["detail"]


def test_get_incidents_summary(client, mock_session, sample_incidents):
    """Verify GET /api/v1/incidents/summary calculates aggregated tier metrics."""
    mock_session.scalars.return_value.all.return_value = sample_incidents

    res = client.get("/api/v1/incidents/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_incidents"] == 3
    assert data["total_acres"] == 14.0
    assert data["critical_count"] == 1
    assert data["critical_acres"] == 10.0
    assert data["high_count"] == 1
    assert data["high_acres"] == 3.0
    assert data["moderate_count"] == 1
    assert data["moderate_acres"] == 1.0
    assert data["riparian_breaches"] == 1


def test_get_incidents_geojson(client, mock_session, sample_incidents):
    """Verify GET /api/v1/incidents/geojson outputs valid FeatureCollection for QGIS."""
    mock_session.scalars.return_value.all.return_value = sample_incidents

    res = client.get("/api/v1/incidents/geojson")
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 3

    feat = data["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["violation_tier"] == "CRITICAL"
    assert feat["properties"]["encroached_rf_name"] == "Agumbe Rainforest Reserve Forest"
    assert feat["properties"]["area_acres"] == 10.0


def test_aoi_bounds_and_lists(client, mock_session):
    """Verify AOI bounds, forest boundaries, and stream buffers endpoints."""
    # 1. Bounds
    res_bounds = client.get("/api/v1/aoi/bounds")
    assert res_bounds.status_code == 200
    data_bounds = res_bounds.json()
    assert data_bounds["name"] == "Thirthahalli Taluk"
    assert len(data_bounds["bbox"]) == 4

    # 2. AOIs
    mock_aoi = AOI(id=uuid.uuid4(), name="Thirthahalli Taluk", category="Taluk", created_at=datetime.now(UTC))
    mock_session.scalars.return_value.all.return_value = [mock_aoi]
    res_aoi = client.get("/api/v1/aoi")
    assert res_aoi.status_code == 200
    assert len(res_aoi.json()) == 1
    assert res_aoi.json()[0]["name"] == "Thirthahalli Taluk"

    # 3. Forest boundaries
    poly = Polygon([(75.06, 13.55), (75.14, 13.55), (75.14, 13.62), (75.06, 13.62), (75.06, 13.55)])
    mock_forest = ForestBoundary(
        id=uuid.uuid4(),
        name="Agumbe Rainforest Reserve Forest",
        category="RF",
        area_acres=12000.0,
        geom=from_shape(MultiPolygon([poly]), srid=4326),
        created_at=datetime.now(UTC),
    )
    mock_session.scalars.return_value.all.return_value = [mock_forest]
    res_forest = client.get("/api/v1/aoi/forest-boundaries")
    assert res_forest.status_code == 200
    assert len(res_forest.json()) == 1
    assert res_forest.json()[0]["name"] == "Agumbe Rainforest Reserve Forest"

    # 4. Forest boundaries GeoJSON
    res_forest_geojson = client.get("/api/v1/aoi/forest-boundaries/geojson")
    assert res_forest_geojson.status_code == 200
    assert res_forest_geojson.json()["type"] == "FeatureCollection"
    assert len(res_forest_geojson.json()["features"]) == 1

    # 5. Watercourses
    line = LineString([(75.16, 13.61), (75.21, 13.64)])
    mock_water = WaterCourse(
        id=uuid.uuid4(),
        name="Tunga River Corridor",
        buffer_distance_m=50.0,
        geom=from_shape(MultiLineString([line]), srid=4326),
        buffer_geom=from_shape(MultiPolygon([poly]), srid=4326),
        created_at=datetime.now(UTC),
    )
    mock_session.scalars.return_value.all.return_value = [mock_water]
    res_water = client.get("/api/v1/aoi/water-courses")
    assert res_water.status_code == 200
    assert len(res_water.json()) == 1
    assert res_water.json()[0]["name"] == "Tunga River Corridor"

    # 6. Watercourses GeoJSON
    res_water_geojson = client.get("/api/v1/aoi/water-courses/geojson")
    assert res_water_geojson.status_code == 200
    assert res_water_geojson.json()["type"] == "FeatureCollection"


def test_get_incident_dossier_pdf_endpoint(client, mock_session, sample_incidents):
    """Verify GET /api/v1/incidents/{id}/dossier compiles and streams legal PDF."""
    target_inc = sample_incidents[0]
    mock_session.scalar.return_value = target_inc

    response = client.get(f"/api/v1/incidents/{target_inc.id}/dossier")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert f"Haddu_Dossier_{target_inc.id}.pdf" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")


def test_get_incident_dossier_pdf_not_found(client, mock_session):
    """Verify 404 response when requesting dossier for nonexistent incident ID."""
    mock_session.scalar.return_value = None
    random_id = uuid.uuid4()

    response = client.get(f"/api/v1/incidents/{random_id}/dossier")
    assert response.status_code == 404
    assert f"Incident '{random_id}' not found." in response.json()["detail"]
