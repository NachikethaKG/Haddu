"""Unit and integration tests for legal violation tiers and spatial intersection classifier."""

from unittest.mock import MagicMock

import geopandas as gpd
from geoalchemy2.shape import from_shape
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from haddu.core.classifier import (
    ViolationTier,
    classify_incident_violations,
    summarize_violations,
)
from haddu.db.models import ForestBoundary, WaterCourse


def test_empty_dataframe_classification():
    """Verify empty GeoDataFrame returns empty result with standardized legal columns."""
    empty_gdf = gpd.GeoDataFrame(columns=["patch_id", "geometry"], crs="EPSG:4326")
    classified = classify_incident_violations(empty_gdf)

    assert classified.empty
    expected_cols = {
        "violation_tier",
        "encroached_rf_name",
        "rf_area_acres",
        "riparian_breach",
        "esz_breach",
        "distance_to_rf_m",
        "nearest_rf_name",
        "violation_reasons",
    }
    assert expected_cols.issubset(set(classified.columns))


def test_critical_tier_inside_reserve_forest():
    """Verify clearing inside gazetted Reserve Forest is classified as CRITICAL."""
    # Polygon clearly inside Agumbe RF (75.06-75.14 lon, 13.55-13.62 lat)
    agumbe_interior_poly = Polygon([
        (75.08, 13.57),
        (75.10, 13.57),
        (75.10, 13.59),
        (75.08, 13.59),
        (75.08, 13.57),
    ])
    gdf = gpd.GeoDataFrame(
        [
            {
                "patch_id": 1,
                "area_sqm": 50000.0,
                "area_acres": 12.35,
                "geometry": agumbe_interior_poly,
            }
        ],
        crs="EPSG:4326",
    )

    classified = classify_incident_violations(gdf)
    row = classified.iloc[0]

    assert row["violation_tier"] == ViolationTier.CRITICAL
    assert "Agumbe" in str(row["encroached_rf_name"])
    assert row["rf_area_acres"] > 0.0
    assert row["distance_to_rf_m"] == 0.0
    assert bool(row["esz_breach"]) is True
    assert any("Inside Reserve Forest" in r for r in row["violation_reasons"])


def test_high_tier_riparian_buffer_breach():
    """Verify clearing intersecting 50m riparian watercourse buffer is classified as HIGH."""
    # Point along Tunga River channel at (75.29, 13.69)
    # A small 20m box around (75.2901, 13.6901) will lie within 50m of the river line
    riparian_poly = Polygon([
        (75.2900, 13.6900),
        (75.2902, 13.6900),
        (75.2902, 13.6902),
        (75.2900, 13.6902),
        (75.2900, 13.6900),
    ])
    gdf = gpd.GeoDataFrame(
        [
            {
                "patch_id": 2,
                "area_sqm": 1200.0,
                "area_acres": 0.30,
                "geometry": riparian_poly,
            }
        ],
        crs="EPSG:4326",
    )

    classified = classify_incident_violations(gdf)
    row = classified.iloc[0]

    assert row["violation_tier"] in (ViolationTier.HIGH, ViolationTier.CRITICAL)
    assert bool(row["riparian_breach"]) is True
    assert any("Riparian buffer breach" in r for r in row["violation_reasons"])


def test_high_tier_esz_breach():
    """Verify clearing within 200m of Reserve Forest is classified as HIGH (Eco-Sensitive Zone)."""
    # Kundadri Peak RF east boundary is at lon 75.22.
    # At lat 13.60, 0.001 deg longitude ≈ 108 meters.
    # So lon 75.221 is ~108 meters outside Kundadri RF (inside 200m ESZ).
    esz_poly = Polygon([
        (75.2210, 13.590),
        (75.2215, 13.590),
        (75.2215, 13.595),
        (75.2210, 13.595),
        (75.2210, 13.590),
    ])
    gdf = gpd.GeoDataFrame(
        [
            {
                "patch_id": 3,
                "area_sqm": 2500.0,
                "area_acres": 0.62,
                "geometry": esz_poly,
            }
        ],
        crs="EPSG:4326",
    )

    classified = classify_incident_violations(gdf)
    row = classified.iloc[0]

    assert row["violation_tier"] == ViolationTier.HIGH
    assert bool(row["esz_breach"]) is True
    assert 0.0 < row["distance_to_rf_m"] <= 200.0
    assert row["encroached_rf_name"] is None
    assert "Kundadri" in str(row["nearest_rf_name"])
    assert any("Eco-Sensitive Zone" in r for r in row["violation_reasons"])


def test_moderate_tier_private_agricultural_land():
    """Verify clearing far away from forests and watercourses is classified as MODERATE."""
    # Near Thirthahalli town center (75.235, 13.685) but away from the river (>200m) and >200m from RF
    revenue_poly = Polygon([
        (75.240, 13.675),
        (75.242, 13.675),
        (75.242, 13.677),
        (75.240, 13.677),
        (75.240, 13.675),
    ])
    gdf = gpd.GeoDataFrame(
        [
            {
                "patch_id": 4,
                "area_sqm": 4000.0,
                "area_acres": 0.99,
                "geometry": revenue_poly,
            }
        ],
        crs="EPSG:4326",
    )

    classified = classify_incident_violations(gdf)
    row = classified.iloc[0]

    assert row["violation_tier"] == ViolationTier.MODERATE
    assert bool(row["riparian_breach"]) is False
    assert bool(row["esz_breach"]) is False
    assert row["distance_to_rf_m"] > 200.0
    assert row["encroached_rf_name"] is None
    assert row["rf_area_acres"] == 0.0
    assert any("Private/Revenue" in r for r in row["violation_reasons"])


def test_summarize_violations():
    """Verify summarize_violations calculates correct counts and sums across legal tiers."""
    dummy_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame(
        [
            {
                "patch_id": 1,
                "area_acres": 10.0,
                "violation_tier": ViolationTier.CRITICAL,
                "encroached_rf_name": "Agumbe RF",
                "rf_area_acres": 10.0,
                "riparian_breach": False,
                "esz_breach": True,
                "geometry": dummy_poly,
            },
            {
                "patch_id": 2,
                "area_acres": 4.0,
                "violation_tier": ViolationTier.HIGH,
                "encroached_rf_name": None,
                "rf_area_acres": 0.0,
                "riparian_breach": True,
                "esz_breach": False,
                "geometry": dummy_poly,
            },
            {
                "patch_id": 3,
                "area_acres": 2.5,
                "violation_tier": ViolationTier.MODERATE,
                "encroached_rf_name": None,
                "rf_area_acres": 0.0,
                "riparian_breach": False,
                "esz_breach": False,
                "geometry": dummy_poly,
            },
        ],
        crs="EPSG:4326",
    )

    summary = summarize_violations(gdf)

    assert summary["total_incidents"] == 3
    assert summary["total_area_acres"] == 16.5
    assert summary["critical_count"] == 1
    assert summary["critical_acres"] == 10.0
    assert summary["high_count"] == 1
    assert summary["high_acres"] == 4.0
    assert summary["moderate_count"] == 1
    assert summary["moderate_acres"] == 2.5
    assert summary["riparian_breaches"] == 1
    assert summary["esz_breaches"] == 1
    assert len(summary["encroached_forest_tracts"]) == 1
    assert summary["encroached_forest_tracts"][0]["name"] == "Agumbe RF"


def test_db_session_boundary_loading():
    """Verify classifier correctly extracts boundaries from mock database models."""
    sample_coords = [(75.08, 13.57), (75.10, 13.57), (75.10, 13.59), (75.08, 13.59), (75.08, 13.57)]
    poly = Polygon(sample_coords)
    line = LineString([(75.16, 13.61), (75.21, 13.64)])

    mock_forest = ForestBoundary(
        name="Mock National Park",
        category="NationalPark",
        geom=from_shape(MultiPolygon([poly]), srid=4326),
    )
    mock_water = WaterCourse(
        name="Mock Stream",
        geom=from_shape(MultiLineString([line]), srid=4326),
        buffer_geom=from_shape(MultiPolygon([poly.buffer(0.001)]), srid=4326),
        buffer_distance_m=50.0,
    )

    mock_session = MagicMock()
    # Return mock records for ForestBoundary and WaterCourse
    mock_session.scalars.return_value.all.side_effect = [[mock_forest], [mock_water]]

    gdf = gpd.GeoDataFrame(
        [{"patch_id": 1, "area_acres": 2.0, "geometry": poly}],
        crs="EPSG:4326",
    )

    classified = classify_incident_violations(gdf, db_session=mock_session)
    assert len(classified) == 1
    assert classified.iloc[0]["violation_tier"] == ViolationTier.CRITICAL
    assert classified.iloc[0]["encroached_rf_name"] == "Mock National Park"
