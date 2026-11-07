"""Unit and integration tests for Haddu Scheduled Audit Daemon."""

import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import pystac
from shapely.geometry import Polygon

from haddu.core.classifier import ViolationTier
from haddu.scripts.daemon import (
    build_parser,
    get_latest_recorded_pass_date,
    get_seasonal_sensor,
    query_stac_for_passes,
    run_audit_daemon_cycle,
    start_daemon_loop,
)


def test_get_seasonal_sensor():
    """Verify seasonal sensor selection logic matches Western Ghats meteorological patterns."""
    # Southwest Monsoon: June - September -> s1 (SAR Radar)
    assert get_seasonal_sensor(date(2026, 6, 1)) == "s1"
    assert get_seasonal_sensor(date(2026, 7, 15)) == "s1"
    assert get_seasonal_sensor(date(2026, 8, 20)) == "s1"
    assert get_seasonal_sensor(date(2026, 9, 30)) == "s1"

    # Dry / Post-Monsoon: October - May -> s2 (Optical MSI)
    assert get_seasonal_sensor(date(2026, 10, 1)) == "s2"
    assert get_seasonal_sensor(date(2026, 12, 25)) == "s2"
    assert get_seasonal_sensor(date(2026, 1, 15)) == "s2"
    assert get_seasonal_sensor(date(2026, 3, 10)) == "s2"
    assert get_seasonal_sensor(date(2026, 5, 31)) == "s2"


def test_daemon_cli_arguments():
    """Verify CLI parser parses all arguments with proper types and defaults."""
    parser = build_parser()
    args = parser.parse_args([
        "--interval-hours", "12",
        "--dry-run",
        "--once",
        "--sensor", "s1",
        "--aoi-name", "Custom AOI",
        "--output-dir", "./test_daemon_out",
        "--max-cloud", "15.0",
    ])

    assert args.interval_hours == 12.0
    assert args.dry_run is True
    assert args.once is True
    assert args.sensor == "s1"
    assert args.aoi_name == "Custom AOI"
    assert args.output_dir == Path("./test_daemon_out")
    assert args.max_cloud == 15.0


def test_get_latest_recorded_pass_date():
    """Verify database query for most recent satellite pass date."""
    mock_session = MagicMock()
    mock_session.scalar.return_value = date(2026, 8, 10)

    res = get_latest_recorded_pass_date(mock_session, sensor="s1", aoi_name="Thirthahalli Taluk")
    assert res == date(2026, 8, 10)
    assert mock_session.scalar.called

    # When query fails or returns None
    mock_session.scalar.return_value = None
    res_none = get_latest_recorded_pass_date(mock_session, sensor="s2", aoi_name="Thirthahalli Taluk")
    assert res_none is None


def test_query_stac_for_passes():
    """Verify routing to search_sar_passes vs search_passes based on sensor."""
    mock_s1_item = MagicMock(spec=pystac.Item)
    mock_s2_item = MagicMock(spec=pystac.Item)

    with (
        patch("haddu.scripts.daemon.search_sar_passes", return_value=[mock_s1_item]) as mock_s1,
        patch("haddu.scripts.daemon.search_passes", return_value=[mock_s2_item]) as mock_s2,
    ):
        res_s1 = query_stac_for_passes("s1", date(2026, 7, 1), date(2026, 7, 15))
        assert res_s1 == [mock_s1_item]
        assert mock_s1.called

        res_s2 = query_stac_for_passes("s2", date(2026, 1, 1), date(2026, 1, 15))
        assert res_s2 == [mock_s2_item]
        assert mock_s2.called


def test_run_audit_daemon_cycle_dry_run(tmp_path: Path):
    """Verify dry-run mode inspects catalogs and pipeline connectivity without mutating database."""
    mock_item = MagicMock(spec=pystac.Item)
    mock_item.id = "S2A_TEST_PASS_001"
    mock_item.datetime = datetime(2026, 11, 2, 5, 20, tzinfo=UTC)
    mock_item.properties = {"eo:cloud_cover": 5.4}

    with patch("haddu.scripts.daemon.query_stac_for_passes", return_value=[mock_item]):
        report = run_audit_daemon_cycle(
            sensor_override="s2",
            output_dir=tmp_path,
            dry_run=True,
        )

        assert report["status"] == "dry_run_success"
        assert report["sensor"] == "s2"
        assert report["stac_passes_count"] == 1
        assert report["stac_passes"][0]["id"] == "S2A_TEST_PASS_001"
        assert report["dossier_engine"] == "READY"


def test_run_audit_daemon_cycle_live_new_pass_detected(tmp_path: Path):
    """Verify live audit execution, dossier generation, and alert dispatch on new pass detection."""
    mock_item = MagicMock(spec=pystac.Item)
    mock_item.id = "S1A_MOCK_PASS_002"
    mock_item.datetime = datetime(2026, 8, 15, tzinfo=UTC)
    mock_item.properties = {"eo:cloud_cover": 0.0}
    mock_item.get_self_href.return_value = "https://stac.example.com/item.json"

    poly = Polygon([(75.08, 13.57), (75.10, 13.57), (75.10, 13.59), (75.08, 13.59), (75.08, 13.57)])
    classified_gdf = gpd.GeoDataFrame(
        [
            {
                "id": str(uuid.uuid4()),
                "patch_id": 1,
                "area_sqm": 40468.56,
                "area_acres": 10.0,
                "centroid_lat": 13.5824,
                "centroid_lon": 75.0912,
                "violation_tier": ViolationTier.CRITICAL,
                "encroached_rf_name": "Agumbe Rainforest Reserve Forest",
                "rf_area_acres": 10.0,
                "riparian_breach": False,
                "sensor": "s1",
                "pre_date": "2026-06-28",
                "post_date": "2026-08-15",
                "geometry": poly,
            }
        ],
        crs="EPSG:4326",
    )

    mock_summary = {
        "total_incidents": 1,
        "total_area_acres": 10.0,
        "critical_count": 1,
        "high_count": 0,
    }

    mock_session = MagicMock()
    # No prior pass recorded (cold start)
    mock_session.scalar.side_effect = [None, MagicMock(id=uuid.uuid4())]

    with (
        patch("haddu.scripts.daemon.query_stac_for_passes", return_value=[mock_item]),
        patch("haddu.scripts.daemon.run_audit_pipeline", return_value=(tmp_path / "out.geojson", classified_gdf, mock_summary)) as mock_run_audit,
        patch("haddu.scripts.daemon.compile_dossier_pdf") as mock_dossier,
        patch("haddu.scripts.daemon.dispatch_incident_alert_sync") as mock_alert,
    ):
        result = run_audit_daemon_cycle(
            sensor_override="s1",
            output_dir=tmp_path,
            dry_run=False,
            session_override=mock_session,
        )

        assert result["status"] == "audit_complete"
        assert result["sensor"] == "s1"
        assert result["incidents_count"] == 1
        assert result["critical_count"] == 1
        assert mock_run_audit.called
        assert mock_dossier.called
        assert mock_alert.called


def test_run_audit_daemon_cycle_already_processed(tmp_path: Path):
    """Verify daemon skips execution when latest pass is already audited."""
    mock_item = MagicMock(spec=pystac.Item)
    mock_item.id = "S2_ALREADY_PROCESSED"
    mock_item.datetime = datetime(2026, 3, 15, tzinfo=UTC)
    mock_item.properties = {"eo:cloud_cover": 0.0}

    mock_session = MagicMock()
    # Database already has pass recorded for 2026-03-15
    mock_session.scalar.return_value = date(2026, 3, 15)

    with patch("haddu.scripts.daemon.query_stac_for_passes", return_value=[mock_item]):
        result = run_audit_daemon_cycle(
            sensor_override="s2",
            output_dir=tmp_path,
            dry_run=False,
            session_override=mock_session,
        )

        assert result["status"] == "already_processed"
        assert result["pass_date"] == "2026-03-15"


def test_start_daemon_loop_once(tmp_path: Path):
    """Verify start_daemon_loop executes single cycle when once=True."""
    with patch("haddu.scripts.daemon.run_audit_daemon_cycle") as mock_cycle:
        mock_cycle.return_value = {"status": "dry_run_success"}

        start_daemon_loop(
            interval_hours=24.0,
            dry_run=True,
            once=True,
            output_dir=tmp_path,
        )

        assert mock_cycle.called
