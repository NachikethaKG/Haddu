"""Unit and integration tests for run_audit CLI and multi-sensor legal audit pipeline."""

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pystac
import pytest
import xarray as xr
from shapely.geometry import Polygon

from haddu.core.classifier import ViolationTier
from haddu.scripts.run_audit import (
    build_parser,
    find_best_sar_pass_near_date,
    parse_date,
    print_audit_report,
    record_audit_violations_to_db,
    run_audit_pipeline,
)


def test_audit_cli_arguments():
    """Verify run_audit CLI arguments are parsed correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--sensor", "s1",
        "--pre-date", "2026-07-01",
        "--post-date", "2026-08-15",
        "--threshold", "2.8",
        "--min-area-sqm", "600",
        "--output-dir", "./test_audit_out",
        "--skip-db",
    ])

    assert args.sensor == "s1"
    assert args.pre_date == date(2026, 7, 1)
    assert args.post_date == date(2026, 8, 15)
    assert args.threshold == 2.8
    assert args.min_area_sqm == 600.0
    assert args.output_dir == Path("./test_audit_out")
    assert args.skip_db is True


def test_audit_cli_invalid_date_and_sensor():
    """Verify invalid sensor or date format raises argument errors."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_date("2026/08/15")

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--sensor", "modis", "--pre-date", "2026-07-01", "--post-date", "2026-08-15"])


def test_find_best_sar_pass_near_date():
    """Verify find_best_sar_pass_near_date returns pass closest to target date."""
    target = date(2026, 7, 10)

    # 3 mock items with different dates
    item1 = MagicMock(spec=pystac.Item)
    item1.id = "S1_ITEM_FAR"
    item1.datetime = datetime(2026, 7, 1, 0, 0, 0)

    item2 = MagicMock(spec=pystac.Item)
    item2.id = "S1_ITEM_CLOSE"
    item2.datetime = datetime(2026, 7, 11, 0, 0, 0)

    item3 = MagicMock(spec=pystac.Item)
    item3.id = "S1_ITEM_LATER"
    item3.datetime = datetime(2026, 7, 20, 0, 0, 0)

    with patch("haddu.scripts.run_audit.search_sar_passes", return_value=[item1, item2, item3]):
        selected = find_best_sar_pass_near_date(target)
        assert selected.id == "S1_ITEM_CLOSE"


def test_run_audit_pipeline_s1_mocked(tmp_path):
    """Verify full end-to-end SAR audit pipeline execution with mocked STAC data."""
    mock_pre_item = MagicMock(spec=pystac.Item)
    mock_pre_item.id = "S1_MOCK_PRE"
    mock_pre_item.datetime = datetime(2026, 7, 10)

    mock_post_item = MagicMock(spec=pystac.Item)
    mock_post_item.id = "S1_MOCK_POST"
    mock_post_item.datetime = datetime(2026, 8, 15)

    # Mock SAR raster datasets
    ny, nx = 15, 15
    y_coords = np.arange(1530000.0, 1530000.0 - ny * 20.0, -20.0)
    x_coords = np.arange(500000.0, 500000.0 + nx * 20.0, 20.0)
    coords = {"y": y_coords, "x": x_coords, "spatial_ref": 32643}

    vh_pre = np.full((ny, nx), -12.0, dtype=np.float32)
    vh_post = np.full((ny, nx), -12.0, dtype=np.float32)

    # Synthetic disturbance cluster at rows 3:7, cols 3:7
    vh_post[3:7, 3:7] = -16.0  # 4 dB drop

    mock_pre_ds = xr.Dataset({"vh": (["y", "x"], vh_pre)}, coords=coords)
    mock_post_ds = xr.Dataset({"vh": (["y", "x"], vh_post)}, coords=coords)

    with (
        patch("haddu.scripts.run_audit.find_best_sar_pass_near_date", side_effect=[mock_pre_item, mock_post_item]),
        patch("haddu.scripts.run_audit.fetch_sar_passes", side_effect=[mock_pre_ds, mock_post_ds]),
    ):
        geojson_path, gdf, summary = run_audit_pipeline(
            sensor="s1",
            pre_date=date(2026, 7, 1),
            post_date=date(2026, 8, 15),
            output_dir=tmp_path,
            skip_db=True,
        )

        assert geojson_path.exists()
        assert len(gdf) >= 1
        assert "violation_tier" in gdf.columns
        assert summary["total_incidents"] >= 1
        assert summary["total_area_acres"] > 0.0

        # Verify summary JSON was written
        summary_file = tmp_path / "audit_summary_s1_2026-07-01_2026-08-15.json"
        assert summary_file.exists()
        with open(summary_file, "r") as f:
            saved_summary = json.load(f)
        assert saved_summary["total_incidents"] == summary["total_incidents"]


def test_run_audit_pipeline_s2_mocked(tmp_path):
    """Verify full end-to-end optical audit pipeline execution with mocked STAC data."""
    mock_pre_item = MagicMock(spec=pystac.Item)
    mock_pre_item.id = "S2_MOCK_PRE"
    mock_pre_item.datetime = datetime(2026, 1, 10)

    mock_post_item = MagicMock(spec=pystac.Item)
    mock_post_item.id = "S2_MOCK_POST"
    mock_post_item.datetime = datetime(2026, 3, 15)

    ny, nx = 15, 15
    y_coords = np.arange(1530000.0, 1530000.0 - ny * 10.0, -10.0)
    x_coords = np.arange(500000.0, 500000.0 + nx * 10.0, 10.0)
    coords = {"y": y_coords, "x": x_coords, "spatial_ref": 32643}

    b08 = np.full((ny, nx), 5000.0, dtype=np.float32)
    b04 = np.full((ny, nx), 500.0, dtype=np.float32)
    scl = np.full((ny, nx), 4, dtype=np.uint8)  # Vegetation SCL class

    mock_pre_ds = xr.Dataset({"B08": (["y", "x"], b08), "B04": (["y", "x"], b04), "SCL": (["y", "x"], scl)}, coords=coords)

    b08_post = b08.copy()
    b04_post = b04.copy()
    b08_post[4:8, 4:8] = 1200.0  # NDVI drop
    b04_post[4:8, 4:8] = 1500.0
    mock_post_ds = xr.Dataset({"B08": (["y", "x"], b08_post), "B04": (["y", "x"], b04_post), "SCL": (["y", "x"], scl)}, coords=coords)

    with (
        patch("haddu.scripts.run_audit.find_best_pass_near_date", side_effect=[mock_pre_item, mock_post_item]),
        patch("haddu.scripts.run_audit.load_stac_dataset", side_effect=[mock_pre_ds, mock_post_ds]),
    ):
        geojson_path, gdf, summary = run_audit_pipeline(
            sensor="s2",
            pre_date=date(2026, 1, 1),
            post_date=date(2026, 3, 31),
            output_dir=tmp_path,
            skip_db=True,
        )

        assert geojson_path.exists()
        assert len(gdf) >= 1
        assert "violation_tier" in gdf.columns
        assert summary["total_incidents"] >= 1


def test_record_audit_violations_to_db_mocked():
    """Verify database insertion of classified incidents with legal tier columns."""
    sample_poly = Polygon([(75.08, 13.57), (75.10, 13.57), (75.10, 13.59), (75.08, 13.59), (75.08, 13.57)])
    gdf = gpd.GeoDataFrame(
        [
            {
                "patch_id": 10,
                "area_sqm": 5000.0,
                "area_acres": 1.24,
                "centroid_lat": 13.58,
                "centroid_lon": 75.09,
                "mean_delta_ndvi": 0.42,
                "max_delta_ndvi": 0.58,
                "violation_tier": ViolationTier.CRITICAL,
                "encroached_rf_name": "Agumbe Rainforest Reserve Forest",
                "rf_area_acres": 1.24,
                "riparian_breach": False,
                "esz_breach": True,
                "distance_to_rf_m": 0.0,
                "nearest_rf_name": "Agumbe Rainforest Reserve Forest",
                "violation_reasons": ["Inside Reserve Forest: Agumbe (1.24 acres)"],
                "geometry": sample_poly,
            }
        ],
        crs="EPSG:4326",
    )

    mock_session = MagicMock()
    mock_session.scalar.return_value = MagicMock(id=MagicMock())  # Existing AOI

    with (
        patch("haddu.scripts.run_audit.init_db"),
        patch("haddu.scripts.run_audit.get_db") as mock_get_db,
    ):
        mock_get_db.return_value.__enter__.return_value = mock_session

        count = record_audit_violations_to_db(
            classified_gdf=gdf,
            sensor="s1",
            pre_date=date(2026, 7, 1),
            post_date=date(2026, 8, 15),
            aoi_name="Thirthahalli Taluk",
            bbox=[75.05, 13.55, 75.40, 13.85],
        )

        assert count == 1
        assert mock_session.add.called
        assert mock_session.commit.called


def test_print_audit_report(capsys):
    """Verify print_audit_report displays formatted output without error."""
    summary = {
        "aoi_name": "Thirthahalli Taluk",
        "sensor": "s1",
        "pre_date": "2026-07-01",
        "post_date": "2026-08-15",
        "total_incidents": 5,
        "total_area_acres": 18.5,
        "critical_count": 2,
        "critical_acres": 10.2,
        "high_count": 2,
        "high_acres": 6.1,
        "moderate_count": 1,
        "moderate_acres": 2.2,
        "riparian_breaches": 1,
        "esz_breaches": 2,
        "encroached_forest_tracts": [{"name": "Agumbe RF", "incidents": 2, "rf_area_acres": 10.2}],
    }
    dummy_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame(
        [
            {
                "patch_id": 1,
                "violation_tier": "CRITICAL",
                "area_acres": 5.1,
                "encroached_rf_name": "Agumbe RF",
                "nearest_rf_name": "Agumbe RF",
                "violation_reasons": ["Inside Reserve Forest: Agumbe RF"],
                "geometry": dummy_poly,
            }
        ],
        crs="EPSG:4326",
    )

    print_audit_report(summary, gdf)
    captured = capsys.readouterr()
    assert "HADDU SATELLITE LEGAL AUDIT REPORT" in captured.out
    assert "CRITICAL" in captured.out
    assert "Agumbe RF" in captured.out
