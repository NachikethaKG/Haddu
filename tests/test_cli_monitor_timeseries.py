"""Unit and integration tests for time-series monitoring CLI and persistence update."""

from datetime import date, datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd

from haddu.core.harmonic import HarmonicModel
from haddu.core.persistence import PersistentBreakResult, plot_pixel_profile
from haddu.core.vectorizer import vectorize_persistent_breaks, vectorize_change_mask
from haddu.scripts.monitor_timeseries import (
    parse_args,
    record_persistent_incidents_to_db,
    run_monitoring_pipeline,
)


def _build_synthetic_persistent_result(h: int = 10, w: int = 10) -> PersistentBreakResult:
    """Helper to build a synthetic PersistentBreakResult with a 3x3 clearing patch."""
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
        "spatial_ref": 32643,
    }
    mask_arr = np.zeros((h, w), dtype=bool)
    mask_arr[2:5, 3:6] = True  # 9 pixels

    break_dates_arr = np.full((h, w), np.datetime64("NaT"), dtype="datetime64[ns]")
    break_dates_arr[2:5, 3:6] = np.datetime64("2026-03-15T00:00:00")

    break_idx_arr = np.full((h, w), -1, dtype=np.int32)
    break_idx_arr[2:5, 3:6] = 4

    z_scores_arr = np.zeros((6, h, w), dtype=np.float32)
    z_scores_arr[4:, 2:5, 3:6] = -4.5

    residuals_arr = np.zeros((6, h, w), dtype=np.float32)
    residuals_arr[4:, 2:5, 3:6] = -0.55

    da_mask = xr.DataArray(mask_arr, dims=["y", "x"], coords=spatial_coords)
    da_dates = xr.DataArray(break_dates_arr, dims=["y", "x"], coords=spatial_coords)
    da_idx = xr.DataArray(break_idx_arr, dims=["y", "x"], coords=spatial_coords)
    da_z = xr.DataArray(
        z_scores_arr,
        dims=["time", "y", "x"],
        coords={"time": pd.date_range("2026-01-01", periods=6, freq="15D"), **spatial_coords},
    )
    da_res = xr.DataArray(residuals_arr, dims=["time", "y", "x"], coords=da_z.coords)

    patch_info = [{
        "patch_id": 1,
        "pixel_count": 9,
        "area_sqm": 900.0,
        "area_acres": 0.2224,
        "earliest_break_date": "2026-03-15",
        "latest_break_date": "2026-03-15",
        "min_zscore": -4.5,
        "centroid_y": 1529965.0,
        "centroid_x": 730045.0,
        "centroid_lat": 13.82512,
        "centroid_lon": 75.12589,
    }]

    return PersistentBreakResult(
        change_mask=da_mask,
        break_dates=da_dates,
        break_indices=da_idx,
        z_scores=da_z,
        residuals=da_res,
        num_patches=1,
        total_area_sqm=900.0,
        total_area_acres=0.2224,
        patch_metrics=patch_info,
    )


def test_cli_argument_parsing():
    """Verify argument parsing defaults and overrides."""
    # Defaults
    args = parse_args([])
    assert args.target_year == 2026
    assert args.baseline_years == 2
    assert args.consecutive_passes == 2
    assert args.output_dir == "./output"
    assert args.z_threshold == 2.5
    assert args.min_area_sqm == 500.0
    assert not args.persist_db

    # Custom flags
    custom_args = parse_args([
        "--target-year", "2025",
        "--baseline-years", "3",
        "--consecutive-passes", "3",
        "--z-threshold", "3.0",
        "--min-area-sqm", "1000",
        "--output-dir", "./custom_out",
        "--persist-db",
    ])
    assert custom_args.target_year == 2025
    assert custom_args.baseline_years == 3
    assert custom_args.consecutive_passes == 3
    assert custom_args.z_threshold == 3.0
    assert custom_args.min_area_sqm == 1000.0
    assert custom_args.output_dir == "./custom_out"
    assert custom_args.persist_db


def test_vectorize_persistent_breaks_synthetic(tmp_path):
    """Verify vectorization of PersistentBreakResult into GeoJSON polygons with break attributes."""
    result = _build_synthetic_persistent_result(h=10, w=10)

    # 1. Vectorize
    gdf = vectorize_persistent_breaks(result, source_crs="EPSG:32643", target_crs="EPSG:4326")

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) == 1
    assert gdf.crs.to_epsg() == 4326

    # Verify required attribute columns
    row = gdf.iloc[0]
    assert row["patch_id"] == 1
    assert row["break_date"] == "2026-03-15"
    assert row["confidence_zscore"] == pytest.approx(4.5, abs=0.1)
    assert row["area_sqm"] == pytest.approx(900.0, abs=10.0)
    assert row["area_acres"] > 0.20
    assert row["centroid_lat"] > 0
    assert row["centroid_lon"] > 0

    # 2. GeoJSON serialization
    geojson_out = tmp_path / "test_persistent.geojson"
    gdf.to_file(geojson_out, driver="GeoJSON")
    assert geojson_out.exists()

    with open(geojson_out, "r") as f:
        data = json.load(f)

    props = data["features"][0]["properties"]
    assert props["break_date"] == "2026-03-15"
    assert props["confidence_zscore"] == pytest.approx(4.5, abs=0.1)
    assert props["area_acres"] > 0.20

    # 3. Interoperability with vectorize_change_mask
    gdf_via_mask = vectorize_change_mask(result)
    assert len(gdf_via_mask) == 1
    assert "break_date" in gdf_via_mask.columns


def test_plot_pixel_profile_synthetic(tmp_path):
    """Verify time-series verification profile plot generation."""
    h, w, t_steps = 4, 4, 20
    times = pd.date_range("2024-01-01", periods=t_steps, freq="20D")
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
    }
    coords = {"time": times, **spatial_coords}

    # Data with break at index 12 (2024-08-28)
    ndvi_data = np.full((t_steps, h, w), 0.80, dtype=np.float32)
    ndvi_data[12:, 1, 1] = 0.25

    cube = xr.Dataset(
        data_vars={"ndvi": (["time", "y", "x"], ndvi_data)},
        coords=coords,
    )

    # Harmonic model
    c0 = xr.DataArray(np.full((h, w), 0.80), dims=["y", "x"], coords=spatial_coords)
    c1 = xr.DataArray(np.zeros((h, w)), dims=["y", "x"], coords=spatial_coords)
    a1 = xr.DataArray(np.zeros((h, w)), dims=["y", "x"], coords=spatial_coords)
    b1 = xr.DataArray(np.zeros((h, w)), dims=["y", "x"], coords=spatial_coords)
    sigma = xr.DataArray(np.full((h, w), 0.03), dims=["y", "x"], coords=spatial_coords)
    amp = xr.DataArray(np.zeros((h, w)), dims=["y", "x"], coords=spatial_coords)
    phase = xr.DataArray(np.zeros((h, w)), dims=["y", "x"], coords=spatial_coords)

    model = HarmonicModel(
        c0=c0, c1=c1, a1=a1, b1=b1, sigma=sigma, amplitude=amp, phase=phase,
        reference_time=pd.Timestamp("2024-01-01"),
    )

    plot_file = tmp_path / "pixel_profile_13.62145_75.12458.png"

    dest = plot_pixel_profile(
        cube=cube,
        harmonic_model=model,
        y=spatial_coords["y"][1],
        x=spatial_coords["x"][1],
        break_date=times[12],
        lat=13.62145,
        lon=75.12458,
        z_threshold=2.5,
        output_path=plot_file,
    )

    assert dest == plot_file
    assert plot_file.exists()
    assert plot_file.stat().st_size > 10000  # PNG image > 10KB


def test_cli_end_to_end_mocked(monkeypatch, tmp_path):
    """Test full monitoring pipeline execution with mocked STAC data."""
    h, w, t_steps = 10, 10, 8
    # 2 baseline years (2024, 2025) and target year 2026
    times = pd.to_datetime([
        "2024-03-01", "2024-11-01",
        "2025-03-01", "2025-11-01",
        "2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15"
    ])
    spatial_coords = {
        "y": [1530000.0 - i * 10.0 for i in range(h)],
        "x": [730000.0 + j * 10.0 for j in range(w)],
        "spatial_ref": 32643,
    }
    coords = {"time": times, **spatial_coords}

    ndvi_data = np.full((t_steps, h, w), 0.80, dtype=np.float32)
    ndbi_data = np.full((t_steps, h, w), -0.25, dtype=np.float32)

    # In 2026 at pass 5 (2026-02-15), introduce persistent clearing at (2:5, 3:6)
    ndvi_data[5:, 2:5, 3:6] = 0.25
    ndbi_data[5:, 2:5, 3:6] = 0.22

    mock_cube = xr.Dataset(
        data_vars={
            "ndvi": (["time", "y", "x"], ndvi_data),
            "ndbi": (["time", "y", "x"], ndbi_data),
        },
        coords=coords,
    )

    monkeypatch.setattr(
        "haddu.scripts.monitor_timeseries.build_timeseries_cube",
        lambda **kwargs: mock_cube,
    )

    result = run_monitoring_pipeline(
        target_year=2026,
        baseline_years=2,
        consecutive_passes=2,
        output_dir=str(tmp_path),
        plot_top_n=1,
        persist_db=False,
    )

    assert result.num_patches == 1
    assert result.total_area_sqm == 900.0
    assert result.earliest_break_date == pd.Timestamp("2026-02-15")

    # Verify output GeoJSON
    geojson_path = tmp_path / "persistent_change_2026.geojson"
    assert geojson_path.exists()
    gdf = gpd.read_file(geojson_path)
    assert len(gdf) == 1
    assert str(gdf.iloc[0]["break_date"]).startswith("2026-02-15")

    # Verify output verification plot
    png_plots = list(tmp_path.glob("pixel_profile_*.png"))
    assert len(png_plots) == 1
    assert png_plots[0].stat().st_size > 10000


@pytest.mark.integration
def test_cli_end_to_end_real_imagery(tmp_path):
    """Integration test: Verify Acceptance Criteria running monitor_timeseries over real Sentinel-2 passes.

    Acceptance Criteria:
    Running python -m haddu.scripts.monitor_timeseries --baseline-years 2 --target-year 2026
    runs the harmonic fit, filters out seasonal phenology, exports verified persistent
    encroachment polygons, and saves temporal trajectory plots.
    """
    test_bbox = [75.10, 13.60, 75.15, 13.65]

    result = run_monitoring_pipeline(
        target_year=2026,
        baseline_years=2,
        consecutive_passes=2,
        output_dir=str(tmp_path),
        bbox=test_bbox,
        resolution=100,  # 100m for rapid execution
        max_cloud=20.0,
        max_passes=8,
        plot_top_n=1,
        persist_db=False,
    )

    assert isinstance(result, PersistentBreakResult)

    # Verify GeoJSON export exists and is valid
    geojson_path = tmp_path / "persistent_change_2026.geojson"
    assert geojson_path.exists()
    gdf = gpd.read_file(geojson_path)
    assert isinstance(gdf, gpd.GeoDataFrame)
