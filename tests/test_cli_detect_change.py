"""Unit and integration tests for change vectorizer and detect_change CLI."""

from datetime import date
from pathlib import Path
import json
import numpy as np
import pytest
from shapely.geometry import Polygon
import xarray as xr

from haddu.core.change_detector import detect_canopy_loss, ChangeDetectionResult
from haddu.core.vectorizer import vectorize_change_mask, extract_affine_transform
from haddu.scripts.detect_change import (
    build_parser,
    parse_date,
    render_difference_heatmap,
    record_incidents_to_db,
    run_change_detection_pipeline,
)
from tests.test_change_detector import make_synthetic_scene


def test_cli_argument_parsing():
    """Verify detect_change CLI arguments are parsed correctly."""
    parser = build_parser()
    args = parser.parse_args([
        "--pre-date", "2025-12-15",
        "--post-date", "2026-03-15",
        "--threshold", "0.30",
        "--min-area-sqm", "600",
        "--output-dir", "./test_output",
        "--skip-db",
    ])

    assert args.pre_date == date(2025, 12, 15)
    assert args.post_date == date(2026, 3, 15)
    assert args.threshold == 0.30
    assert args.min_area_sqm == 600.0
    assert args.output_dir == Path("./test_output")
    assert args.skip_db is True


def test_cli_invalid_date_format():
    """Verify that invalid date formats raise ArgumentTypeError."""
    with pytest.raises(Exception):
        parse_date("2026/03/15")


def test_vectorize_change_mask_synthetic(tmp_path):
    """Verify vectorization of detected clearing clusters into a GeoDataFrame."""
    # Pre-scene: healthy canopy (B08=5000, B04=500)
    pre_ds = make_synthetic_scene(shape=(15, 15), b08_val=5000.0, b04_val=500.0)
    post_ds = pre_ds.copy(deep=True)

    # 3x3 clearing patch (9 pixels = 900 m²)
    post_ds["B08"].values[3:6, 3:6] = 1200.0
    post_ds["B04"].values[3:6, 3:6] = 1500.0

    result = detect_canopy_loss(pre_ds, post_ds, ndvi_threshold=0.25, min_area_sqm=500.0)
    gdf = vectorize_change_mask(result, source_crs="EPSG:32643", target_crs="EPSG:4326")

    assert len(gdf) == 1
    assert gdf.crs.to_string() == "EPSG:4326"

    row = gdf.iloc[0]
    assert row["patch_id"] == 1
    assert row["area_sqm"] == 900.0
    assert row["area_acres"] == round(900.0 / 4046.8564224, 4)
    assert row["mean_delta_ndvi"] is not None and row["mean_delta_ndvi"] > 0.60
    assert row["max_delta_ndvi"] is not None

    # Check centroid coordinates lie within valid geographic range
    assert 70.0 < row["centroid_lon"] < 80.0
    assert 10.0 < row["centroid_lat"] < 20.0

    # Test GeoJSON export
    out_file = tmp_path / "change_test.geojson"
    gdf.to_file(out_file, driver="GeoJSON")
    assert out_file.exists()

    with open(out_file, "r") as f:
        data = json.load(f)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    assert data["features"][0]["properties"]["area_sqm"] == 900.0


def test_vectorize_empty_change_mask():
    """Verify that empty change masks return a valid empty GeoDataFrame."""
    pre_ds = make_synthetic_scene(shape=(10, 10))
    post_ds = pre_ds.copy(deep=True)

    result = detect_canopy_loss(pre_ds, post_ds, ndvi_threshold=0.25)
    gdf = vectorize_change_mask(result)

    assert len(gdf) == 0
    assert "patch_id" in gdf.columns
    assert "area_sqm" in gdf.columns
    assert "centroid_lon" in gdf.columns
    assert "geometry" in gdf.columns


def test_render_difference_heatmap(tmp_path):
    """Verify heatmap PNG is generated and saved correctly."""
    delta_arr = np.random.uniform(0.0, 0.6, size=(20, 20))
    mask_arr = (delta_arr > 0.25)
    out_png = tmp_path / "heatmap_test.png"

    render_difference_heatmap(
        delta_ndvi=delta_arr,
        change_mask=mask_arr,
        pre_date_str="2025-12-15",
        post_date_str="2026-03-15",
        threshold=0.25,
        output_path=out_png,
        num_patches=1,
        total_acres=0.5,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 5000  # Non-trivial image file


def test_record_incidents_to_db_empty():
    """Verify record_incidents_to_db handles empty GeoDataFrames with zero count."""
    pre_ds = make_synthetic_scene(shape=(5, 5))
    gdf = vectorize_change_mask(detect_canopy_loss(pre_ds, pre_ds))
    count = record_incidents_to_db(
        gdf=gdf,
        pre_date=date(2025, 12, 15),
        post_date=date(2026, 3, 15),
        aoi_name="Thirthahalli Taluk",
        bbox=[75.05, 13.55, 75.40, 13.85],
    )
    assert count == 0


@pytest.mark.integration
def test_cli_end_to_end_real_imagery(tmp_path):
    """Integration test: Execute full pipeline between 2025-12-15 and 2026-03-15 passes.

    Acceptance Criteria:
    - Finds valid Sentinel-2 passes.
    - Saves output/change_<pre>_<post>.geojson.
    - Saves output/change_heatmap_<pre>_<post>.png.
    - Generates valid output files without crashing.
    """
    out_dir = tmp_path / "output"

    # Use a small spatial test window inside Thirthahalli Taluk to keep test fast (< 25s)
    test_bbox = [75.10, 13.60, 75.20, 13.70]

    geojson_path, heatmap_path, _ = run_change_detection_pipeline(
        pre_date=date(2025, 12, 15),
        post_date=date(2026, 3, 15),
        threshold=0.25,
        min_area_sqm=500.0,
        output_dir=out_dir,
        bbox=test_bbox,
        skip_db=True,
    )

    assert geojson_path.exists(), f"Expected GeoJSON at {geojson_path}"
    assert heatmap_path.exists(), f"Expected Heatmap PNG at {heatmap_path}"

    with open(geojson_path, "r") as f:
        geojson_data = json.load(f)
    assert geojson_data["type"] == "FeatureCollection"
    assert heatmap_path.stat().st_size > 10000
