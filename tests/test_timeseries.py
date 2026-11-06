"""Unit and integration tests for multi-temporal timeseries cube loader."""

from datetime import date, datetime
import numpy as np
import pytest
import xarray as xr

from haddu.core.timeseries import (
    DEFAULT_CUBE_BANDS,
    format_date_str,
    build_timeseries_cube,
)
from haddu.core.stac_client import DEFAULT_AOI_BBOX, search_passes


def test_format_date_str():
    """Verify format_date_str handles string, date, and datetime correctly."""
    assert format_date_str("2024-01-15") == "2024-01-15"
    assert format_date_str(date(2024, 3, 20)) == "2024-03-20"
    assert format_date_str(datetime(2024, 7, 4, 10, 30)) == "2024-07-04"


def test_build_timeseries_cube_empty_search():
    """Verify ValueError is raised when no passes are found in the query window."""
    with pytest.raises(ValueError, match="No Sentinel-2 passes found between 1999-01-01 and 1999-01-02"):
        build_timeseries_cube(
            bbox=DEFAULT_AOI_BBOX,
            start_date="1999-01-01",
            end_date="1999-01-02",
        )


def test_build_timeseries_cube_mocked(monkeypatch):
    """Unit test: Verify cube construction, masking, and index calculations with mock loader."""
    # Create synthetic 2-timestep raw dataset on 5x5 grid
    times = [np.datetime64("2024-01-15T05:20:00"), np.datetime64("2024-02-15T05:20:00")]
    coords = {
        "time": times,
        "y": [1530000.0, 1529990.0, 1529980.0, 1529970.0, 1529960.0],
        "x": [730000.0, 730010.0, 730020.0, 730030.0, 730040.0],
        "spatial_ref": 32643,
    }
    shape = (2, 5, 5)

    # Time 0: Healthy forest (B08=5000, B04=500, B11=1200, SCL=4)
    # Time 1: Cleared bare soil at pixel (1, 1), and cloud at pixel (4, 4)
    b08_data = np.full(shape, 5000.0, dtype=np.float32)
    b04_data = np.full(shape, 500.0, dtype=np.float32)
    b11_data = np.full(shape, 1200.0, dtype=np.float32)
    scl_data = np.full(shape, 4, dtype=np.int32)

    # Time 1 modifications
    b08_data[1, 1, 1] = 1500.0
    b04_data[1, 1, 1] = 1400.0
    b11_data[1, 1, 1] = 3200.0  # High SWIR1 -> Bare soil
    scl_data[1, 4, 4] = 9        # Cloud high probability at far corner (4, 4)

    mock_raw_ds = xr.Dataset(
        data_vars={
            "B08": (["time", "y", "x"], b08_data),
            "B04": (["time", "y", "x"], b04_data),
            "B11": (["time", "y", "x"], b11_data),
            "SCL": (["time", "y", "x"], scl_data),
        },
        coords=coords,
    )

    # Mock search_passes and load_stac_dataset
    mock_item = type("MockItem", (), {
        "id": "S2A_MSIL2A_20240115_T43PER",
        "datetime": datetime(2024, 1, 15),
        "properties": {"eo:cloud_cover": 5.0},
    })()
    monkeypatch.setattr("haddu.core.timeseries.search_passes", lambda **kwargs: [mock_item, mock_item])
    monkeypatch.setattr("haddu.core.timeseries.load_stac_dataset", lambda **kwargs: mock_raw_ds)

    cube = build_timeseries_cube(
        bbox=DEFAULT_AOI_BBOX,
        start_date="2024-01-01",
        end_date="2024-02-28",
        dilate_pixels=1,
    )

    assert isinstance(cube, xr.Dataset)
    assert set(cube.dims) == {"time", "y", "x"}
    assert "ndvi" in cube.data_vars
    assert "ndbi" in cube.data_vars
    assert "valid_mask" in cube.data_vars
    assert len(cube.time) == 2

    # Check Time 0: Healthy forest NDVI > 0.70, NDBI < 0
    t0_ndvi = float(cube.ndvi.values[0, 0, 0])
    t0_ndbi = float(cube.ndbi.values[0, 0, 0])
    assert t0_ndvi > 0.70
    assert t0_ndbi < 0.0

    # Check Time 1 pixel (1, 1): Bare soil -> Low NDVI (< 0.1), positive NDBI (> 0.2)
    t1_ndvi = float(cube.ndvi.values[1, 1, 1])
    t1_ndbi = float(cube.ndbi.values[1, 1, 1])
    assert t1_ndvi < 0.10
    assert t1_ndbi > 0.20

    # Check Time 1 pixel (4, 4): Clouded -> masked with NaN
    assert np.isnan(cube.ndvi.values[1, 4, 4])
    assert np.isnan(cube.ndbi.values[1, 4, 4])
    assert cube.valid_mask.values[1, 4, 4] == False


@pytest.mark.integration
def test_build_timeseries_cube_one_year_real_imagery():
    """Integration test: Build timeseries cube over a 1-year window for Thirthahalli.

    Acceptance Criteria:
    - Calling build_timeseries_cube() over a 1-year window yields an xarray dataset.
    - Dimensions are (time, y, x) with accurate chronological timestamps.
    - Contains clean, cloud-masked NDVI and NDBI arrays ready for temporal analysis.
    """
    # 1-year window in 2024 over Thirthahalli
    start_date = "2024-01-01"
    end_date = "2024-12-31"

    # Small test bounding box inside Thirthahalli Taluk at 100m resolution to ensure fast streaming (< 25s)
    test_bbox = [75.10, 13.60, 75.20, 13.70]

    cube = build_timeseries_cube(
        bbox=test_bbox,
        start_date=start_date,
        end_date=end_date,
        max_cloud=25.0,
        resolution=100,  # 100m for rapid multi-temporal test execution
        dilate_pixels=2,
        max_passes=4,    # Sample multiple passes across seasons
        compute=True,
    )

    # 1. Dataset type and dimensions
    assert isinstance(cube, xr.Dataset)
    assert set(cube.dims) == {"time", "y", "x"}
    assert len(cube.time) >= 2, "Expected multiple passes across the year."

    # 2. Variable presence
    assert "ndvi" in cube.data_vars
    assert "ndbi" in cube.data_vars
    assert "valid_mask" in cube.data_vars

    # 3. Chronological time coordinates
    time_vals = cube.time.values
    for i in range(len(time_vals) - 1):
        assert time_vals[i] <= time_vals[i + 1], "Timestamps must be chronologically sorted."

    # 4. Values within [-1.0, 1.0] and valid canopy signal
    valid_ndvi = cube.ndvi.values[~np.isnan(cube.ndvi.values)]
    assert len(valid_ndvi) > 0, "Expected valid NDVI observations."
    assert np.all(valid_ndvi >= -1.0) and np.all(valid_ndvi <= 1.0)
    assert np.max(valid_ndvi) > 0.50

    valid_ndbi = cube.ndbi.values[~np.isnan(cube.ndbi.values)]
    assert len(valid_ndbi) > 0, "Expected valid NDBI observations."
    assert np.all(valid_ndbi >= -1.0) and np.all(valid_ndbi <= 1.0)
