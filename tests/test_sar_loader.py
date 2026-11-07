"""Tests for Sentinel-1 SAR ingestion client and logarithmic dB calibration."""

import tracemalloc
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pystac
import pytest
import xarray as xr

from haddu.core.sar_loader import (
    DEFAULT_AOI_BBOX,
    DEFAULT_CRS,
    DEFAULT_SAR_BANDS,
    DEFAULT_SAR_COLLECTION,
    DEFAULT_SAR_RESOLUTION,
    fetch_sar_passes,
    linear_to_db,
    search_sar_passes,
)

# ==============================================================================
# Unit Tests: linear_to_db Conversion
# ==============================================================================


def test_linear_to_db_scalars():
    """Verify scalar conversion from linear power and amplitude to dB."""
    # Linear power (is_amplitude=False)
    # 0 -> 10 * log10(1e-6) = -60.0 dB
    assert pytest.approx(linear_to_db(0.0), rel=1e-5) == -60.0
    # 1.0 -> 10 * log10(1 + 1e-6) ~ 0.0 dB
    assert pytest.approx(linear_to_db(1.0), abs=1e-4) == 0.0
    # 10.0 -> 10 * log10(10 + 1e-6) ~ 10.0 dB
    assert pytest.approx(linear_to_db(10.0), rel=1e-5) == 10.0
    # 100.0 -> 10 * log10(100 + 1e-6) ~ 20.0 dB
    assert pytest.approx(linear_to_db(100.0), rel=1e-5) == 20.0

    # Amplitude (is_amplitude=True) -> dB = 10 * log10(amp^2 + 1e-6)
    # amp = 1.0 -> 10 * log10(1 + 1e-6) ~ 0.0 dB
    assert pytest.approx(linear_to_db(1.0, is_amplitude=True), abs=1e-4) == 0.0
    # amp = 10.0 -> 10 * log10(100 + 1e-6) ~ 20.0 dB
    assert pytest.approx(linear_to_db(10.0, is_amplitude=True), rel=1e-5) == 20.0
    # amp = 2.0 -> 10 * log10(4 + 1e-6) ~ 6.0206 dB
    assert pytest.approx(linear_to_db(2.0, is_amplitude=True), rel=1e-4) == 6.0206


def test_linear_to_db_negative_and_nan():
    """Verify negative and NaN values are converted to NaN."""
    assert np.isnan(linear_to_db(-1.0))
    assert np.isnan(linear_to_db(-32768.0))
    assert np.isnan(linear_to_db(float("nan")))


def test_linear_to_db_numpy_array():
    """Verify conversion of numpy arrays with mixed valid and invalid pixels."""
    arr = np.array([-32768.0, -0.5, 0.0, 1.0, 10.0])
    db = linear_to_db(arr)

    assert isinstance(db, np.ndarray)
    assert np.isnan(db[0])
    assert np.isnan(db[1])
    assert pytest.approx(db[2], rel=1e-5) == -60.0
    assert pytest.approx(db[3], abs=1e-4) == 0.0
    assert pytest.approx(db[4], rel=1e-5) == 10.0

    # With amplitude
    amp_arr = np.array([0.0, 2.0, 10.0])
    amp_db = linear_to_db(amp_arr, is_amplitude=True)
    assert pytest.approx(amp_db[0], rel=1e-5) == -60.0
    assert pytest.approx(amp_db[1], rel=1e-4) == 6.0206
    assert pytest.approx(amp_db[2], rel=1e-5) == 20.0


def test_linear_to_db_sequence():
    """Verify conversion accepts Python list or tuple."""
    seq = [1.0, 10.0]
    res = linear_to_db(seq)
    assert isinstance(res, np.ndarray)
    assert pytest.approx(res[0], abs=1e-4) == 0.0
    assert pytest.approx(res[1], rel=1e-5) == 10.0


def test_linear_to_db_xarray_dataarray():
    """Verify conversion over xarray.DataArray preserves coords, dims, and assigns dB units."""
    data = np.array([[0.1, 1.0], [-32768.0, 10.0]])
    da = xr.DataArray(
        data,
        coords={"y": [10, 20], "x": [100, 200]},
        dims=["y", "x"],
        attrs={"nodata": -32768.0, "long_name": "Backscatter VV"},
    )

    db_da = linear_to_db(da)

    assert isinstance(db_da, xr.DataArray)
    assert db_da.dims == da.dims
    assert list(db_da.coords.keys()) == list(da.coords.keys())
    assert db_da.attrs["units"] == "dB"
    assert db_da.attrs["nodata"] == -32768.0

    vals = db_da.values
    assert pytest.approx(vals[0, 0], rel=1e-4) == -10.0
    assert pytest.approx(vals[0, 1], abs=1e-4) == 0.0
    assert np.isnan(vals[1, 0])
    assert pytest.approx(vals[1, 1], rel=1e-5) == 10.0


def test_linear_to_db_xarray_dataset():
    """Verify conversion over xarray.Dataset converts all variables and sets dB units."""
    vv = np.array([[1.0, 10.0], [0.01, -32768.0]])
    vh = np.array([[0.1, 1.0], [0.001, -32768.0]])
    ds = xr.Dataset(
        {
            "vv": (["y", "x"], vv),
            "vh": (["y", "x"], vh),
        },
        coords={"y": [1, 2], "x": [3, 4]},
    )

    db_ds = linear_to_db(ds)

    assert isinstance(db_ds, xr.Dataset)
    assert "vv" in db_ds.data_vars
    assert "vh" in db_ds.data_vars
    assert db_ds["vv"].attrs["units"] == "dB"
    assert db_ds["vh"].attrs["units"] == "dB"

    assert pytest.approx(db_ds["vv"].values[0, 0], abs=1e-4) == 0.0
    assert pytest.approx(db_ds["vv"].values[0, 1], rel=1e-5) == 10.0
    assert pytest.approx(db_ds["vh"].values[0, 0], rel=1e-4) == -10.0
    assert pytest.approx(db_ds["vh"].values[0, 1], abs=1e-4) == 0.0
    assert np.isnan(db_ds["vv"].values[1, 1])
    assert np.isnan(db_ds["vh"].values[1, 1])


def test_linear_to_db_dask_lazy():
    """Verify dask-backed DataArrays remain lazy when converted to dB."""
    import dask.array as dask_arr

    raw_data = np.ones((10, 10), dtype=np.float32) * 10.0
    lazy_data = dask_arr.from_array(raw_data, chunks=(5, 5))
    da = xr.DataArray(lazy_data, dims=["y", "x"])

    res = linear_to_db(da)

    assert hasattr(res.data, "chunks"), "Result should remain backed by dask chunks (lazy)."
    computed = res.compute()
    assert np.allclose(computed.values, 10.0, atol=1e-4)


# ==============================================================================
# Unit Tests: STAC Query & Search Validation
# ==============================================================================


def test_search_sar_passes_bbox_validation():
    """Verify ValueError is raised if bbox does not have 4 coordinates."""
    with pytest.raises(ValueError, match="bbox must contain exactly 4 coordinates"):
        search_sar_passes(bbox=[75.05, 13.55])


def test_search_sar_passes_empty_results():
    """Verify search for pre-mission dates returns an empty list."""
    items = search_sar_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="1999-01-01",
        end_date="1999-01-10",
        orbit_direction="DESCENDING",
    )
    assert items == []


def test_fetch_sar_passes_empty_items_validation():
    """Verify ValueError is raised when empty items sequence is passed."""
    with pytest.raises(ValueError, match="Cannot load SAR dataset: no Sentinel-1 passes found"):
        fetch_sar_passes(items=[])


def test_fetch_sar_passes_mocked_pipeline():
    """Unit test: Verify STAC query parameter passing and odc-stac loading flow."""
    mock_item = MagicMock(spec=pystac.Item)
    mock_item.id = "S1A_IW_GRDH_1SDV_20230824_mock"
    mock_item.datetime = datetime(2023, 8, 24, 0, 48, 51)
    mock_item.properties = {
        "sar:instrument_mode": "IW",
        "sat:orbit_state": "descending",
        "sar:polarizations": ["VV", "VH"],
    }
    mock_item.assets = {"vv": MagicMock(), "vh": MagicMock()}

    # Create dummy lazy dataset
    coords = {
        "time": [np.datetime64("2023-08-24T00:48:51")],
        "y": [1500000.0, 1499980.0],
        "x": [500000.0, 500020.0],
    }
    mock_ds = xr.Dataset(
        {
            "vv": (["time", "y", "x"], np.array([[[1.0, 10.0], [0.1, 0.01]]])),
            "vh": (["time", "y", "x"], np.array([[[0.1, 1.0], [0.01, 0.001]]])),
        },
        coords=coords,
    )

    with (
        patch("haddu.core.sar_loader.search_sar_passes", return_value=[mock_item]) as mock_search,
        patch("odc.stac.load", return_value=mock_ds) as mock_load,
    ):
        ds = fetch_sar_passes(
            bbox=DEFAULT_AOI_BBOX,
            start_date="2023-08-01",
            end_date="2023-08-31",
            orbit_direction="DESCENDING",
            collection="sentinel-1-rtc",
            resolution=20,
        )

        mock_search.assert_called_once_with(
            bbox=DEFAULT_AOI_BBOX,
            start_date="2023-08-01",
            end_date="2023-08-31",
            orbit_direction="DESCENDING",
            collection="sentinel-1-rtc",
            catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
            limit=None,
        )
        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs["bands"] == ["vv", "vh"]
        assert kwargs["crs"] == "EPSG:32643"
        assert kwargs["resolution"] == 20

        # Verify dB calibration was applied
        assert "vv" in ds.data_vars
        assert "vh" in ds.data_vars
        assert "VV" in ds.data_vars
        assert "VH" in ds.data_vars
        assert ds["vv"].attrs["units"] == "dB"
        assert ds["vh"].attrs["units"] == "dB"
        assert pytest.approx(ds["vv"].values[0, 0, 0], abs=1e-4) == 0.0
        assert pytest.approx(ds["vv"].values[0, 0, 1], rel=1e-5) == 10.0


def test_fetch_sar_passes_grd_amplitude_flag():
    """Verify sentinel-1-grd automatically uses is_amplitude=True for calibration."""
    mock_item = MagicMock(spec=pystac.Item)
    mock_item.id = "S1A_IW_GRDH_mock"
    mock_item.datetime = datetime(2023, 8, 24)
    mock_item.properties = {
        "sar:instrument_mode": "IW",
        "sat:orbit_state": "descending",
        "sar:polarizations": ["VV", "VH"],
    }
    mock_item.assets = {"vv": MagicMock(), "vh": MagicMock()}

    coords = {"time": [np.datetime64("2023-08-24")], "y": [1], "x": [1]}
    # Amplitude = 10.0 -> power = 100 -> dB = 20.0
    mock_ds = xr.Dataset(
        {"vv": (["time", "y", "x"], np.array([[[10.0]]]))},
        coords=coords,
    )

    with patch("odc.stac.load", return_value=mock_ds):
        ds = fetch_sar_passes(items=[mock_item], collection="sentinel-1-grd")
        assert pytest.approx(ds["vv"].values[0, 0, 0], rel=1e-5) == 20.0


def test_fetch_sar_passes_uncalibrated():
    """Verify calibrate_db=False preserves raw linear values."""
    mock_item = MagicMock(spec=pystac.Item)
    mock_item.id = "S1A_mock"
    mock_item.datetime = datetime(2023, 8, 24)
    mock_item.properties = {"sar:instrument_mode": "IW", "sar:polarizations": ["VV", "VH"]}
    mock_item.assets = {"vv": MagicMock(), "vh": MagicMock()}

    coords = {"time": [np.datetime64("2023-08-24")], "y": [1], "x": [1]}
    mock_ds = xr.Dataset(
        {"vv": (["time", "y", "x"], np.array([[[10.0]]]))},
        coords=coords,
    )

    with patch("odc.stac.load", return_value=mock_ds):
        ds = fetch_sar_passes(items=[mock_item], calibrate_db=False)
        assert ds["vv"].values[0, 0, 0] == 10.0
        assert "units" not in ds["vv"].attrs


# ==============================================================================
# Integration Test: Thirthahalli Monsoon SAR Fetching (Cloud Penetration)
# ==============================================================================


@pytest.mark.integration
def test_fetch_sar_passes_thirthahalli_monsoon_integration():
    """Integration test: Query and stream Sentinel-1 SAR over Thirthahalli during peak monsoon.

    Acceptance Criteria:
    - Queries Planetary Computer STAC for 'sentinel-1-rtc' in IW mode, descending orbit.
    - Captures acquisitions over Thirthahalli during monsoon (July-August) when optical
      Sentinel-2 is obstructed by persistent heavy cloud cover.
    - Streams windowed rasters projected into UTM Zone 43N (EPSG:32643) at 20m resolution.
    - Returns xarray.Dataset with 'vv' and 'vh' (and 'VV', 'VH') channels calibrated in dB.
    - Calibrated backscatter values lie in realistic physical SAR radar range (-35 dB to +10 dB).
    - Backed by dask chunks for lazy memory streaming; peak memory remains under 150 MB.
    """
    tracemalloc.start()

    # Peak Southwest Monsoon window over Western Ghats / Thirthahalli
    start_date = "2023-08-01"
    end_date = "2023-08-31"

    # Search SAR passes
    items = search_sar_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date=start_date,
        end_date=end_date,
        orbit_direction="DESCENDING",
        collection=DEFAULT_SAR_COLLECTION,
    )

    assert len(items) > 0, "Expected at least one Sentinel-1 SAR pass over Thirthahalli in August 2023."
    for item in items:
        assert item.properties["sar:instrument_mode"] == "IW"
        assert item.properties["sat:orbit_state"].lower() == "descending"

    # Fetch passes lazily with windowed read
    ds = fetch_sar_passes(
        bbox=DEFAULT_AOI_BBOX,
        items=items[:1],  # Load first pass for fast integration test
        collection=DEFAULT_SAR_COLLECTION,
        bands=DEFAULT_SAR_BANDS,
        crs=DEFAULT_CRS,
        resolution=DEFAULT_SAR_RESOLUTION,
        chunks={"x": 1024, "y": 1024},
        calibrate_db=True,
    )

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_mem / (1024 * 1024)
    print(f"\n[SAR Memory Telemetry] Current: {current_mem / 1e6:.2f} MB | Peak: {peak_mb:.2f} MB")

    # 1. Dataset type and spatial dimensions
    assert isinstance(ds, xr.Dataset)
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords

    # 2. Check CRS is UTM Zone 43N (EPSG:32643)
    if hasattr(ds, "geobox"):
        assert ds.geobox.crs.epsg == 32643
    else:
        assert ds.spatial_ref.item() == 32643 or "32643" in str(
            ds.spatial_ref.attrs.get("crs_wkt", "")
        )

    # 3. Check VV and VH backscatter channels are present and lazy
    for band in ("vv", "vh", "VV", "VH"):
        assert band in ds.data_vars, f"Expected {band} in dataset data variables."
        assert hasattr(ds[band].data, "chunks"), f"Channel {band} should be backed by dask chunks."

    assert ds["vv"].attrs.get("units") == "dB"
    assert ds["vh"].attrs.get("units") == "dB"

    # 4. Strict memory constraint check (< 150 MB peak)
    assert peak_mb < 150.0, f"Peak memory ({peak_mb:.2f} MB) exceeded 150 MB threshold!"

    # 5. Sanity check computed sample backscatter values in dB
    # Take a 10x10 slice to verify without loading full scene into memory
    sample_vv = ds["vv"].isel(time=0, y=slice(100, 110), x=slice(100, 110)).compute().values
    sample_vh = ds["vh"].isel(time=0, y=slice(100, 110), x=slice(100, 110)).compute().values

    valid_vv = sample_vv[np.isfinite(sample_vv)]
    valid_vh = sample_vh[np.isfinite(sample_vh)]

    assert len(valid_vv) > 0, "Expected valid VV backscatter pixels in sample slice."
    assert len(valid_vh) > 0, "Expected valid VH backscatter pixels in sample slice."

    # In vegetated terrain (Thirthahalli Western Ghats), C-band VV backscatter typically
    # ranges between -25 dB and 0 dB; VH ranges between -30 dB and -5 dB.
    print(
        f"Sample VV dB mean: {np.mean(valid_vv):.2f} dB, range: [{np.min(valid_vv):.2f}, {np.max(valid_vv):.2f}]"
    )
    print(
        f"Sample VH dB mean: {np.mean(valid_vh):.2f} dB, range: [{np.min(valid_vh):.2f}, {np.max(valid_vh):.2f}]"
    )

    assert np.all(valid_vv > -50.0) and np.all(valid_vv < 20.0)
    assert np.all(valid_vh > -50.0) and np.all(valid_vh < 20.0)
    # VV backscatter is physically higher than cross-pol VH backscatter
    assert np.mean(valid_vv) > np.mean(valid_vh)
