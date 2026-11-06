"""Tests for STAC client querying and windowed raster ingestion engine."""

import tracemalloc
import pytest
import xarray as xr
from haddu.core.stac_client import (
    DEFAULT_AOI_BBOX,
    search_passes,
)
from haddu.core.loader import (
    DEFAULT_TARGET_BANDS,
    DEFAULT_CRS,
    load_stac_dataset,
)


def test_stac_client_bbox_validation():
    """Verify ValueError is raised when bbox length is invalid."""
    with pytest.raises(ValueError, match="bbox must contain exactly 4 coordinates"):
        search_passes(bbox=[75.05, 13.55])


def test_stac_client_empty_search():
    """Verify that a search yielding no results returns an empty list without error."""
    # A single day in 1999 before Sentinel-2 was launched
    items = search_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="1999-01-01",
        end_date="1999-01-02",
        max_cloud=10,
    )
    assert items == []


def test_loader_empty_items_validation():
    """Verify that attempting to load an empty items sequence raises ValueError."""
    with pytest.raises(ValueError, match="Cannot load dataset: items list is empty."):
        load_stac_dataset(items=[])


@pytest.mark.integration
def test_stac_search_and_windowed_load_peak_memory():
    """Integration test: Query known cloud-free Sentinel-2 pass and lazily load bands.

    Acceptance Criteria:
    - Returns xarray.Dataset with coordinates in UTM Zone 43N (EPSG:32643).
    - Contains all required target bands (B02, B03, B04, B08, B11, B12, SCL).
    - Backed by dask chunks to prevent memory spikes.
    - Peak memory consumption remains strictly under 150 MB.
    """
    tracemalloc.start()

    # Known clear-sky dry season window over Thirthahalli
    items = search_passes(
        bbox=DEFAULT_AOI_BBOX,
        start_date="2024-01-15",
        end_date="2024-01-17",
        max_cloud=15,
    )

    assert len(items) > 0, "Expected at least one Sentinel-2 pass for Jan 2024 in Thirthahalli."

    # Lazily stream windowed target bands into xarray
    ds = load_stac_dataset(
        items=items[:1],
        bands=DEFAULT_TARGET_BANDS,
        crs=DEFAULT_CRS,
        resolution=10,
        bbox=DEFAULT_AOI_BBOX,
        chunks={"x": 2048, "y": 2048},
    )

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_mem / (1024 * 1024)
    print(f"\n[Memory Telemetry] Current: {current_mem / 1e6:.2f} MB | Peak: {peak_mb:.2f} MB")

    # 1. Dataset type and dimensions
    assert isinstance(ds, xr.Dataset)
    assert "x" in ds.coords
    assert "y" in ds.coords
    assert "time" in ds.coords

    # 2. Check CRS is UTM Zone 43N (EPSG:32643)
    if hasattr(ds, "geobox"):
        assert ds.geobox.crs.epsg == 32643
    else:
        assert ds.spatial_ref.item() == 32643 or "32643" in str(ds.spatial_ref.attrs.get("crs_wkt", ""))

    # 3. Check all target bands are present as lazy dask arrays
    for band in DEFAULT_TARGET_BANDS:
        assert band in ds.data_vars, f"Expected band {band} in dataset data variables."
        # Verify lazy execution (backed by dask chunks)
        assert hasattr(ds[band].data, "chunks"), f"Band {band} should be backed by dask chunks."

    # 4. Strict memory constraint check (< 150 MB peak)
    assert peak_mb < 150.0, f"Peak memory ({peak_mb:.2f} MB) exceeded 150 MB threshold!"
