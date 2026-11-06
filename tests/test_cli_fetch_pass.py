"""Tests for the end-to-end extraction CLI script and visual sanity check."""

import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from haddu.scripts.fetch_pass import fetch_and_export_pass


def test_scl_mask_calculation():
    """Verify SCL cloud mask logic identifies classes 3, 8, 9, 10 as invalid."""
    # SCL test array: 4 vegetation (4), 1 cloud shadow (3), 1 cloud med (8), 1 cirrus (10), 3 water (6)
    # Total pixels = 10; Invalid = 3 (3, 8, 10); Valid = 7 (70.0%)
    scl = np.array([4, 4, 4, 4, 3, 8, 10, 6, 6, 6], dtype=np.uint8)

    invalid_mask = np.isin(scl, [3, 8, 9, 10])
    total_pixels = scl.size
    valid_pixels = int(np.count_nonzero(~invalid_mask))
    valid_pct = (valid_pixels / total_pixels) * 100.0

    assert valid_pixels == 7
    assert valid_pct == 70.0


def test_rgb_normalization_and_clipping():
    """Verify raw reflectance (0-10000) normalizes to [0, 1] without overflow."""
    raw_reflectance = np.array([0, 1000, 3000, 10000, 15000], dtype=np.float32)

    normalized = np.clip(raw_reflectance / 10000.0, 0.0, 1.0)
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    assert normalized[0] == 0.0
    assert normalized[1] == 0.1
    assert normalized[3] == 1.0
    assert normalized[4] == 1.0


def test_fetch_and_export_pass_with_mocked_stac(tmp_path):
    """Test fetch_and_export_pass workflow with mocked STAC search and loader."""
    mock_item = MagicMock()
    mock_item.id = "S2B_MSIL2A_TEST_T43PER"
    mock_item.bbox = [75.0, 13.4, 75.5, 13.9]
    mock_item.properties = {"eo:cloud_cover": 2.5, "platform": "Sentinel-2B"}
    mock_item.datetime.strftime.return_value = "2026-01-15"
    mock_item.datetime.date.return_value = "2026-01-15"
    mock_item.datetime.isoformat.return_value = "2026-01-15T05:20:00Z"

    # Mock xarray Dataset with small synthetic arrays
    mock_ds = MagicMock()
    mock_arr = MagicMock()
    mock_arr.isel.return_value.values.astype.return_value = np.zeros((50, 50), dtype=np.float32)
    mock_scl_arr = MagicMock()
    mock_scl_arr.isel.return_value.values = np.full((50, 50), 4, dtype=np.uint8)

    mock_ds.B04 = mock_arr
    mock_ds.B03 = mock_arr
    mock_ds.B02 = mock_arr
    mock_ds.SCL = mock_scl_arr
    mock_ds.compute.return_value = mock_ds

    with patch("haddu.scripts.fetch_pass.search_passes", return_value=[mock_item]), \
         patch("haddu.scripts.fetch_pass.load_stac_dataset", return_value=mock_ds), \
         patch("haddu.scripts.fetch_pass.init_db"), \
         patch("haddu.scripts.fetch_pass.get_db"):

        result = fetch_and_export_pass(
            start_date="2026-01-01",
            end_date="2026-01-31",
            max_cloud=15,
            output_dir=str(tmp_path),
            resolution=20,
        )

        assert result["stac_item_id"] == "S2B_MSIL2A_TEST_T43PER"
        assert result["date"] == "2026-01-15"
        assert result["valid_cloud_free_pct"] == 100.0
        assert Path(result["preview_filepath"]).is_file()
