"""Vectorized spectral indices calculation engine for optical satellite imagery.

Provides standardized formulas for vegetation, canopy vigor, and fire/clearing disturbance
indices (NDVI, NDRE, NBR, dNBR) computed over xarray Datasets with automatic grid alignment
and validity mask preservation.
"""

import logging
from typing import Optional, Sequence, Union
import numpy as np
import scipy.ndimage as ndi
import xarray as xr

logger = logging.getLogger(__name__)

# Small positive constant to prevent division by zero in spectral ratios
DEFAULT_EPSILON = 1e-6


def _get_band(ds: xr.Dataset, *candidate_names: str) -> xr.DataArray:
    """Retrieve a spectral band DataArray by testing candidate names and aliases.

    Args:
        ds: Input xarray.Dataset.
        candidate_names: Preferred name and aliases (e.g. 'B08', 'nir').

    Returns:
        The matched xarray.DataArray.

    Raises:
        KeyError: If none of the candidate names are present in the dataset.
    """
    for name in candidate_names:
        if name in ds.data_vars:
            return ds[name]
        if name.lower() in ds.data_vars:
            return ds[name.lower()]
        if name.upper() in ds.data_vars:
            return ds[name.upper()]

    available_bands = list(ds.data_vars.keys())
    raise KeyError(
        f"None of candidate band names {candidate_names} found in dataset. "
        f"Available bands: {available_bands}"
    )


def align_to_grid(source: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Align or resample a source band (e.g., 20m B05/B12) to match a target grid (e.g., 10m B08).

    Args:
        source: Source DataArray to be aligned/resampled.
        target: Reference target DataArray providing spatial grid, dimensions, and coordinates.

    Returns:
        Resampled or interpolated DataArray matching target dimensions and coordinates.
    """
    # Fast path: identical dimensions, shape, and coordinates
    if source.dims == target.dims and source.shape == target.shape:
        coords_match = True
        for dim in target.dims:
            if dim in target.coords and dim in source.coords:
                if not np.array_equal(source.coords[dim].values, target.coords[dim].values):
                    coords_match = False
                    break
        if coords_match:
            return source

    # If coordinates are present along spatial dimensions 'x' and 'y'
    if (
        "x" in target.coords
        and "y" in target.coords
        and "x" in source.coords
        and "y" in source.coords
    ):
        try:
            return source.interp_like(target, method="nearest", kwargs={"fill_value": "extrapolate"})
        except Exception as exc:
            logger.debug(f"Coordinate interp_like failed ({exc}), falling back to array zoom.")

    # Fallback to scipy zoom or grid resize if coordinates are missing/unaligned
    if source.ndim == target.ndim:
        zoom_factors = [t / s for t, s in zip(target.shape, source.shape)]
        if all(abs(z - 1.0) < 1e-5 for z in zoom_factors):
            return source
        zoomed = ndi.zoom(source.values, zoom_factors, order=0)
        return xr.DataArray(
            zoomed,
            coords=target.coords,
            dims=target.dims,
            attrs=source.attrs,
        )

    return source


def calculate_ndvi(
    ds: xr.Dataset,
    valid_mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    clamp: bool = True,
    eps: float = DEFAULT_EPSILON,
) -> xr.DataArray:
    """Calculate the Normalized Difference Vegetation Index (NDVI).

    Formula: (B08 - B04) / (B08 + B04 + 1e-6)

    Args:
        ds: xarray.Dataset containing NIR (B08) and Red (B04) bands.
        valid_mask: Optional boolean mask where True indicates valid clean pixels.
                    If None and 'valid_mask' exists in ds, it will be automatically applied.
        clamp: Whether to clamp the resulting index values strictly to [-1.0, 1.0] (default: True).
        eps: Small constant added to denominator to avoid division by zero (default: 1e-6).

    Returns:
        xarray.DataArray representing NDVI with values strictly clamped between -1.0 and +1.0.
    """
    b08 = _get_band(ds, "B08", "b08", "nir").astype("float32")
    b04 = _get_band(ds, "B04", "b04", "red").astype("float32")

    b04_aligned = align_to_grid(b04, b08)

    numerator = b08 - b04_aligned
    denominator = b08 + b04_aligned + eps
    ndvi = numerator / denominator

    # Apply validity mask if provided or present in dataset
    if valid_mask is not None:
        ndvi = ndvi.where(valid_mask, np.nan)
    elif "valid_mask" in ds:
        ndvi = ndvi.where(ds["valid_mask"], np.nan)

    if clamp:
        ndvi = ndvi.clip(min=-1.0, max=1.0)

    ndvi.name = "ndvi"
    ndvi.attrs = {
        "long_name": "Normalized Difference Vegetation Index",
        "units": "1",
        "formula": "(B08 - B04) / (B08 + B04 + 1e-6)",
        "valid_range": [-1.0, 1.0],
    }
    return ndvi


def calculate_ndre(
    ds: xr.Dataset,
    valid_mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    b05: Optional[Union[xr.DataArray, np.ndarray]] = None,
    clamp: bool = True,
    eps: float = DEFAULT_EPSILON,
) -> xr.DataArray:
    """Calculate the Normalized Difference Red Edge Index (NDRE).

    Formula: (B08 - B05) / (B08 + B05 + 1e-6)
    Automatically resamples or aligns band B05 (RedEdge, native 20m) to match the B08 (10m) grid.

    Args:
        ds: xarray.Dataset containing NIR (B08) and optionally RedEdge 1 (B05) bands.
        valid_mask: Optional boolean mask where True indicates valid clean pixels.
                    If None and 'valid_mask' exists in ds, it will be automatically applied.
        b05: Optional explicit B05 DataArray or ndarray (if stored outside ds or in 20m resolution).
        clamp: Whether to clamp the resulting index values strictly to [-1.0, 1.0] (default: True).
        eps: Small constant added to denominator to avoid division by zero (default: 1e-6).

    Returns:
        xarray.DataArray representing NDRE with values strictly clamped between -1.0 and +1.0.
    """
    b08 = _get_band(ds, "B08", "b08", "nir").astype("float32")
    if b05 is None:
        b05_da = _get_band(ds, "B05", "b05", "rededge").astype("float32")
    elif isinstance(b05, xr.DataArray):
        b05_da = b05.astype("float32")
    else:
        b05_da = xr.DataArray(b05).astype("float32")

    b05_aligned = align_to_grid(b05_da, b08)

    numerator = b08 - b05_aligned
    denominator = b08 + b05_aligned + eps
    ndre = numerator / denominator

    # Apply validity mask if provided or present in dataset
    if valid_mask is not None:
        ndre = ndre.where(valid_mask, np.nan)
    elif "valid_mask" in ds:
        ndre = ndre.where(ds["valid_mask"], np.nan)

    if clamp:
        ndre = ndre.clip(min=-1.0, max=1.0)

    ndre.name = "ndre"
    ndre.attrs = {
        "long_name": "Normalized Difference Red Edge Index",
        "units": "1",
        "formula": "(B08 - B05) / (B08 + B05 + 1e-6)",
        "valid_range": [-1.0, 1.0],
    }
    return ndre


def calculate_nbr(
    ds: xr.Dataset,
    valid_mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    b12: Optional[Union[xr.DataArray, np.ndarray]] = None,
    clamp: bool = True,
    eps: float = DEFAULT_EPSILON,
) -> xr.DataArray:
    """Calculate the Normalized Burn Ratio (NBR).

    Formula: (B08 - B12) / (B08 + B12 + 1e-6)
    Automatically resamples or aligns band B12 (SWIR2, native 20m) to match the B08 (10m) grid.

    Args:
        ds: xarray.Dataset containing NIR (B08) and optionally SWIR2 (B12) bands.
        valid_mask: Optional boolean mask where True indicates valid clean pixels.
                    If None and 'valid_mask' exists in ds, it will be automatically applied.
        b12: Optional explicit B12 DataArray or ndarray (if stored outside ds or in 20m resolution).
        clamp: Whether to clamp the resulting index values strictly to [-1.0, 1.0] (default: True).
        eps: Small constant added to denominator to avoid division by zero (default: 1e-6).

    Returns:
        xarray.DataArray representing NBR with values strictly clamped between -1.0 and +1.0.
    """
    b08 = _get_band(ds, "B08", "b08", "nir").astype("float32")
    if b12 is None:
        b12_da = _get_band(ds, "B12", "b12", "swir2").astype("float32")
    elif isinstance(b12, xr.DataArray):
        b12_da = b12.astype("float32")
    else:
        b12_da = xr.DataArray(b12).astype("float32")

    b12_aligned = align_to_grid(b12_da, b08)

    numerator = b08 - b12_aligned
    denominator = b08 + b12_aligned + eps
    nbr = numerator / denominator

    # Apply validity mask if provided or present in dataset
    if valid_mask is not None:
        nbr = nbr.where(valid_mask, np.nan)
    elif "valid_mask" in ds:
        nbr = nbr.where(ds["valid_mask"], np.nan)

    if clamp:
        nbr = nbr.clip(min=-1.0, max=1.0)

    nbr.name = "nbr"
    nbr.attrs = {
        "long_name": "Normalized Burn Ratio",
        "units": "1",
        "formula": "(B08 - B12) / (B08 + B12 + 1e-6)",
        "valid_range": [-1.0, 1.0],
    }
    return nbr


def calculate_dnbr(
    pre: Union[xr.Dataset, xr.DataArray],
    post: Union[xr.Dataset, xr.DataArray],
    valid_mask: Optional[Union[xr.DataArray, np.ndarray]] = None,
    clamp: bool = True,
) -> xr.DataArray:
    """Calculate Delta Normalized Burn Ratio (dNBR = pre_nbr - post_nbr).

    Higher positive dNBR values indicate burn severity, canopy loss, or clearing disturbance.

    Args:
        pre: Pre-disturbance Dataset or pre-computed NBR DataArray.
        post: Post-disturbance Dataset or post-computed NBR DataArray.
        valid_mask: Optional combined boolean validity mask.
        clamp: Whether to clamp values strictly between -2.0 and +2.0 (default: True).

    Returns:
        xarray.DataArray representing dNBR change matrix.
    """
    pre_nbr = calculate_nbr(pre) if isinstance(pre, xr.Dataset) else pre
    post_nbr = calculate_nbr(post) if isinstance(post, xr.Dataset) else post

    dnbr = pre_nbr - post_nbr

    if valid_mask is not None:
        dnbr = dnbr.where(valid_mask, np.nan)

    if clamp:
        dnbr = dnbr.clip(min=-2.0, max=2.0)

    dnbr.name = "dnbr"
    dnbr.attrs = {
        "long_name": "Delta Normalized Burn Ratio",
        "units": "1",
        "formula": "pre_nbr - post_nbr",
        "valid_range": [-2.0, 2.0],
    }
    return dnbr
