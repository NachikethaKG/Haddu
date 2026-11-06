"""Harmonic seasonal baseline decomposition engine for multi-temporal satellite NDVI.

Fits a 1-harmonic Fourier curve over historical vegetation indices via vectorized
Ordinary Least Squares (OLS) regression to model natural phenology in the Western Ghats:
    y(t) = c_0 + c_1 * t + a_1 * cos(2 * pi * t / 365.25) + b_1 * sin(2 * pi * t / 365.25)

Enables pixel-wise phenological baseline modeling, residual computation, standard
error estimation, and residual standard deviation (sigma) mapping to distinguish stable
evergreen canopy (low sigma) from deciduous/scrub variations and anthropogenic forest loss.
"""

from dataclasses import dataclass
from datetime import date, datetime
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Standard astronomical tropical year length in days
ANNUAL_PERIOD_DAYS: float = 365.25

# Minimum number of valid (non-NaN) observations required to solve 4 OLS parameters
DEFAULT_MIN_VALID_OBS: int = 4

# Numerical ridge regularization parameter to guarantee matrix invertibility
DEFAULT_RIDGE_ALPHA: float = 1e-8


def extract_time_array(
    time_coord: Any,
    reference_time: Optional[Union[str, datetime, pd.Timestamp, np.datetime64]] = None,
    time_mode: str = "continuous",
) -> Tuple[np.ndarray, Optional[pd.Timestamp]]:
    """Extract numeric time offsets t (in days) from xarray or datetime coordinates.

    Args:
        time_coord: Sequence, numpy array, xarray DataArray, or pandas DatetimeIndex.
        reference_time: Reference origin t0 for continuous days calculation.
            If None (default), defaults to Jan 1 00:00:00 of the earliest year in time_coord.
            If 'first_pass', uses the earliest timestamp.
        time_mode: 'continuous' for continuous days elapsed since reference_time,
            or 'doy' for fractional day of year.

    Returns:
        Tuple of (t_array, resolved_reference_timestamp).
    """
    raw_vals = getattr(time_coord, "values", time_coord)

    # 1. Handle already numeric time coordinates (e.g. synthetic test offsets)
    if isinstance(raw_vals, (int, float, np.integer, np.floating)):
        return np.array([float(raw_vals)], dtype=np.float64), None
    if isinstance(raw_vals, np.ndarray) and np.issubdtype(raw_vals.dtype, np.number):
        return raw_vals.astype(np.float64), None

    # 2. Convert datetime-like values to pandas DatetimeIndex
    dt_series = pd.to_datetime(raw_vals)
    if isinstance(dt_series, pd.Timestamp):
        dt_index = pd.DatetimeIndex([dt_series])
    else:
        dt_index = pd.DatetimeIndex(dt_series)

    # Resolve reference timestamp
    if reference_time is None:
        min_year = int(dt_index.year.min())
        ref_ts = pd.Timestamp(year=min_year, month=1, day=1)
    elif str(reference_time).lower() == "first_pass":
        ref_ts = pd.Timestamp(dt_index.min())
    else:
        ref_ts = pd.Timestamp(reference_time)

    # Compute t in days
    if time_mode.lower() == "doy":
        if hasattr(dt_index, "day_of_year"):
            t = dt_index.day_of_year.values.astype(np.float64)
        else:
            t = dt_index.dayofyear.values.astype(np.float64)
    else:
        # Continuous days elapsed from ref_ts
        t = (dt_index - ref_ts).total_seconds().values / 86400.0

    return t, ref_ts


def construct_harmonic_design_matrix(
    t: np.ndarray,
    period: float = ANNUAL_PERIOD_DAYS,
) -> np.ndarray:
    """Build the 4-column OLS harmonic design matrix X: [1, t, cos(omega*t), sin(omega*t)].

    Args:
        t: 1D array of time coordinates in days of length N.
        period: Harmonic period length in days (default: 365.25 days).

    Returns:
        2D numpy array of shape (N, 4).
    """
    t_1d = np.atleast_1d(np.asarray(t, dtype=np.float64))
    omega = 2.0 * np.pi / period
    x0 = np.ones_like(t_1d)
    x1 = t_1d
    x2 = np.cos(omega * t_1d)
    x3 = np.sin(omega * t_1d)
    return np.column_stack([x0, x1, x2, x3])


@dataclass
class HarmonicModel:
    """Container for fitted Fourier harmonic regression model parameters and metrics.

    Attributes:
        c0: Intercept / baseline constant DataArray.
        c1: Linear trend slope DataArray (NDVI change per day).
        a1: Cosine coefficient DataArray.
        b1: Sine coefficient DataArray.
        sigma: Pixel-wise residual standard deviation DataArray (RMSE).
        amplitude: Seasonal cycle amplitude DataArray: sqrt(a1^2 + b1^2).
        phase: Seasonal peak phase DataArray in radians: arctan2(b1, a1).
        reference_time: Reference origin timestamp t0 used for continuous time t.
        time_mode: Mode used for time evaluation ('continuous' or 'doy').
        period: Harmonic period in days (default: 365.25).
        se_c0: Standard error DataArray of c0.
        se_c1: Standard error DataArray of c1.
        se_a1: Standard error DataArray of a1.
        se_b1: Standard error DataArray of b1.
        r2: Coefficient of determination DataArray (R^2).
        valid_obs: Number of non-NaN observations per pixel DataArray.
    """

    c0: xr.DataArray
    c1: xr.DataArray
    a1: xr.DataArray
    b1: xr.DataArray
    sigma: xr.DataArray
    amplitude: xr.DataArray
    phase: xr.DataArray
    reference_time: Optional[pd.Timestamp] = None
    time_mode: str = "continuous"
    period: float = ANNUAL_PERIOD_DAYS
    se_c0: Optional[xr.DataArray] = None
    se_c1: Optional[xr.DataArray] = None
    se_a1: Optional[xr.DataArray] = None
    se_b1: Optional[xr.DataArray] = None
    r2: Optional[xr.DataArray] = None
    valid_obs: Optional[xr.DataArray] = None

    def __iter__(self) -> Iterator[Union[xr.Dataset, xr.DataArray]]:
        """Enable tuple unpacking: coeffs_ds, sigma_da = model."""
        return iter((self.coefficients, self.sigma))

    def __getitem__(self, key: str) -> xr.DataArray:
        """Allow dict-like access: model['sigma'], model['c0'], etc."""
        return self.to_dataset()[key]

    @property
    def coefficients(self) -> xr.Dataset:
        """Dataset of the four primary regression coefficients (c0, c1, a1, b1)."""
        return xr.Dataset(
            {
                "c0": self.c0,
                "c1": self.c1,
                "a1": self.a1,
                "b1": self.b1,
            },
            attrs={"description": "Harmonic Fourier OLS regression coefficients"},
        )

    @property
    def standard_errors(self) -> xr.Dataset:
        """Dataset of coefficient standard errors and residual standard deviation."""
        vars_dict: Dict[str, xr.DataArray] = {"sigma": self.sigma}
        if self.se_c0 is not None:
            vars_dict["se_c0"] = self.se_c0
        if self.se_c1 is not None:
            vars_dict["se_c1"] = self.se_c1
        if self.se_a1 is not None:
            vars_dict["se_a1"] = self.se_a1
        if self.se_b1 is not None:
            vars_dict["se_b1"] = self.se_b1
        return xr.Dataset(
            vars_dict,
            attrs={"description": "Harmonic Fourier OLS standard errors and sigma"},
        )

    def to_dataset(self) -> xr.Dataset:
        """Combine all model parameters, standard errors, and diagnostics into an xr.Dataset."""
        data_vars = {
            "c0": self.c0,
            "c1": self.c1,
            "a1": self.a1,
            "b1": self.b1,
            "sigma": self.sigma,
            "amplitude": self.amplitude,
            "phase": self.phase,
        }
        if self.se_c0 is not None:
            data_vars["se_c0"] = self.se_c0
        if self.se_c1 is not None:
            data_vars["se_c1"] = self.se_c1
        if self.se_a1 is not None:
            data_vars["se_a1"] = self.se_a1
        if self.se_b1 is not None:
            data_vars["se_b1"] = self.se_b1
        if self.r2 is not None:
            data_vars["r2"] = self.r2
        if self.valid_obs is not None:
            data_vars["valid_obs"] = self.valid_obs

        attrs = {
            "title": "Haddu Harmonic Seasonal Baseline Model",
            "period_days": float(self.period),
            "time_mode": self.time_mode,
            "reference_time": str(self.reference_time) if self.reference_time else "None",
        }
        return xr.Dataset(data_vars=data_vars, attrs=attrs)

    def predict(
        self,
        time_coords: Union[str, datetime, pd.Timestamp, Sequence, xr.DataArray, xr.Dataset],
        squeeze: bool = True,
    ) -> xr.DataArray:
        """Evaluate fitted harmonic curve at arbitrary timestamps or xarray coordinate.

        Args:
            time_coords: Single timestamp, sequence of timestamps, or xarray object
                with a 'time' dimension.
            squeeze: If True and evaluating a single scalar timestamp, squeezes the
                length-1 'time' dimension yielding a 2D spatial array.

    Returns:
        xr.DataArray containing predicted baseline NDVI with spatial dimensions
        matching the fitted model.
        """
        # If passed a Dataset or DataArray with 'time', extract coordinates
        if isinstance(time_coords, xr.Dataset):
            raw_time = time_coords["time"]
        elif isinstance(time_coords, xr.DataArray):
            raw_time = time_coords
        else:
            raw_time = time_coords

        is_scalar = isinstance(raw_time, (str, datetime, date, pd.Timestamp, int, float))

        # Extract time offsets
        t_vals, _ = extract_time_array(
            raw_time,
            reference_time=self.reference_time,
            time_mode=self.time_mode,
        )
        m_steps = len(t_vals)

        # Build prediction design matrix (M, 4)
        x_pred = construct_harmonic_design_matrix(t_vals, period=self.period)

        # Extract coefficient values and shape
        c0_arr = np.asarray(self.c0.values)
        spatial_shape = c0_arr.shape
        p_pixels = int(np.prod(spatial_shape)) if spatial_shape else 1

        beta_matrix = np.column_stack(
            [
                np.asarray(self.c0.values).ravel(),
                np.asarray(self.c1.values).ravel(),
                np.asarray(self.a1.values).ravel(),
                np.asarray(self.b1.values).ravel(),
            ]
        )  # Shape (P, 4)

        # Vectorized matrix product: (M, 4) @ (4, P) -> (M, P)
        y_pred_flat = x_pred @ beta_matrix.T

        # Reshape to (M, *spatial_shape)
        if spatial_shape:
            y_pred_arr = y_pred_flat.reshape((m_steps, *spatial_shape))
        else:
            y_pred_arr = y_pred_flat.squeeze(-1)

        # Build coordinates dictionary
        pred_coords = dict(self.c0.coords)
        if is_scalar and squeeze:
            out_dims = self.c0.dims
            out_arr = y_pred_arr[0]
        else:
            out_dims = ("time", *self.c0.dims)
            out_arr = y_pred_arr
            # Attach time coordinate
            if isinstance(raw_time, xr.DataArray) and "time" in raw_time.dims:
                pred_coords["time"] = raw_time
            elif hasattr(raw_time, "__iter__") and not isinstance(raw_time, str):
                pred_coords["time"] = list(raw_time)
            else:
                pred_coords["time"] = [raw_time]

        da_pred = xr.DataArray(
            out_arr,
            dims=out_dims,
            coords=pred_coords,
            name="ndvi_baseline",
            attrs={
                "long_name": "Harmonic Baseline Predicted NDVI",
                "period_days": float(self.period),
            },
        )
        return da_pred

    def compute_residuals(
        self,
        cube_or_da: Union[xr.Dataset, xr.DataArray],
    ) -> xr.DataArray:
        """Compute observed minus predicted residuals: res_t = NDVI_t - hat{NDVI}_t.

        Args:
            cube_or_da: xarray Dataset containing 'ndvi' or DataArray of NDVI.

        Returns:
            xr.DataArray of residuals with same dimensions and coordinates as input.
        """
        if isinstance(cube_or_da, xr.Dataset):
            if "ndvi" not in cube_or_da.data_vars:
                raise ValueError("Dataset must contain 'ndvi' data variable.")
            da = cube_or_da["ndvi"]
        else:
            da = cube_or_da

        pred = self.predict(da.time, squeeze=False)
        residuals = da - pred
        residuals.name = "ndvi_residual"
        residuals.attrs["long_name"] = "NDVI Residual (Observed - Seasonal Baseline)"
        return residuals

    def compute_zscores(
        self,
        cube_or_da: Union[xr.Dataset, xr.DataArray],
        min_sigma: float = 0.01,
    ) -> xr.DataArray:
        """Compute pixel-wise standardized z-scores: z_t = (NDVI_t - hat{NDVI}_t) / sigma.

        Args:
            cube_or_da: xarray Dataset containing 'ndvi' or DataArray of NDVI.
            min_sigma: Lower clamp floor on residual standard deviation to avoid
                division by zero or runaway z-scores on invariant pixels.

        Returns:
            xr.DataArray of z-scores with negative values indicating vegetation loss.
        """
        residuals = self.compute_residuals(cube_or_da)
        safe_sigma = xr.where(self.sigma > min_sigma, self.sigma, min_sigma)
        zscore = residuals / safe_sigma
        zscore.name = "ndvi_zscore"
        zscore.attrs["long_name"] = "Standardized NDVI Residual Z-Score"
        return zscore


def fit_harmonic_baseline(
    ndvi_cube: Union[xr.Dataset, xr.DataArray],
    min_valid_obs: int = DEFAULT_MIN_VALID_OBS,
    time_dim: str = "time",
    reference_time: Optional[Union[str, datetime, pd.Timestamp, np.datetime64]] = None,
    time_mode: str = "continuous",
    period: float = ANNUAL_PERIOD_DAYS,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
) -> HarmonicModel:
    """Fit a 1-harmonic Fourier OLS phenological baseline over NDVI observations.

    Fits the linear harmonic regression equation:
        y(t) = c_0 + c_1 * t + a_1 * cos(2 * pi * t / 365.25) + b_1 * sin(2 * pi * t / 365.25)

    Vectorized across all spatial pixels via normal equations and analytical matrix
    inversion, handling cloud/shadow masked NaN values on an individual pixel basis.

    Args:
        ndvi_cube: xarray.Dataset containing an 'ndvi' variable, or xarray.DataArray
            with dimensions (time, y, x) or (time, ...).
        min_valid_obs: Minimum non-NaN observations required per pixel (default: 4).
        time_dim: Name of the time dimension (default: 'time').
        reference_time: Origin timestamp t0 for continuous time representation.
            Defaults to Jan 1 of the earliest year in the dataset.
        time_mode: Time representation mode: 'continuous' (default) or 'doy'.
        period: Harmonic period length in days (default: 365.25).
        ridge_alpha: Small L2 regularization weight added to diagonal of X^T X
            to ensure well-conditioned inversion for collinear samples.

    Returns:
        HarmonicModel instance containing fitted coefficient arrays (c0, c1, a1, b1),
        residual standard deviation (sigma), standard error arrays, amplitude, phase,
        and prediction/evaluation methods.

    Raises:
        ValueError: If input does not contain the specified time dimension or valid data.
    """
    # 1. Extract NDVI DataArray
    if isinstance(ndvi_cube, xr.Dataset):
        if "ndvi" not in ndvi_cube.data_vars:
            raise ValueError("Input Dataset must contain an 'ndvi' data variable.")
        da = ndvi_cube["ndvi"]
    elif isinstance(ndvi_cube, xr.DataArray):
        da = ndvi_cube
    else:
        raise TypeError(
            f"Expected xarray Dataset or DataArray, received {type(ndvi_cube).__name__}."
        )

    if time_dim not in da.dims:
        raise ValueError(
            f"Expected time dimension '{time_dim}' in input array, found {da.dims}."
        )

    # 2. Extract numeric time offsets t in days
    t_vals, ref_ts = extract_time_array(
        da[time_dim],
        reference_time=reference_time,
        time_mode=time_mode,
    )
    n_obs = len(t_vals)
    if n_obs < min_valid_obs:
        raise ValueError(
            f"Input time-series has only {n_obs} observation(s), but min_valid_obs={min_valid_obs}."
        )

    # 3. Construct (N, 4) design matrix X: [1, t, cos(2*pi*t/T), sin(2*pi*t/T)]
    x_matrix = construct_harmonic_design_matrix(t_vals, period=period)

    # 4. Prepare data arrays and spatial shapes
    spatial_dims = [d for d in da.dims if d != time_dim]
    spatial_shape = tuple(da.sizes[d] for d in spatial_dims)
    p_pixels = int(np.prod(spatial_shape)) if spatial_shape else 1

    # Extract raw values: ensure shape (N, P)
    # Move time_dim to first axis if not already
    da_ordered = da.transpose(time_dim, *spatial_dims)
    y_raw = da_ordered.values
    y_2d = y_raw.reshape((n_obs, p_pixels))

    # Pixel validity mask: 1.0 for valid, 0.0 for NaN
    w_mask = (~np.isnan(y_2d)).astype(np.float64)  # (N, P)
    valid_counts = np.sum(w_mask, axis=0)  # (P,)
    good_pixels = valid_counts >= min_valid_obs  # (P,)

    # Zero out NaNs for matrix multiplication
    y_zeroed = np.nan_to_num(y_2d, nan=0.0)  # (N, P)

    # 5. Vectorized normal equations via einsum
    # outer_X: (N, 4, 4) tensor of row outer products X_t * X_t^T
    outer_x = x_matrix[:, :, None] * x_matrix[:, None, :]
    # XtX for each pixel: sum_t w_{t,p} * (X_t * X_t^T) -> shape (P, 4, 4)
    xtx = np.einsum("nij,np->pij", outer_x, w_mask)
    # XtY for each pixel: sum_t X_{t,i} * Y_{t,p} -> shape (P, 4)
    xty = np.einsum("ni,np->pi", x_matrix, y_zeroed)

    # 6. Regularized OLS solution for valid pixels
    beta_flat = np.full((p_pixels, 4), np.nan, dtype=np.float64)
    se_flat = np.full((p_pixels, 4), np.nan, dtype=np.float64)
    sigma_flat = np.full(p_pixels, np.nan, dtype=np.float64)
    r2_flat = np.full(p_pixels, np.nan, dtype=np.float64)

    n_good = int(np.sum(good_pixels))
    if n_good > 0:
        # Add small ridge regularization to prevent singular matrix inversion
        ridge = ridge_alpha * np.eye(4, dtype=np.float64)
        xtx_good = xtx[good_pixels] + ridge
        xty_good = xty[good_pixels]

        # Solve (P_good, 4, 4) @ (P_good, 4, 1) -> (P_good, 4)
        beta_good = np.linalg.solve(xtx_good, xty_good[:, :, None]).squeeze(-1)
        beta_flat[good_pixels] = beta_good

        # Analytical covariance matrix: Cov(beta) = sigma^2 * (X^T X)^{-1}
        inv_xtx_good = np.linalg.inv(xtx_good)
        diag_inv_good = np.diagonal(inv_xtx_good, axis1=1, axis2=2)  # (P_good, 4)

        # 7. Fitted values, residuals, and residual standard deviation (sigma)
        # Y_pred: (N, P_good) = X_matrix @ beta_good^T
        y_pred_good = x_matrix @ beta_good.T
        y_good = y_2d[:, good_pixels]
        w_good = w_mask[:, good_pixels]

        residuals_good = np.where(w_good > 0, y_good - y_pred_good, 0.0)
        ss_res_good = np.sum(residuals_good**2, axis=0)  # (P_good,)
        counts_good = valid_counts[good_pixels]

        # sigma = sqrt( (1 / N_valid) * sum (y - hat{y})^2 )
        sigma_good = np.sqrt(ss_res_good / counts_good)
        sigma_flat[good_pixels] = sigma_good

        # Standard errors for each coefficient: se_j = sigma * sqrt(inv_xtx_jj)
        se_good = sigma_good[:, None] * np.sqrt(np.maximum(0.0, diag_inv_good))
        se_flat[good_pixels] = se_good

        # 8. Coefficient of determination R^2
        y_zeroed_good = y_zeroed[:, good_pixels]
        y_mean_good = np.sum(y_zeroed_good, axis=0) / counts_good
        ss_tot_good = np.sum(w_good * (y_zeroed_good - y_mean_good[None, :]) ** 2, axis=0)

        r2_vals = np.full(n_good, np.nan, dtype=np.float64)
        has_variance = ss_tot_good > 1e-8
        r2_vals[has_variance] = np.clip(
            1.0 - (ss_res_good[has_variance] / ss_tot_good[has_variance]),
            -1.0,
            1.0,
        )
        # For perfectly constant profiles where ss_tot == 0
        r2_vals[~has_variance] = np.where(ss_res_good[~has_variance] <= 1e-8, 1.0, 0.0)
        r2_flat[good_pixels] = r2_vals

    # 9. Compute amplitude and phase
    amplitude_flat = np.sqrt(beta_flat[:, 2] ** 2 + beta_flat[:, 3] ** 2)
    phase_flat = np.arctan2(beta_flat[:, 3], beta_flat[:, 2])

    # 10. Reshape arrays back to spatial dimensions and wrap into xarray DataArrays
    spatial_coords = {d: da.coords[d] for d in spatial_dims if d in da.coords}
    if "spatial_ref" in da.coords:
        spatial_coords["spatial_ref"] = da.coords["spatial_ref"]

    def _wrap_da(
        arr_1d: np.ndarray,
        name: str,
        long_name: str,
        units: Optional[str] = None,
    ) -> xr.DataArray:
        if spatial_shape:
            reshaped = arr_1d.reshape(spatial_shape)
        else:
            reshaped = arr_1d[0]
        attrs: Dict[str, Any] = {"long_name": long_name}
        if units:
            attrs["units"] = units
        return xr.DataArray(
            reshaped,
            dims=spatial_dims,
            coords=spatial_coords,
            name=name,
            attrs=attrs,
        )

    da_c0 = _wrap_da(beta_flat[:, 0], "c0", "Harmonic Baseline Intercept", "NDVI")
    da_c1 = _wrap_da(beta_flat[:, 1], "c1", "Harmonic Linear Trend Slope", "NDVI/day")
    da_a1 = _wrap_da(beta_flat[:, 2], "a1", "Harmonic Cosine Amplitude", "NDVI")
    da_b1 = _wrap_da(beta_flat[:, 3], "b1", "Harmonic Sine Amplitude", "NDVI")
    da_sigma = _wrap_da(sigma_flat, "sigma", "Residual Standard Deviation (RMSE)", "NDVI")
    da_amp = _wrap_da(amplitude_flat, "amplitude", "Seasonal Peak-to-Peak Amplitude", "NDVI")
    da_phase = _wrap_da(phase_flat, "phase", "Seasonal Peak Phase Angle", "radians")
    da_se_c0 = _wrap_da(se_flat[:, 0], "se_c0", "Standard Error of Intercept", "NDVI")
    da_se_c1 = _wrap_da(se_flat[:, 1], "se_c1", "Standard Error of Linear Trend", "NDVI/day")
    da_se_a1 = _wrap_da(se_flat[:, 2], "se_a1", "Standard Error of Cosine Amplitude", "NDVI")
    da_se_b1 = _wrap_da(se_flat[:, 3], "se_b1", "Standard Error of Sine Amplitude", "NDVI")
    da_r2 = _wrap_da(r2_flat, "r2", "Coefficient of Determination (R^2)")
    da_valid = _wrap_da(valid_counts, "valid_obs", "Count of Valid Observations", "count")

    logger.info(
        f"Fitted harmonic baseline across {p_pixels} pixels: {n_good} solved successfully "
        f"({p_pixels - n_good} underdetermined/masked)."
    )

    return HarmonicModel(
        c0=da_c0,
        c1=da_c1,
        a1=da_a1,
        b1=da_b1,
        sigma=da_sigma,
        amplitude=da_amp,
        phase=da_phase,
        reference_time=ref_ts,
        time_mode=time_mode,
        period=period,
        se_c0=da_se_c0,
        se_c1=da_se_c1,
        se_a1=da_se_a1,
        se_b1=da_se_b1,
        r2=da_r2,
        valid_obs=da_valid,
    )
