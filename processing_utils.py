import numpy as np
import xarray as xr
import xesmf as xe
from scipy.stats import linregress, pearsonr
from scipy import signal

def load_atm_fosi_variable(field, fpath, fosi_montime_vals):
    fname = f'g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005.pop.h.{field}.030601-036812.nc'
    ds_smyle_fosi = xr.open_dataset(fpath+fname)[field]
    ds_smyle_fosi['time'] = fosi_montime_vals
    return ds_smyle_fosi

def pop_find_lat_ind(loc, LATDAT):
    return np.abs(LATDAT[:, 0].values - loc).argmin()

def pop_find_lon_ind(loc, LONDAT, direction="w"):
    if direction.lower() in ["east", "e"]:
        value = loc
    elif direction.lower() in ["west", "w"]:
        value = 360 - loc
    else:
        print("I do not know which direction.")
    return np.nanargmin(np.abs(LONDAT[152, :].values - value))

def process_fosi_atm_var(field, fosi_montime_vals):
    fname = f'g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005.pop.h.{field}.030601-036812.nc'
    fpath = '/glade/campaign/cesm/development/espwg/SMYLE/initial_conditions/SMYLE-FOSI/ocn/proc/tseries/month_1/'
    ds_smyle_fosi_var = xr.open_dataset(fpath+fname)[field]
    ds_smyle_fosi_var['time'] = fosi_montime_vals
    var_fosi = ds_smyle_fosi_var.compute()
    fosi_1deg_wzeros_var = regrid_SMYLE(var_fosi)
    fosi_1deg_var = fosi_1deg_wzeros_var.where(
        fosi_1deg_wzeros_var!=0, np.nan)
    return fosi_1deg_var

def regrid_SMYLE(ds, glat=1, glon=1): # from Jacob's notebook
    """
    Inputs:
        ds: xr.DataArray with coordinates that include TLAT and TLONG
    Returns:
        Regridded xr.DataArray with coordinates lat and lon
    """
    ds = ds.rename(({'TLONG': 'lon', 'TLAT': 'lat'}))
    ds_out = xe.util.grid_global(glon, glat)
    regridder = xe.Regridder(ds, ds_out, 'bilinear', periodic=True)
    regridded = regridder(ds)
    new_coords = regridded.assign_coords({'y': regridded.lat[:, 0].values, 'x': regridded.lon[0].values})
    return new_coords.drop_vars(['lat', 'lon']).rename({'x': 'lon', 'y': 'lat'})

def calculate_detrended(timeseries):
    # Convert time to decimal years for linear regression
    time_vals = timeseries.time.dt.year + timeseries.time.dt.month/12
    
    # --- 1. Calculate Monthly Climatology ---
    climatology = timeseries.groupby("time.month").mean("time")
    noclimatology = timeseries.groupby("time.month") - climatology
    # --- 3a. Linear Trend ---
    slope, intercept = linregress(time_vals, noclimatology.values)[:2]
    trend_component = intercept + slope * time_vals

    # X(t) = Climatology(t) + Trend(t) + Anomalies(t)
    # Anomalies(t) = X(t) - Climatology(t) - Trend(t)
    detrended_linear = noclimatology - trend_component

    return detrended_linear

def normalize(da):
    min_val = da.min()
    max_val = da.max()
    normalized_da = (da - min_val) / (max_val - min_val)
    return normalized_da

def create_region_mask(lon, lat, vertices):
    """Pure NumPy implementation using ray casting."""
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    mask = np.zeros_like(lon_grid, dtype=bool)
    
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            mask[i,j] = point_in_polygon(
                vertices, 
                lon_grid[i,j], 
                lat_grid[i,j]
            )
    
    return xr.DataArray(
        mask,
        dims=('lat', 'lon'),
        coords={'lat': lat, 'lon': lon}
    )

def point_in_polygon(poly, x, y):
    """Ray casting algorithm for point-in-polygon test."""
    n = len(poly)
    inside = False
    px, py = x, y
    xints = 0.0
    
    for i in range(n):
        p1x, p1y = poly[i]
        p2x, p2y = poly[(i+1)%n]
        
        if min(p1y,p2y) < py <= max(p1y,p2y):
            if px <= max(p1x,p2x):
                if p1y != p2y:
                    xints = (py-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                if p1x == p2x or px <= xints:
                    inside = not inside
    return inside

def lowpass_filter_xarray(da, cutoff_period_years, time_freq='monthly'):
    """
    Apply Butterworth low-pass filter to retain variability with periods LONGER than cutoff.
    
    Parameters:
    da: xarray DataArray with time dimension
    cutoff_period_years: cutoff period in years (e.g., 5 for 5-year and longer variability)
    time_freq: frequency of time data ('monthly', 'daily', etc.)
    """
    # Get sampling frequency
    if time_freq == 'monthly':
        sampling_freq = 12  # samples per year
    elif time_freq == 'daily':
        sampling_freq = 365.25
    else:
        sampling_freq = 1  # yearly data
    
    # Calculate cutoff frequency in correct units
    # For periods LONGER than 5 years, we want to KEEP frequencies LOWER than 1/5 cycles per year
    cutoff_freq_cycles_per_year = 1.0 / cutoff_period_years
    
    # Convert to normalized frequency (0 to 0.5 for scipy)
    nyquist_freq = 0.5 * sampling_freq
    Wn = cutoff_freq_cycles_per_year / nyquist_freq
    
    # Design 4th order Butterworth low-pass filter
    b, a = signal.butter(4, Wn, btype='low', analog=False)
    
    # Apply zero-phase filtering (filtfilt) to avoid phase shift
    filtered_data = signal.filtfilt(b, a, da.values, axis=da.get_axis_num('time'))
    
    return xr.DataArray(
        filtered_data,
        dims=da.dims,
        coords=da.coords,
        attrs={**da.attrs, **{'filtering': f'Butterworth low-pass, cutoff: {cutoff_period_years} years'}}
    )

def highpass_filter_xarray(da, cutoff_period_years, time_freq='monthly'):
    """Remove variability longer than cutoff period (keep shorter periods)."""
    # Same frequency calculation as above
    if time_freq == 'monthly':
        sampling_freq = 12
    elif time_freq == 'daily':
        sampling_freq = 365.25
    else:
        sampling_freq = 1
    
    cutoff_freq_cycles_per_year = 1.0 / cutoff_period_years
    nyquist_freq = 0.5 * sampling_freq
    Wn = cutoff_freq_cycles_per_year / nyquist_freq
    
    # Design high-pass filter
    b, a = signal.butter(4, Wn, btype='high', analog=False)
    
    filtered_data = signal.filtfilt(b, a, da.values, axis=da.get_axis_num('time'))
    
    return xr.DataArray(
        filtered_data,
        dims=da.dims,
        coords=da.coords,
        attrs={**da.attrs, **{'filtering': f'Butterworth high-pass, cutoff: {cutoff_period_years} years'}}
    )
