import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections
import matplotlib.colors as colors
import numpy as np
import cmocean
import processing_utils as proc_utils
import cesm2_lens_utils
import analysis_funcs as afuncs
import xesmf as xe
import pop_tools
import cftime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import gaussian_kde

def calc_density_eqthermocline(lon_data, lat_data, z_data, lat_min, lat_max, lon_min, lon_max, timestart, timeend):
    lon_trunc,lat_trunc, z_trunc = lon_data, lat_data, z_data
    region_idx = get_region_lonlat(lat_trunc, lon_trunc, lat_min, lat_max, lon_min, lon_max)    
    lons_points = lon_trunc[region_idx, timestart:timeend].data.flatten()
    lats_points = lat_trunc[region_idx, timestart:timeend].data.flatten()
    f_xi, f_yi, f_zi = calc_density_sh(lons_points, lats_points)
    da = xr.DataArray(f_zi, dims=['y', 'x'],
        coords={'lon': (['y', 'x'], f_xi),'lat': (['y', 'x'], f_yi)},name='density')
    return da, region_idx, lon_trunc,lat_trunc, z_trunc

def compute_data(data_xarray):
    time = data_xarray.time.compute()
    lon = data_xarray.lon.compute()
    lat = data_xarray.lat.compute()
    z = data_xarray.z.compute()
    return time, lon, lat, z

def ocn_var_ens(ens_ind, VAR):
    COMP = 'ocn'
    DIRECTORY = f'/glade/campaign/cgd/cesm/CESM2-LE/{COMP}/proc/tseries/month_1/{VAR}/'
    ds_var_hist_var, ds_var_fut_var = cesm2_lens_utils.get_ds_var(
        directory=DIRECTORY, var=VAR, comp=COMP, index_hist = ens_ind)
    var_ds = ds_var_hist_var[VAR].sel(time=slice('1958-01', '2015-01'))
    return var_ds

def atm_var_ens(ens_ind, VAR):
    COMP = 'atm'
    DIRECTORY = f'/glade/campaign/cgd/cesm/CESM2-LE/{COMP}/proc/tseries/month_1/{VAR}/'
    ds_var_hist_var, ds_var_fut_var = cesm2_lens_utils.get_ds_var(
        directory=DIRECTORY, var=VAR, comp=COMP, index_hist = ens_ind)
    var_ds = ds_var_hist_var[VAR].sel(time=slice('1958-01', '2015-01')).compute()
    return var_ds

def plot_parcel_trajectories_w_bg(cf_data, 
                                 lon_data, lat_data, z_data,
                                 indices=None,
                                 index_min=0, index_max=300,
                                 vmin=10, vmax=30, levels=15, 
                                 cmap=cmocean.cm.thermal,
                                 line_cmap='hsv',
                                 start_color='cyan', end_color='m',
                                 xlim=(110, 290), ylim=(-60, 60),
                                 title='Parcel Trajectories',
                                 figsize=(10, 6),
                                 linewidth=1,
                                 start_size=1, end_size=1):
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot background
    cf = cf_data.plot.contourf(
        levels=levels, vmax=vmax, cmap=cmap, vmin=vmin, alpha=1, ax=ax
    )
    
    # Determine which indices to plot
    if indices is None:
        plot_indices = range(index_min, index_max)
    else:
        plot_indices = indices
    
    # Plot start and end points
    ax.scatter(lon_data[plot_indices, 0], lat_data[plot_indices, 0], 
               s=start_size, c=start_color, zorder=100, label='Start')
    ax.scatter(lon_data[plot_indices, -1], lat_data[plot_indices, -1], 
               s=end_size, c=end_color, zorder=100, label='End')
        
    # Plot trajectories
    for i in plot_indices:
        lon = lon_data.isel(trajectory=i).values
        lat = lat_data.isel(trajectory=i).values
        depth = z_data.isel(trajectory=i).values
        
        points = np.array([lon, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = mcollections.LineCollection(
            segments, cmap=line_cmap, norm=plt.Normalize(depth.min(), depth.max()), linewidth=linewidth
        )
        lc.set_array(depth[:-1])
        ax.add_collection(lc)
    
    # Add reference lines
    ax.axhline(y=5, c='k', linestyle='dotted', alpha=0.7)
    ax.axhline(y=0, c='k', linestyle='dashed', alpha=0.7)
    ax.axhline(y=-5, c='k', linestyle='dotted', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(c='k', linestyle='dashed', alpha=0.2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=15)
    
    # Add colorbar for trajectories
    sm = plt.cm.ScalarMappable(cmap=line_cmap, norm=plt.Normalize(depth.min(), depth.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Depth')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    return fig, ax

def truncate_lonlatz(lon_data, lat_data, z_data):
    lon_data_truncated = lon_data.copy()
    lat_data_truncated = lat_data.copy()
    z_data_truncated = z_data.copy()
    
    # lon_min, lon_max = 260, 291
    lon_min, lon_max = 260, 280#290, 291 #260, 280
    lat_min, lat_max = -1, 1
    
    for i in range(lon_data.sizes['trajectory']):
        lon_traj = lon_data.isel(trajectory=i).values
        lat_traj = lat_data.isel(trajectory=i).values
        
        in_target_region = (lon_traj >= lon_min) & (lon_traj <= lon_max) & (lat_traj >= lat_min) & (lat_traj <= lat_max)
        
        # Find first occurrence in target region
        region_indices = np.where(in_target_region)[0]
        
        if len(region_indices) > 0:
            first_in_region = region_indices[0]
            # Set all observations after entering region to NaN
            lon_data_truncated[i, first_in_region+1:] = np.nan
            lat_data_truncated[i, first_in_region+1:] = np.nan
            z_data_truncated[i, first_in_region+1:] = np.nan

    return lon_data_truncated, lat_data_truncated, z_data_truncated

# ###################
# lon_data_truncated = lon_data.copy()
# lat_data_truncated = lat_data.copy()
# z_data_truncated = z_data.copy()

# lon_min, lon_max = 240, 280
# lat_min, lat_max = -1, 1

# for i in range(lsh_lon.sizes['trajectory']):
#     lon_traj = lsh_lon.isel(trajectory=i).values
#     lat_traj = lsh_lat.isel(trajectory=i).values

#     in_target_region = (lon_traj >= lon_min) & (lon_traj <= lon_max) & (lat_traj >= lat_min) & (lat_traj <= lat_max)

#     # Find first occurrence in target region
#     region_indices = np.where(in_target_region)[0]

#     if len(region_indices) > 0:
#         first_in_region = region_indices[0]
#         # Set all observations after entering region to NaN
#         lon_data_truncated[i, first_in_region+1:] = np.nan
#         lat_data_truncated[i, first_in_region+1:] = np.nan
#         z_data_truncated[i, first_in_region+1:] = np.nan

# original working
def truncate_after_exit_equator(lon_data, lat_data, z_data):
    lon_data_truncated = lon_data.copy()
    lat_data_truncated = lat_data.copy()
    z_data_truncated = z_data.copy()
    
    # Define equatorial region boundaries
    lat_min, lat_max = -0.5, 0.5     # 2°S to 2°N
    lon_min, lon_max = 140, 305   # 200° to 280° longitude
    
    for i in range(lon_data.sizes['trajectory']):
        lon_traj = lon_data.isel(trajectory=i).values
        lat_traj = lat_data.isel(trajectory=i).values
        
        # Find where trajectory is within equatorial region (both lat and lon constraints)
        in_equator = ((lat_traj >= lat_min) & (lat_traj <= lat_max) & 
                      (lon_traj >= lon_min) & (lon_traj <= lon_max))
        
        # Find the last time the trajectory is in the equatorial region
        equator_indices = np.where(in_equator)[0]
        
        if len(equator_indices) > 0:
            last_in_equator = equator_indices[-1]
            
            # Find the first point AFTER last_in_equator that exits the equatorial region
            exit_found = False
            for j in range(last_in_equator + 1, len(lat_traj)):
                # Check if point is outside equatorial region (either lat or lon constraint violated)
                if (lat_traj[j] < lat_min or lat_traj[j] > lat_max or 
                    lon_traj[j] < lon_min or lon_traj[j] > lon_max):
                    # Truncate from this exit point onward
                    lon_data_truncated[i, j:] = np.nan
                    lat_data_truncated[i, j:] = np.nan
                    z_data_truncated[i, j:] = np.nan
                    exit_found = True
                    break
            
            # If trajectory ends while still in equatorial region, no truncation needed
            # (no action needed in this case)

    return lon_data_truncated, lat_data_truncated, z_data_truncated

def truncate_after_first_exit_equator(lon_data, lat_data, z_data):
    lon_data_truncated = lon_data.copy()
    lat_data_truncated = lat_data.copy()
    z_data_truncated = z_data.copy()
    
    # Define equatorial region boundaries
    lat_min, lat_max = -1, 1     # 2°S to 2°N
    lon_min, lon_max = 140, 305   # 200° to 280° longitude
    
    for i in range(lon_data.sizes['trajectory']):
        lon_traj = lon_data.isel(trajectory=i).values
        lat_traj = lat_data.isel(trajectory=i).values
        
        # Track if we've entered the equatorial region at least once
        has_entered_equator = False
        first_exit_index = None
        
        for j in range(len(lat_traj)):
            # Check if current point is in equatorial region
            in_equator = ((lat_traj[j] >= lat_min) & (lat_traj[j] <= lat_max) & 
                          (lon_traj[j] >= lon_min) & (lon_traj[j] <= lon_max))
            
            if in_equator:
                has_entered_equator = True
            elif has_entered_equator and first_exit_index is None:
                # We've entered equator before and now we're exiting for the first time
                first_exit_index = j
                break
        
        # Truncate after first exit point
        if first_exit_index is not None:
            lon_data_truncated[i, first_exit_index:] = np.nan
            lat_data_truncated[i, first_exit_index:] = np.nan
            z_data_truncated[i, first_exit_index:] = np.nan
       #     print(f"Trajectory {i}: first exit at index {first_exit_index}")
       # elif has_entered_equator:
      #      print(f"Trajectory {i}: entered equator but never exited")
      #  else:
      #      print(f"Trajectory {i}: never entered equator")

    return lon_data_truncated, lat_data_truncated, z_data_truncated

def get_region_lonlat(lat_data, lon_data, region_lat_min, region_lat_max, region_lon_min, region_lon_max):
    index_min = 0
    index_max = 400
    
    in_region_mask = (
        (lat_data[index_min:index_max, 0] >= region_lat_min) & 
        (lat_data[index_min:index_max, 0] <= region_lat_max) &
        (lon_data[index_min:index_max, 0] >= region_lon_min) & 
        (lon_data[index_min:index_max, 0] <= region_lon_max)
    )
    
    region_indices = np.where(in_region_mask)[0] + index_min
    return region_indices

def calc_density_sh(lons_points, lats_points):
    
    lon_trunc,lat_trunc, z_trunc = truncate_lonlatz(
        lon_data, lat_data, z_data)

    region_idx_p1 = get_region_lonlat(lat_trunc, lon_trunc, -30, -20, 240, 270)
    region_idx_p2 = get_region_lonlat(lat_trunc, lon_trunc, -50, -30, 180, 270)
    
    lons_points = lon_trunc[np.append(region_idx_p1, region_idx_p2), :].data.flatten()
    lats_points = lat_trunc[np.append(region_idx_p1, region_idx_p2), :].data.flatten()
    
    non_nan_mask = ~np.isnan(lons_points)
    cleaned_lats_points = lats_points[non_nan_mask]
    cleaned_lons_points = lons_points[non_nan_mask]
    
    x = cleaned_lons_points
    y = cleaned_lats_points
    k = gaussian_kde(np.vstack([x, y]))

    num_bins_x = 200
    num_bins_y = 200
    
    xi = np.linspace(x.min(), x.max(), num_bins_x)
    yi = np.linspace(y.min(), y.max(), num_bins_y)
    xi, yi = np.meshgrid(xi, yi)
    
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi.reshape(xi.shape)
    return xi, yi, zi

def calc_density(lon_data, lat_data, z_data, lat_min, lat_max, lon_min, lon_max, timestart, timeend):
    
    lon_trunc,lat_trunc, z_trunc = truncate_after_first_exit_equator(#truncate_after_exit_equator(#truncate_lonlatz(
        lon_data, lat_data, z_data)

    region_idx = get_region_lonlat(lat_trunc, lon_trunc, lat_min, lat_max, lon_min, lon_max)
    # region_idx_p1 = get_region_lonlat(lat_trunc, lon_trunc, -35, -30, 220, 240)
    # region_idx_p2 = get_region_lonlat(lat_trunc, lon_trunc, -40, -35, 220,240)
    
    lons_points = lon_trunc[region_idx, timestart:timeend].data.flatten()
    lats_points = lat_trunc[region_idx, timestart:timeend].data.flatten()
    
    f_xi, f_yi, f_zi = calc_density_sh(lons_points, lats_points)
    
    da = xr.DataArray(f_zi, dims=['y', 'x'],
        coords={'lon': (['y', 'x'], f_xi),'lat': (['y', 'x'], f_yi)},name='density')
    
    # da_nan = xr.where(da <= 0.000001, np.nan, da)
    # return da_nan, region_idx, lon_trunc,lat_trunc, z_trunc
    return da, region_idx, lon_trunc,lat_trunc, z_trunc

def calc_density_sh(lons, lats, gridsize=100, bw_method=None, lon_bounds=(110, 305), lat_bounds=(-50, 50)):
    """
    Calculate 2D Gaussian KDE density with fixed grid bounds.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(lons) & ~np.isnan(lats)
    lons_clean = lons[valid_mask]
    lats_clean = lats[valid_mask]
    
    if len(lons_clean) == 0:
        raise ValueError("No valid data points for KDE calculation")
    
    # Use fixed bounds instead of data-derived bounds
    xi = np.linspace(lon_bounds[0], lon_bounds[1], gridsize)
    yi = np.linspace(lat_bounds[0], lat_bounds[1], gridsize)
    f_xi, f_yi = np.meshgrid(xi, yi)
    
    # Calculate Gaussian KDE
    coordinates = np.vstack([lons_clean, lats_clean])
    kde = gaussian_kde(coordinates, bw_method=bw_method)
    
    # Evaluate KDE on the fixed grid
    grid_coords = np.vstack([f_xi.ravel(), f_yi.ravel()])
    f_zi = kde(grid_coords).reshape(f_xi.shape)
    
    return f_xi, f_yi, f_zi

# def calc_density(lon_data, lat_data, z_data, lat_min, lat_max, lon_min, lon_max, timestart, timeend):

#     lon_trunc, lat_trunc, z_trunc = truncate_lonlatz(lon_data, lat_data, z_data)

# region _idx = get_region_lonlat(lat_trunc, lon_trunc, lat_min, lat_max, lon_min, lon_max)
    
#     lons_points = lon_trunc[region_idx, timestart:timeend].data.flatten()
#     lats_points = lat_trunc[region_idx, timestart:timeend].data.flatten()
    
#     # Use fixed grid bounds that match your plot extent (140-284°E, -40-40°N)
#     f_xi, f_yi, f_zi = calc_density_sh(lons_points, lats_points, 
#                                       lon_bounds=(110, 305), 
#                                       lat_bounds=(-50, 50))
    
#     da = xr.DataArray(f_zi, dims=['y', 'x'],
#         coords={'lon': (['y', 'x'], f_xi), 'lat': (['y', 'x'], f_yi)}, name='density')
    
#     da_nan = xr.where(da <= 0.000001, np.nan, da)
#     return da_nan, region_idx

def find_sigma_theta_points(lsh_lon, lsh_lat, sigma_theta_pd, num_points=400, sigma_min=24.4, sigma_max=24.8):
    """
    Find indices of points within specified sigma theta range.
    
    Parameters:
    -----------
    lsh_lon, lsh_lat : array
        Longitude and latitude arrays
    sigma_theta_pd : xarray.DataArray
        Sigma theta data
    num_points : int
        Number of points to check
    sigma_min, sigma_max : float
        Sigma theta range to filter
    
    Returns:
    --------
    yellow_indices : list
        Indices of points within the sigma theta range
    """
    yellow_indices = []
    
    for i in range(num_points):
        lon = lsh_lon[i, 0]
        lat = lsh_lat[i, 0]
        
        try:
            sigma_val = sigma_theta_pd.sel(lon=lon, lat=lat, method='nearest').values
        except:
            try:
                sigma_val = sigma_theta_pd.interp(lon=lon, lat=lat).values
            except:
                lon_idx = np.argmin(np.abs(sigma_theta_pd.lon.values - lon))
                lat_idx = np.argmin(np.abs(sigma_theta_pd.lat.values - lat))
                sigma_val = sigma_theta_pd.values[lat_idx, lon_idx]
        
        if sigma_min <= sigma_val <= sigma_max:
            yellow_indices.append(i)
    
    return yellow_indices

def spatial_lens_SH_map_w_traj(lon_data, lat_data, z_data, region_idx, da_selected, ENS_MEMB, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -50, 10], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### TRAJECTORIES
    for i in region_idx[:]:
        lon = lon_data.isel(trajectory=i).values
        lat = lat_data.isel(trajectory=i).values
        depth = z_data.isel(trajectory=i).values
        points = np.array([lon, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcollections.LineCollection(
                segments, cmap='jet', norm=plt.Normalize(0, 25000), 
                linewidth=1., alpha=1, transform=ccrs.PlateCarree())
        lc.set_array(depth[:-1])
        ax.add_collection(lc)
    
    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=80.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 80.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 80, 20)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('{} LENS begin at {}m'.format(ENS_MEMB, INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()

def spatial_lens_NH_map_w_traj(lon_data, lat_data, z_data, region_idx, da_selected, ENS_MEMB, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -10, 50], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### TRAJECTORIES
    for i in region_idx[::2]:
        lon = lon_data.isel(trajectory=i).values
        lat = lat_data.isel(trajectory=i).values
        depth = z_data.isel(trajectory=i).values
        points = np.array([lon, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcollections.LineCollection(
                segments, cmap='jet', norm=plt.Normalize(0, 25000), 
                linewidth=1., alpha=1, transform=ccrs.PlateCarree())
        lc.set_array(depth[:-1])
        ax.add_collection(lc)
    
    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=40.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 40.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 50, 10)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('{} LENS begin at {}m'.format(ENS_MEMB, INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()

def spatial_lens_SH_map_no_traj(lon_data, lat_data, z_data, region_idx, da_selected, ENS_MEMB, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -50, 10], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=80.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 80.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 80, 20)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('{} LENS begin at {}m'.format(ENS_MEMB, INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()

def spatial_lens_NH_map_no_traj(lon_data, lat_data, z_data, region_idx, da_selected, ENS_MEMB, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -10, 50], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=40.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 40.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 50, 10)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('{} LENS begin at {}m'.format(ENS_MEMB, INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()

def spatial_fosi_SH_map_no_traj(lon_data, lat_data, z_data, region_idx, da_selected, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -50, 10], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=80.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 80.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 80, 20)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('FOSI begin at {}m'.format(INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()

def spatial_fosi_NH_map_no_traj(lon_data, lat_data, z_data, region_idx, da_selected, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -10, 50], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=40.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 40.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 50, 10)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('FOSI begin at {}m'.format(INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()

def spatial_fosi_SH_map_w_traj(lon_data, lat_data, z_data, region_idx, da_selected, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -50, 10], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### TRAJECTORIES
    for i in region_idx[:]:
        lon = lon_data.isel(trajectory=i).values
        lat = lat_data.isel(trajectory=i).values
        depth = z_data.isel(trajectory=i).values
        points = np.array([lon, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcollections.LineCollection(
                segments, cmap='jet', norm=plt.Normalize(0, 25000), 
                linewidth=1., alpha=1, transform=ccrs.PlateCarree())
        lc.set_array(depth[:-1])
        ax.add_collection(lc)
    
    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=80.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 80.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 80, 20)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('FOSI begin at {}m'.format(INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()

def spatial_fosi_NH_map_w_traj(lon_data, lat_data, z_data, region_idx, da_selected, INIT_DEPTH):
    fig, ax = plt.subplots(figsize=(5, 4),
                      subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180, globe=None)})
    ####### STATICS
    ax.add_feature(cfeature.LAND, color='lightgray', zorder=100)
    ax.add_feature(cfeature.COASTLINE, linewidth=1., zorder=100)
    ax.grid(c='k', linestyle='dashed', alpha=0.2, zorder=4)
    ax.set_extent([120, 284, -10, 50], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels={'left': True, 'bottom': True, 'right': False, 'top': False}, 
                      zorder=4, linestyle='--', alpha=0.5)
    gl.xlabel_style = {'size': 12}  # Longitude labels
    gl.ylabel_style = {'size': 12}  # Latitude labels
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1, zorder=5)
    ax.spines['geo'].set_edgecolor('black')
    ax.spines['geo'].set_linewidth(1.5)
    ax.set_aspect('auto')
    
    ax.scatter(lon_data[region_idx, 0], lat_data[region_idx, 0], 
               s=0.5, c='m', marker='o', edgecolor='m', zorder=100,transform=ccrs.PlateCarree())

    #### TRAJECTORIES
    for i in region_idx[::2]:
        lon = lon_data.isel(trajectory=i).values
        lat = lat_data.isel(trajectory=i).values
        depth = z_data.isel(trajectory=i).values
        points = np.array([lon, lat]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = mcollections.LineCollection(
                segments, cmap='jet', norm=plt.Normalize(0, 25000), 
                linewidth=1., alpha=1, transform=ccrs.PlateCarree())
        lc.set_array(depth[:-1])
        ax.add_collection(lc)
    
    #### CONTOURF AND CONTOUR
    contourf_plot = da_selected.plot.contourf(
        x='lon', y='lat', cmap='Blues', add_colorbar=False, transform=ccrs.PlateCarree(),
        levels=21, vmax=40.5e-5, vmin=0.5e-5, alpha=1)
    
    da_selected.plot.contour(
        x='lon', y='lat', transform=ccrs.PlateCarree(), colors='k', linewidths=0.5,
        levels=[2e-5, 5e-5, 10e-5, 15e-5, 20e-5, 25e-5], alpha=1)
    
    #### COLORBAR
    density_sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0.5e-5, 40.5e-5))
    density_sm.set_array([])
    density_cbar = plt.colorbar(density_sm, ax=ax, aspect=30, shrink=0.8, pad=0.1, orientation='horizontal')
    density_cbar.set_label('Normalized Trajectory Density', fontsize=11)
    tick_values = np.arange(0, 50, 10)  # This gives 5, 10, 15, 20, 25, 30
    density_cbar.set_ticks(tick_values * 1e-5)
    density_cbar.set_ticklabels([f'{x}' for x in tick_values])  # Shows as 5, 10, 15, etc.
    density_cbar.ax.tick_params(labelsize=12)
    
    #### TITLE
    plt.title('FOSI begin at {}m'.format(INIT_DEPTH), fontsize=12, fontweight='bold', zorder=12, loc='left')
    plt.show()
    
# endregion
