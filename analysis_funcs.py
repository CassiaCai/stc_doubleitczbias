# from spectral_analysis_functions import *
# from picontrol_functions import load_var_data, compute_fft, plot_power_spectrum
import cesm2_lens_utils
import data_access_functions as dac
import meridional_transport as calc_mer_transp
from meridional_transport import MeridionalVolumeTransport
# from picontrol_functions import *
# from lens_analysis_functions import *
import processing_utils as proc_utils
import xarray as xr

def atm_var_ens(ens_ind, VAR):
    COMP = 'atm'
    DIRECTORY = f'/glade/campaign/cgd/cesm/CESM2-LE/{COMP}/proc/tseries/month_1/{VAR}/'
    ds_var_hist_var, ds_var_fut_var = cesm2_lens_utils.get_ds_var(
        directory=DIRECTORY, var=VAR, comp=COMP, index_hist = ens_ind)
    var_ds = ds_var_hist_var[VAR].sel(time=slice('1958-01', '2015-01')).compute()
    return var_ds

def ocn_var_ens(ens_ind, VAR):
    COMP = 'ocn'
    DIRECTORY = f'/glade/campaign/cgd/cesm/CESM2-LE/{COMP}/proc/tseries/month_1/{VAR}/'
    ds_var_hist_var, ds_var_fut_var = cesm2_lens_utils.get_ds_var(
        directory=DIRECTORY, var=VAR, comp=COMP, index_hist = ens_ind)
    var_ds = ds_var_hist_var[VAR].sel(time=slice('1958-01', '2015-01'))
    return var_ds

def make_ds(data):
    data_ds = xr.DataArray(
        np.array(data), 
        dims=('time', 'nlat', 'nlon'), 
        coords={
            'time': data.time,
            'nlat': data.ULAT.mean(dim='nlon'), 
            'nlon': data.ULONG.mean(dim='nlat')})
    return data_ds

def standardize(data):
    return (data - data.mean()) / data.std()

def LENS_for_regridding():
    # LENS for regridding purposes
    ens_memb_index = 0
    comp = 'atm'; var = 'AREA'
    directory = f'/glade/campaign/cgd/cesm/CESM2-LE/{comp}/proc/tseries/month_1/{var}/'
    
    ds_var_hist_var, ds_var_fut_var = cesm2_lens_utils.get_ds_var(
        directory, var=var,comp=comp, 
        index_hist = ens_memb_index)
    
    # FOSI is from 1958 to 2020
    CESMLENS_hist_var = ds_var_hist_var[var].sel(time=slice('1958-01', '2015-01')).compute()
    CESMLENS_fut_var = ds_var_fut_var[var].sel(time=slice('2015-02', '2020-12')).compute()
    
    CESMLENS_var = xr.concat(
        [CESMLENS_hist_var, CESMLENS_fut_var], 
        dim='time')
    return CESMLENS_var

def calculate_DII(precip_ds):
    mean_Prec_N = precip_ds.sel(lat=slice(0,20), lon=slice(150, 270)).mean(dim=('lat','lon'))
    mean_Prec_S = precip_ds.sel(lat=slice(-20,0), lon=slice(150, 270)).mean(dim=('lat','lon'))
    mean_Prec_NS = precip_ds.sel(lat=slice(-20,20), lon=slice(150, 270)).mean(dim=('lat','lon'))
    DII = (mean_Prec_N - mean_Prec_S) / mean_Prec_NS
    return DII

def remove_climatology(ds):
    ds_climatology = ds.groupby("time.month").mean("time")
    ds_anomalies = ds.groupby("time.month") - ds_climatology
    return ds_anomalies

def load_PV_lens(ens_ind):
    PV_lens = ocn_var_ens(ens_ind, 'PV')
    PV_Lens_compute = PV_lens.isel(z_t = 20).compute()
    fosi_1deg_wzeros_var = proc_utils.regrid_SMYLE(PV_Lens_compute)
    regridded_PV_lens = regridder(fosi_1deg_wzeros_var)
    return regridded_PV_lens

def calculate_surface_flow(ens_ind, ds_var_hist_var):
    CONVERSION_FACTOR = (0.01 ** 3) / 1e6  # m^3/s to Sv 
    VVEL_ds = ocn_var_ens(ens_ind, 'VVEL')
    VVEL_cds = VVEL_ds.isel(
        z_t = slice(0,5)).sel(
        time=slice('1958-01', '2015-01'))[:,:,:,120:295].compute()
    dxu_in_cm_lat = ds_var_hist_var.DXU[:, nlat_ind_ls, 120:295].compute()
    dz_in_cm = ds_var_hist_var.dz.isel(z_t = slice(0,5)).compute()
    vvel_subselect = VVEL_cds[:,:,nlat_ind_ls, :]
    transport = vvel_subselect * dxu_in_cm_lat * dz_in_cm  # shape: time x z x lon
    integrated_transport = transport.sum(dim='z_t')  # shape: time x lon
    transport_Sv = integrated_transport * CONVERSION_FACTOR  # m³/s → Sv
    transport_Sv_nan = xr.where(transport_Sv == 0., np.nan, transport_Sv)
    transport_Sv_nan.to_netcdf(
        '/glade/derecho/scratch/cassiacai/lens_{}_surfaceflow.nc'.format(ens_ind))
    print('Done with',ens_ind)

