import xarray as xr
import pandas as pd
import numpy as np
import salem
#import geopandas as gpd
from scipy import stats

#### ---------------------- ####
#### Load functions         ####
#### ---------------------- ####

## Currently not implemented ##
# Utils to download files ##
# Utils to preprocess  ##
# Working script build on file download from https://esgf-metagrid.cloud.dkrz.de/search - allows merging of all pr files into a singular .nc file #
## ------------------------- ##

def _utils_bounds_shp(refds, shp, load_files=True):
    if load_files == True:
        ref = salem.open_metum_dataset(refds)
        #shp_file = gpd.read_file(shp)
        shp_file = salem.read_shapefile(shp)
    else:
        ref = refds.copy()
        shp_file = shp.copy()
    #these utilities can prob. be done only with salem as well
    
    reproj_shp = shp_file.to_crs(ref.pyproj_srs)
    #'+ellps=WGS84 +proj=ob_tran +o_proj=latlon +to_meter=0.0174532925199433\
        # +o_lon_p=0.0 +o_lat_p=43.0 +lon_0=10.0 +no_defs'

    reproj_shp.crs = ref.pyproj_srs
    bounds = reproj_shp.bounds
    centroid = reproj_shp.dissolve().centroid
    
    return ref, reproj_shp, bounds, centroid

def _utils_subset_ds(ds_path, shp_path, box_size, load_files=True, option="centroid"):
    
    ds, reproj_shp, bounds, centroid = _utils_bounds_shp(ds_path, shp_path, load_files=load_files)

    if option == "bounds":
        #select within bounds
        subset = ds.sel(rlat=slice(bounds.miny.item(), bounds.maxy.item()), rlon=slice(bounds.minx.item(), bounds.maxx.item()))
    elif option == "buffer":
        #drlon = dsr.rlon.diff(dim='rlon').mean()
        #drlat = dsr.rlat.diff(dim='rlat').mean()

        idx_lat = np.argmin(np.abs(ds.rlat.values - centroid.y.values))
        idx_lon = np.argmin(np.abs(ds.rlon.values - centroid.x.values))
        
        # Calculate bounds of buffer
        min_lat = max(idx_lat - box_size, 0)
        max_lat = min(idx_lat + box_size, ds.rlat.size - 1)
        min_lon = max(idx_lon - box_size, 0)
        max_lon = min(idx_lon + box_size, ds.rlon.size - 1)

        # Assert within limits
        assert min_lat <= idx_lat <= max_lat, f"Latitude selection exceeds bounds: {min_lat} <= {idx_lat} <= {max_lat}"
        assert min_lon <= idx_lon <= max_lon, f"Longitude selection exceeds bounds: {min_lon} <= {idx_lon} <= {max_lon}"

        # Select the subset of the dataset based on the adjusted box size
        subset = ds.isel(rlat=slice(min_lat, max_lat + 1),
                            rlon=slice(min_lon, max_lon + 1))
    elif option == "centroid":
        idx_lat = np.argmin(np.abs(ds.rlat.values - centroid.y.values))
        idx_lon = np.argmin(np.abs(ds.rlon.values - centroid.x.values))
        
        subset = ds.isel(rlat=idx_lat,
                         rlon=idx_lon)
    return subset
    

def _utils_correct_field_units(ds, fieldname, dt=1, checkonly=False):
    """_summary_
    Args:
        ds (xr.Dataset): xr. Dataset that holds the COSMO field to correct
        dt (int): Integer temporal resolution of ds in hours (1=hourly, 24=daily)
        fieldname (str): Variable name to lookup proper unit conversion
    """ 
    
    lookup_sanity = {'pr': (0, 100),
                    'ps': (200.0, 1080.0),
                    'hurs': (0.0, 100.0),
                    'uas': (0.0, 50.0),
                    'vas': (0.0, 50.0),
                    'U2': (0.0, 50.0),
                    'rsds': (0.0, 1600.0),
                    'clt': (0.0, 1.0),
                    'SNOWFALL': (0.0, 0.1),
                    'tas': (223.16, 316.16),
                    'prsn': (0.0, 100)}
    if checkonly == True:
        if np.nanmax(ds[fieldname]) > lookup_sanity[fieldname][1] or np.nanmin(ds[fieldname]) < lookup_sanity[fieldname][0]:
            print("WARNING! Please check your data, it seems to be outside a reasonable range.")
            print(f"Field {fieldname} max: {np.nanmax(ds[fieldname])}, min: {np.nanmin(ds[fieldname])}.")
        else:
            print(f"Checked {fieldname} without any flags.")
    else:
        lookup_conv = {'pr': 3600*dt, #pr from kg/m2/s to kg/m2/h or kg/m2/d
                    'prsn': 3600*dt,
                    'ps': 0.01, #Pa to hPa
                    'clt': 0.01, #perc. conv.
                    } #inc. SW seems to be correct (W/m2)
                        # did not have inc. LW,
                        # wind speeds in m/s
                        # SNOWFALL needs to be in meter if provided
                        # T2 in K 
        #https://cosipy.readthedocs.io/en/latest/resources.html#input-data
        print(f"Adjusting field {fieldname} by factor {lookup_conv[fieldname]}.")
        ds[fieldname+"_fix"] = ds[fieldname] * lookup_conv[fieldname]
        print(f"Corrected field {fieldname}.")
        if np.nanmax(ds[fieldname+"_fix"]) > lookup_sanity[fieldname][1] or np.nanmin(ds[fieldname+"_fix"]) < lookup_sanity[fieldname][0]:
            print("WARNING! Please check your data, it seems to be outside a reasonable range.")
        else:
            print(f"Checked {fieldname} without any flags.")
                              
        return ds
    
def _utils_calc_u2(ds_v, ds_u, z0=2.12/1000, z=10):
    """Script to calculate the wind speed at 2m from zonal and meridional components, 
        assuming a logarithmic wind profile.

    Args:
        ds_v (xr.Dataset): v-component of the horizontal wind speed
        ds_u (xr.Dataset): u-component of the horizontal wind speed
        z0 (float, optional): Roughness length of the surface in m. Defaults to 2.12/1000, based on mean of firn and snow
        Brock et al., 2006: 10.3189/172756506781828746, Gromke et al., 2011: https://doi.org/10.1007/s10546-011-9623-3
        z (int, optional): Measurement height in m. Defaults to 10.
    """
    ds_u['U10'] = np.sqrt( (ds_u['uas']**2) + (ds_v['vas']**2) )
    z_target = 2
    ds_u['U2'] = ds_u['U10'] * ((np.log(z_target/z0) / np.log(z/z0)))
    
    return ds_u['U2']

def _utils_intp_pres(ds_ps):
    return ds_ps.resample(time="1H").interpolate("linear")
    
def _utils_derive_snowfall(ds_t2, ds_u, ds_sf):
    #COSMO has SNOWFALL flux in SWE by default - COSIPY requires [m]
    #Follow COSIPYs parameterisation of fresh snow density
    density_fresh_snow = np.maximum(109.0+6.0*(ds_t2['tas']-273.16)+26.0*np.sqrt(ds_u['U2']), 50.0)
    ds_sf['SNOW'] = ds_sf['prsn'] / density_fresh_snow
    return ds_sf['SNOW']

def _utils_set_df_bounds(df):
    #df.loc[df['SNOWFALL'] < 0, 'SNOWFALL'] = 0
    #df.loc[df['RAIN'] < 0, 'RAIN'] = 0
    df.loc[df['RRR'] < 0, 'RRR'] = 0
    df.loc[df['G'] < 0, 'G'] = 0
    df.loc[df['RH2'] > 100, 'RH2'] = 100
    df.loc[df['RH2'] < 0, 'RH2'] = 0
    return df

                
## If snowfall not present, use parametrisation in aws2cosipy!

#### Only required for distributed runs and lapse rates ####

#get linear regr slope
def _utils_lin_regr(x,y):
    #fit = np.polyfit(x,y,1)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    #r_value ** 2 in case of simple linear regression is equal to coefficient of determination
    return xr.DataArray(slope)

#separate function for ufunc 
def _utils_lin_regr_r2(x,y):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return xr.DataArray(r_value**2)

def create_forcing_with_lr(ds, centroid):
    """Take your bounding box dataset and calculate lapse rates using linear regressions.
       Crops the bounding box to the centroid, which will be used in aws2cosipy utilities.
       Requires adjustment to the aws2cosipy code, to allow for the online lapse rate.

    Args:
        ds (xr.Dataset): xr.Dataset that holds all variables and the neighbouring grid cells.
        centroid (pd.DataFrame): Dataframe that holds the coordinates of the centroid to output.

    Returns:
        xr.Dataset: Cropped dataset to centroid including lapse rate fields
    """
        
    #calculate lapse rates
    hsurf2 = ds.HGT.values
    hsurf2_new = np.expand_dims(hsurf2, axis=0)
    hsurf2_new = np.repeat(hsurf2_new, len(ds.time.values), axis=0)
    ds['HGT2'] = (('time','rlat','rlon'), hsurf2_new)

    # get r2 as well and only apply lapse rate where clear relationship visible?

    #combine dim into one
    stack_ds = ds.stack(latlon=['rlat','rlon'])
    
    # calculate lapse rates over time
    lr_t2m = xr.apply_ufunc(_utils_lin_regr, stack_ds.HGT2, stack_ds.T2,dask='parallelized',
                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    lr_rh2 = xr.apply_ufunc(_utils_lin_regr, stack_ds.HGT2, stack_ds.RH2,dask='parallelized',
                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    #lr_lwin = xr.apply_ufunc(lin_regr, stack_ds.HGT2, stack_ds.ATHB_S ,dask='parallelized',
    #                         vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    lr_clt = xr.apply_ufunc(_utils_lin_regr, stack_ds.HGT2, stack_ds.N ,dask='parallelized',
                             vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    lr_tp = xr.apply_ufunc(_utils_lin_regr, stack_ds.HGT2, stack_ds.RRR, dask='parallelized',
                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    #lr_rain = xr.apply_ufunc(lin_regr, stack_ds.HGT2, stack_ds.RAIN, dask='parallelized',
    #                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    #lr_sf = xr.apply_ufunc(_utils_lin_regr, stack_ds.HGT2, stack_ds.SNOWFALL, dask='parallelized',
    #                    vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    
    # calculate r2 values over time
    
    lr_t2m_r2 = xr.apply_ufunc(_utils_lin_regr_r2, stack_ds.HGT2, stack_ds.T2,dask='parallelized',
                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    lr_rh2_r2 = xr.apply_ufunc(_utils_lin_regr_r2, stack_ds.HGT2, stack_ds.RH2,dask='parallelized',
                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    #lr_lwin = xr.apply_ufunc(_utils_lin_regr_r2, stack_ds.HGT2, stack_ds.ATHB_S ,dask='parallelized',
    #                         vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    lr_clt_r2 = xr.apply_ufunc(_utils_lin_regr_r2, stack_ds.HGT2, stack_ds.N ,dask='parallelized',
                            vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    lr_tp_r2 = xr.apply_ufunc(_utils_lin_regr_r2, stack_ds.HGT2, stack_ds.RRR, dask='parallelized',
                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    #lr_rain_r2 = xr.apply_ufunc(_utils_lin_regr_r2, stack_ds.HGT2, stack_ds.RAIN, dask='parallelized',
    #                        vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    #lr_sf_r2 = xr.apply_ufunc(_utils_lin_regr_r2, stack_ds.HGT2, stack_ds.SNOWFALL, dask='parallelized',
    #                    vectorize=True, input_core_dims=[['latlon'], ['latlon']])
    
    # Create single file at centroid from which to distribute using lapse rates 
    idx_lat = np.argmin(np.abs(ds.rlat.values - centroid.y.values))
    idx_lon = np.argmin(np.abs(ds.rlon.values - centroid.x.values))

    ds_closest = ds.isel(rlat=idx_lat, rlon=idx_lon)
    # add lapse rates
    ds_closest['lr_t2m'] = lr_t2m
    ds_closest['lr_rh2'] = lr_rh2
    ds_closest['lr_clt'] = lr_clt
    #ds_closest['lr_rain'] = lr_rain
    ds_closest['lr_tp'] = lr_tp 
    #ds_closest['lr_sf'] = lr_sf
    # add r2 scores
    ds_closest['lr_t2m_r2'] = lr_t2m_r2
    ds_closest['lr_rh2_r2'] = lr_rh2_r2
    ds_closest['lr_clt_r2'] = lr_clt_r2
    #ds_closest['lr_rain'] = lr_rain
    ds_closest['lr_tp_r2'] = lr_tp_r2 
    #ds_closest['lr_sf_r2'] = lr_sf_r2

    return ds_closest
    
def fix_lapse_rates(df, field_var, r2_thres):
    raw_data = df[field_var].values
    r2_data = df[field_var+"_r2"].values

    if field_var == "lr_t2m":
        corr_field = np.where(r2_data > r2_thres, raw_data, -0.0065)
    else:
        corr_field = np.where(r2_data > r2_thres, raw_data, 0.0)

    return corr_field

#### ------------------ ####
#### Load all the paths ####
#### ------------------ ####
path = "/mnt/C4AEBBABAEBB9500/OneDrive/PhD/PhD/Data/Hintereisferner/Climate/CORDEX-DKRZ/"
ps_path = path+"cosmo_1998-2010_6h_ps.nc" #pressure
pr_path = path+"cosmo_1998-2010_1h_pr.nc" #tota precipitaiton flux
tas_path = path+"cosmo_1998-2010_1h_tas.nc" #temperature
hurs_path = path+"cosmo_1998-2010_1h_hurs.nc" #relative humidity
rsds_path = path+"cosmo_1998-2010_1h_rsds.nc" #inc. shortwave
clt_path = path+"cosmo_1998-2010_1h_clt.nc" #total cloud cover
uas_path = path+"cosmo_1998-2010_1h_uas.nc" #u-component wind speed
vas_path = path+"cosmo_1998-2010_1h_vas.nc" #v-component wind speed
orog_path = path+"cosmo_1998-2010_fx_orog.nc" #static file (elevation)
#Note the daily resolution - various downscaling options exist - easiest workflow is to use total precipitation instead
prsn_path = path+"cosmo_1998-2010_1d_prsn.nc" #snowfall flux
#Ideally we want LWin, but in the ESGF-grid only N was given


#load your shp file path
shp_path = "/mnt/C4AEBBABAEBB9500/OneDrive/PhD/PhD/Data/Hintereisferner/Static/RGI6/HEF_RGI6.shp"
# Create output path where the .csv should be stored in
outpath = "./your_csv_name.csv"

#### -------------------- ####
#### Run script           ####
#### -------------------- ####

## Set constants!
lapse_rates = False
cosipy_vars = ['time','T2','RH2','G','RRR','U2','N','PRES'] #'RAIN','SNOWFALL'
roughness_length = 2.12/1000 #z0

#optional: clt, snowfall etc. - easy to implement if necessary
cosipy_vars_extended = cosipy_vars + ['lr_t2m', 'lr_t2m_r2', 'lr_rh2', 'lr_rh2_r2', 'lr_tp', 'lr_tp_r2']

"""Code can be called by opening files beforehand. Example:
    test = salem.open_metum_dataset(tas_path)
    shp_test = gpd.read_file(shp_path)
    tas_ds = _utils_subset_ds(test, shp_test, box_size=3, load_files=False, option="centroid")
"""

if lapse_rates == True:
    tas_ds = _utils_subset_ds(tas_path, shp_path, box_size=3, load_files=True, option="buffer")
    _utils_correct_field_units(tas_ds, "tas", checkonly=True)
    pr_ds = _utils_correct_field_units(_utils_subset_ds(pr_path, shp_path, box_size=3, load_files=True, option="buffer"),
                                       "pr", checkonly=False)
    clt_ds = _utils_correct_field_units(_utils_subset_ds(clt_path, shp_path, box_size=3, load_files=True, option="buffer"),
                                        "clt", checkonly=False)
    hurs_ds = _utils_subset_ds(hurs_path, shp_path, box_size=3, load_files=True, option="buffer")
    _utils_correct_field_units(hurs_ds, "hurs", checkonly=True)
    ps_ds = _utils_correct_field_units(_utils_subset_ds(ps_path, shp_path, box_size=3, load_files=True, option="buffer"),
                                       "ps", checkonly=False)
    ps_ds = _utils_intp_pres(ps_ds)
    uas_ds = _utils_subset_ds(uas_path, shp_path, box_size=3, load_files=True, option="buffer")
    vas_ds = _utils_subset_ds(vas_path, shp_path, box_size=3, load_files=True, option="buffer")
    uas_ds['U2'] = _utils_calc_u2(vas_ds, uas_ds, z0=roughness_length, z=10)
    _utils_correct_field_units(uas_ds, "U2", checkonly=True)
    rsds_ds = _utils_subset_ds(rsds_path, shp_path, box_size=3, load_files=True, option="buffer")
    _utils_correct_field_units(rsds_ds, "rsds", checkonly=True)
    #load static data
    orog_ds = _utils_subset_ds(orog_path, shp_path, box_size=3, load_files=True, option="buffer")
    #Optional load snowfall - here daily so we don't load it as we rely on pr
    #prsn_ds = _utils_correct_field_units(_utils_subset_ds(prsn_path, shp_path, box_size=3, load_files=True, option="centroid"),
    #                                     "prsn", dt=24, checkonly=False)
    
    ## Join all fields - intp. pressure has fewest time stamps due to 6h data, starts in 1999 as well
    #xr. should handle the time steps

    print(pr_ds.time.values[0], pr_ds.time.values[-1])
    print(pr_ds.time_bnds.values[0]) #00:00:00 to 01:00:00 -> sum should be at 1h not at 0h
    #design custom time because some variables are on bounds (every half hour)
    start_time_pr = pd.Timestamp(pr_ds.time.values[0]).ceil('h') #1998-11-01T00:30.. to 1998-11-01T01:00
    start_time_rsds = pd.Timestamp(rsds_ds.time.values[0]).ceil('h')
    start_time_clt = pd.Timestamp(clt_ds.time.values[0]).ceil('h')
    end = pd.Timestamp(ps_ds.time.values[-1])
    
    # Select time up to last pressure data point
    pr_ds = pr_ds.sel(time=slice(None, ps_ds.time[-1]))
    rsds_ds = rsds_ds.sel(time=slice(None, ps_ds.time[-1]))
    clt_ds = clt_ds.sel(time=slice(None, ps_ds.time[-1]))
    #Fix time format by hand - beware of the bounds (check by hand)
    pr_ds["time"] = ("time", pd.date_range(start_time_pr, end, freq="1h"))
    rsds_ds["time"] = ("time", pd.date_range(start_time_rsds, end, freq="1h"))
    clt_ds["time"] = ("time", pd.date_range(start_time_clt, end, freq="1h"))
    
    ps_ds["T2"] = tas_ds["tas"]
    ps_ds["RH2"] = hurs_ds["hurs"]
    ps_ds["G"] = rsds_ds["rsds"]
    ps_ds["U2"] = uas_ds["U2"]
    ps_ds["HGT"] = orog_ds["orog"] 
    #
    ps_ds["N"] = clt_ds["clt_fix"]
    ps_ds["RRR"] = pr_ds["pr_fix"]
    ps_ds = ps_ds.rename({'ps_fix': 'PRES'})
    
    _, _, _, centroid =_utils_bounds_shp(tas_path, shp_path)
    final_ds = create_forcing_with_lr(ps_ds, centroid=centroid)
    
    df = final_ds[cosipy_vars_extended].to_dataframe().reset_index()
    print(f"COSMO grid cell at coordinates lat:{df['lat'].iloc[0]}, lon:{df['lon'].iloc[0]}.")
    df.drop(['rlat','rlon','lat','lon','height'], axis=1, inplace=True)
    df.rename(columns={'time': 'TIMESTAMP'}, inplace=True)
    # some times I got an error because of the timestamp variable - set by hand otherwise
    df['TIMESTAMP'] = pd.date_range(df['TIMESTAMP'].iloc[0],df['TIMESTAMP'].iloc[-1], freq="1h")
    cosipy_df = _utils_set_df_bounds(df)
    
    ## Adjust lapse rates based on R2 0.7
    lr_t2_new = fix_lapse_rates(cosipy_df, "lr_t2m", 0.7)
    lr_rh2_new = fix_lapse_rates(cosipy_df, "lr_rh2", 0.7)
    lr_tp_new = fix_lapse_rates(cosipy_df, "lr_tp", 0.7)
    ##
    cosipy_df['lr_t2m'] = lr_t2_new
    cosipy_df['lr_rh2'] = lr_rh2_new
    cosipy_df['lr_tp'] = lr_tp_new
    cosipy_df.drop(['lr_t2m_r2','lr_rh2_r2','lr_tp_r2'], axis=1, inplace=True)
    
else:
    tas_ds = _utils_subset_ds(tas_path, shp_path, box_size=0, load_files=True, option="centroid")
    _utils_correct_field_units(tas_ds, "tas", checkonly=True)
    pr_ds = _utils_correct_field_units(_utils_subset_ds(pr_path, shp_path, box_size=0, load_files=True, option="centroid"),
                                       "pr", checkonly=False)
    clt_ds = _utils_correct_field_units(_utils_subset_ds(clt_path, shp_path, box_size=0, load_files=True, option="centroid"),
                                        "clt", checkonly=False)
    hurs_ds = _utils_subset_ds(hurs_path, shp_path, box_size=0, load_files=True, option="centroid")
    _utils_correct_field_units(hurs_ds, "hurs", checkonly=True)
    ps_ds = _utils_correct_field_units(_utils_subset_ds(ps_path, shp_path, box_size=0, load_files=True, option="centroid"),
                                       "ps", checkonly=False)
    ps_ds = _utils_intp_pres(ps_ds)
    uas_ds = _utils_subset_ds(uas_path, shp_path, box_size=0, load_files=True, option="centroid")
    vas_ds = _utils_subset_ds(vas_path, shp_path, box_size=0, load_files=True, option="centroid")
    uas_ds['U2'] = _utils_calc_u2(vas_ds, uas_ds, z0=roughness_length, z=10)
    _utils_correct_field_units(uas_ds, "U2", checkonly=True)
    rsds_ds = _utils_subset_ds(rsds_path, shp_path, box_size=0, load_files=True, option="centroid")
    _utils_correct_field_units(rsds_ds, "rsds", checkonly=True)
    #load static data
    orog_ds = _utils_subset_ds(orog_path, shp_path, box_size=0, load_files=True, option="centroid")
    #Optional load snowfall - here daily so we don't load it as we rely on pr
    #prsn_ds = _utils_correct_field_units(_utils_subset_ds(prsn_path, shp_path, box_size=3, load_files=True, option="centroid"),
    #                                     "prsn", dt=24, checkonly=False)
    
    ## Join all fields - intp. pressure has fewest time stamps due to 6h data, starts in 1999 as well
    #xr. should handle the time steps

    print(pr_ds.time.values[0], pr_ds.time.values[-1])
    print(pr_ds.time_bnds.values[0]) #00:00:00 to 01:00:00 -> sum should be at 1h not at 0h
    #design custom time because some variables are on bounds (every half hour)
    start_time_pr = pd.Timestamp(pr_ds.time.values[0]).ceil('h') #1998-11-01T00:30.. to 1998-11-01T01:00
    start_time_rsds = pd.Timestamp(rsds_ds.time.values[0]).ceil('h')
    start_time_clt = pd.Timestamp(clt_ds.time.values[0]).ceil('h')
    end = pd.Timestamp(ps_ds.time.values[-1])
    
    # Select time up to last pressure data point
    pr_ds = pr_ds.sel(time=slice(None, ps_ds.time[-1]))
    rsds_ds = rsds_ds.sel(time=slice(None, ps_ds.time[-1]))
    clt_ds = clt_ds.sel(time=slice(None, ps_ds.time[-1]))
    #Fix time format by hand - beware of the bounds (check by hand)
    pr_ds["time"] = ("time", pd.date_range(start_time_pr, end, freq="1h"))
    rsds_ds["time"] = ("time", pd.date_range(start_time_rsds, end, freq="1h"))
    clt_ds["time"] = ("time", pd.date_range(start_time_clt, end, freq="1h"))
    
    ps_ds["T2"] = tas_ds["tas"]
    ps_ds["RH2"] = hurs_ds["hurs"]
    ps_ds["G"] = rsds_ds["rsds"]
    ps_ds["U2"] = uas_ds["U2"]
    ps_ds["HGT"] = orog_ds["orog"] 
    #
    ps_ds["N"] = clt_ds["clt_fix"]
    ps_ds["RRR"] = pr_ds["pr_fix"]
    ps_ds = ps_ds.rename({'ps_fix': 'PRES'}) 
    
    # Elevation not used in forcing data
    print(f"COSMO grid cell at elevation of {ps_ds['HGT'].item()} m. a.s.l.")
    # Select only relevant variables
    ## fix some dataframe aspects
    df = ps_ds[cosipy_vars].to_dataframe().reset_index()
    print(f"COSMO grid cell at coordinates lat:{df['lat'].iloc[0]}, lon:{df['lon'].iloc[0]}.")
    df.drop(['rlat','rlon','lat','lon','height'], axis=1, inplace=True)
    df.rename(columns={'time': 'TIMESTAMP'}, inplace=True)
    # some times I got an error because of the timestamp variable - set by hand otherwise
    df['TIMESTAMP'] = pd.date_range(df['TIMESTAMP'].iloc[0],df['TIMESTAMP'].iloc[-1], freq="1h")
    cosipy_df = _utils_set_df_bounds(df)
    
cosipy_df.to_csv(outpath, index=False)