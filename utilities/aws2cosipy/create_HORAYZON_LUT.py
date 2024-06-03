# Description: Compute gridded correction factor for downward direct shortwave
#              radiation from given DEM data (~30 m) and
#              mask all non-glacier grid cells according to the glacier outline.
#              Consider Earth's surface curvature.
#
# Important note: An Earthdata account is required and 'wget' has to be set
#                 (https://disc.gsfc.nasa.gov/data-access) to download NASADEM
#                 data successfully.
#
# Source of applied DEM data: https://lpdaac.usgs.gov/products/nasadem_hgtv001/
#
# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

##WORK IN PROGRESS!##

### Current TASK: make 1D version runnable - involves probably an averaging of the SW corr. factor per elevation band
### Is the double rgridding a good work?
### OGGM support recent texts to derive the OGGM-based elevation bands etc.?  Combine with older Notebook from patrick

# Load modules
import os
import numpy as np
import subprocess
from netCDF4 import Dataset, date2num
import zipfile
from skyfield.api import load, wgs84
import time
import fiona
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import shape
import datetime
import datetime as dt
import horayzon as hray
import xarray as xr
import sys
import xesmf as xe
import pandas as pd

sys.path.append("../..")
from utilities.aws2cosipy.crop_file_to_glacier import crop_file_to_glacier
from utilities.aws2cosipy.aws2cosipyConfig import WRF

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

# set paths
regrid = False #regrid to coarser resolution
elevation_profile = False ## 1D COSIPY
elev_bandsize = 30 #in m 
if elevation_profile == True:
    print("Routine check. Regrid Option is set to: ", regrid)
    print("Setting regrid to False.")
    print("Elevation band size is set to: ", elev_bandsize, "m")
    regrid = False
ellps = "WGS84"  # Earth's surface approximation (sphere, GRS80 or WGS84)
path_out = "../../data/static/HEF/"
file_sw_dir_cor = "LUT_HORAYZON_sw_dir_cor_raw.nc"

static_file = "../../data/static/HEF/HEF_static_raw.nc" #path to high resolution dataset
coarse_static_file = "../../data/static/HEF/HEF_static_agg.nc" #Load coarse grid

# ----------------------------
# Some helper functions
# ----------------------------

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    if WRF:
         ds[name] = (('time','south_north','west_east'), var)	
    else:
        ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function self.adds missing variables to the self.DATA class """
    if WRF: 
        ds[name] = (('south_north','west_east'), var)
    else:
        ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].encoding['_FillValue'] = -9999
    return ds

### function to assign attributes to variable
def assign_attrs(ds, name, units, long_name):
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999

#function for mean of circular values strongly inspired by http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%201
def aspect_means(x):
    mean_sine = np.nanmean(np.sin(np.radians(x)))
    mean_cosine = np.nanmean(np.cos(np.radians(x)))
    r = np.sqrt(mean_cosine**2 + mean_sine**2)
    cos_mean = mean_cosine/r
    sin_mean = mean_sine/r
    mean_angle = np.arctan2(sin_mean, cos_mean)
    return np.degrees(mean_angle) 


## ASPECT test raises error.. fix that! And then do some tests to see that it actually works, especially considering the masks and elevation bins
## WhY?? - np.nanmean also doesnt work, so issue maybe with "map" ? 

def calculate_1d_elevationband(xds, elevation_var, mask_var, var_of_interest, elev_bandsize, slice_idx=None):
    
    ## first mask vals
    xds = xds.where(xds[mask_var] == 1, drop=True)
    
    #test groupby bins
    full_elev_range = xds[elevation_var].values[xds[mask_var] == 1]
    bins = np.arange(np.nanmin(full_elev_range), np.nanmax(full_elev_range)+elev_bandsize, elev_bandsize)
    labels = bins[:-1] + elev_bandsize/2
    
    if var_of_interest in ["lat","lon"]:
        values = []
        for i in (bins):
            sub = xds.where((xds[mask_var] == 1) & (xds[elevation_var] >= i) & (xds[elevation_var] < i + elev_bandsize), drop=True)
            values.append(np.nanmean(sub[var_of_interest].values))
    elif var_of_interest == "ASPECT":
        elvs = xds[elevation_var].values.flatten()[xds[mask_var].values.flatten() == 1]
        aspects = xds[var_of_interest].values.flatten()[xds[mask_var].values.flatten() == 1]
        values = []
        for i in (bins):
            values.append(aspect_means(aspects[np.logical_and(elvs >= i, elvs < i+elev_bandsize)]))
    elif var_of_interest == mask_var:
        if slice_idx is None:
            values = xds[var_of_interest].groupby_bins(xds[elevation_var], bins, labels=labels, include_lowest=True).sum(skipna=True, min_count=1)
        else:
            values = xds[var_of_interest][slice_idx].groupby_bins(xds[elevation_var][slice_idx], bins, labels=labels, include_lowest=True).sum(skipna=True, min_count=1)
        ## below calculation doesnt work
    #elif var_of_interest in ["aspect","ASPECT"]:
    #    if slice_idx is None:          
    #        values = xds[var_of_interest].groupby_bins(xds[elevation_var], bins, labels=labels, include_lowest=True).map(aspect_means)
    #    else:
    #        values = xds[var_of_interest][slice_idx].groupby_bins(xds[elevation_var][slice_idx], bins, labels=labels, include_lowest=True).map(aspect_means)
    else:
        if slice_idx is None:
            values = xds[var_of_interest].groupby_bins(xds[elevation_var], bins, labels=labels, include_lowest=True).mean(skipna=True)
            
        else:
            values = xds[var_of_interest][slice_idx].groupby_bins(xds[elevation_var][slice_idx], bins, labels=labels, include_lowest=True).mean(skipna=True)
    
    return values    

def construct_1d_dataset(df):
    elev_ds = df.to_xarray()
    elev_ds.lon.attrs['standard_name'] = 'lon'
    elev_ds.lon.attrs['long_name'] = 'longitude'
    elev_ds.lon.attrs['units'] = 'Average Lon of elevation bands'
    
    elev_ds.lat.attrs['standard_name'] = 'lat'
    elev_ds.lat.attrs['long_name'] = 'latitude'
    elev_ds.lat.attrs['units'] = 'Average Lat of elevation bands'
    assign_attrs(elev_ds, 'HGT','meters','Mean of elevation range per bin as meter above sea level')
    assign_attrs(elev_ds, 'ASPECT','degrees','Mean Aspect of slope')
    assign_attrs(elev_ds, 'SLOPE','degrees','Mean Terrain slope')
    assign_attrs(elev_ds, 'MASK','boolean','Glacier mask')
    assign_attrs(elev_ds, 'N_Points','count','Number of Points in each bin')
    assign_attrs(elev_ds, 'sw_dir_cor','-','Average shortwave radiation correction factor per elevation band')
    
    return elev_ds

# -----------------------------------------------------------------------------
# Prepare data and initialise Terrain class
# -----------------------------------------------------------------------------

# Check if output directory exists
if not os.path.isdir(path_out):
    os.makedirs(path_out, exist_ok=True)

# Load high resolution static data
ds = xr.open_dataset(static_file)
elevation = ds["HGT"].values.copy() #else values get overwritten by later line
elevation_original = ds["HGT"].values.copy()
lon = ds["lon"].values
lat = ds["lat"].values

# Compute indices of inner domain -> needs to encompass everything in range for aggregation
slice_in = (slice(1,lat.shape[0]-1, None), slice(1, lon.shape[0]-1))

offset_0 = slice_in[0].start
offset_1 = slice_in[1].start
print("Inner domain size: " + str(elevation[slice_in].shape))

#orthometric height (-> height above mean sea level)
elevation_ortho = np.ascontiguousarray(elevation[slice_in])

# Compute ellipsoidal heights
elevation += hray.geoid.undulation(lon, lat, geoid="EGM96")  # [m]

# Compute glacier mask
mask_glacier_original = ds["MASK"].values
#set NaNs to zero, relict from create static file
mask_glacier_original[np.isnan(mask_glacier_original)] = 0
mask_glacier = mask_glacier_original.astype(bool)
mask_glacier = mask_glacier[slice_in] #-1 -1 verywhere

#mask with buffer for aggregation to lower spatial resolutions
#set +- 11 grid cells to "glacier" to allow ensure regridding
ilist = []
jlist = []
## Note that this list is not based on the original shape, see slice_in above
for i in np.arange(0,mask_glacier.shape[0]):
    for j in np.arange(0,mask_glacier.shape[1]):
        if mask_glacier[i,j] == True:
            #print("Grid cell is glacier.")
            ilist.append(i)
            jlist.append(j)
#create buffer around glacier
ix_latmin = np.min(ilist)
ix_latmax = np.max(ilist)
ix_lonmin = np.min(jlist)
ix_lonmax = np.max(jlist)

#Watch out that the large domain incorporates the buffer
slice_buffer = (slice(ix_latmin-11,ix_latmax+11), slice(ix_lonmin-11, ix_lonmax+11))
mask_glacier[slice_buffer] = True

# Compute ECEF coordinates
x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(*np.meshgrid(lon, lat),
                                                    elevation, ellps=ellps)
dem_dim_0, dem_dim_1 = elevation.shape

# Compute ENU coordinates
trans_ecef2enu = hray.transform.TransformerEcef2enu(
    lon_or=lon[int(len(lon) / 2)], lat_or=lat[int(len(lat) / 2)], ellps=ellps)
x_enu, y_enu, z_enu = hray.transform.ecef2enu(x_ecef, y_ecef, z_ecef,
                                              trans_ecef2enu)

# Compute unit vectors (up and north) in ENU coordinates for inner domain
vec_norm_ecef = hray.direction.surf_norm(*np.meshgrid(lon[slice_in[1]],
                                                      lat[slice_in[0]]))
vec_north_ecef = hray.direction.north_dir(x_ecef[slice_in], y_ecef[slice_in],
                                          z_ecef[slice_in], vec_norm_ecef,
                                          ellps=ellps)
del x_ecef, y_ecef, z_ecef
vec_norm_enu = hray.transform.ecef2enu_vector(vec_norm_ecef, trans_ecef2enu)
vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
del vec_norm_ecef, vec_north_ecef

# Merge vertex coordinates and pad geometry buffer
# holds all the data
vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

# Compute rotation matrix (global ENU -> local ENU)

rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(vec_north_enu,
                                                           vec_norm_enu)

del vec_north_enu

# Compute slope (in global ENU coordinates!)
slice_in_a1 = (slice(slice_in[0].start - 1, slice_in[0].stop + 1),
               slice(slice_in[1].start - 1, slice_in[1].stop + 1))

## Slope vs plain method -> for comparison later
vec_tilt_enu = \
    np.ascontiguousarray(hray.topo_param.slope_vector_meth(
        x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
        rot_mat=rot_mat_glob2loc, output_rot=False)[1:-1, 1:-1])

# Compute surface enlargement factor
surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
print("Surface enlargement factor (min/max): %.3f" % surf_enl_fac.min()
      + ", %.3f" % surf_enl_fac.max())

# Initialise terrain
mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
mask[~mask_glacier] = 0  # mask non-glacier grid cells

terrain = hray.shadow.Terrain()
dim_in_0, dim_in_1 = vec_tilt_enu.shape[0], vec_tilt_enu.shape[1]
terrain.initialise(vert_grid, dem_dim_0, dem_dim_1,
                   offset_0, offset_1, vec_tilt_enu, vec_norm_enu,
                   surf_enl_fac, mask=mask, elevation=elevation_ortho,
                   refrac_cor=False)
# -> neglect atmospheric refraction -> effect is weak due to high
#    surface elevation and thus low atmospheric surface pressure

# Load Skyfield data
load.directory = path_out
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
loc_or = earth + wgs84.latlon(trans_ecef2enu.lat_or, trans_ecef2enu.lon_or)
# -> position lies on the surface of the ellipsoid by default

# -----------------------------------------------------------------------------
# Compute Slope and Aspect
# -----------------------------------------------------------------------------
# Compute slope (in local ENU coordinates!)
vec_tilt_enu_loc = \
    np.ascontiguousarray(hray.topo_param.slope_vector_meth(
        x_enu, y_enu, z_enu,
        rot_mat=rot_mat_glob2loc, output_rot=True)[1:-1, 1:-1])

# Compute slope angle and aspect (in local ENU coordinates)
slope = np.arccos(vec_tilt_enu_loc[:, :, 2].clip(max=1.0))
#beware of aspect orientation -> N = 0 in HORAYZON, adjust here
aspect = np.pi / 2.0 - np.arctan2(vec_tilt_enu_loc[:, :, 1],
                                  vec_tilt_enu_loc[:, :, 0])
aspect[aspect < 0.0] += np.pi * 2.0  # [0.0, 2.0 * np.pi]

#Create output file for HRZ
static_ds = xr.Dataset()
static_ds.coords['lat'] = lat[slice_buffer[0]]
static_ds.coords['lon'] = lon[slice_buffer[1]]
add_variable_along_latlon(static_ds, elevation_ortho[slice_buffer], "elevation", "m", "Orthometric Height")
add_variable_along_latlon(static_ds, np.rad2deg(slope)[slice_buffer], "slope", "degree", "Slope")
add_variable_along_latlon(static_ds, np.rad2deg(aspect)[slice_buffer], "aspect", "m", "Aspect measured clockwise from North")
add_variable_along_latlon(static_ds, surf_enl_fac[slice_buffer], "surf_enl_fac", "-", "Surface enlargement factor")

# -----------------------------------------------------------------------------
# Compute correction factor for direct downward shortwave radiation
# -----------------------------------------------------------------------------

# Create time axis
# time in UTC, set timeframe here
time_dt_beg = dt.datetime(2020, 1, 1, 0, 00, tzinfo=dt.timezone.utc)
time_dt_end = dt.datetime(2021, 1, 1, 0, 00, tzinfo=dt.timezone.utc)
dt_step = dt.timedelta(hours=1)
num_ts = int((time_dt_end - time_dt_beg) / dt_step)
ta = [time_dt_beg + dt_step * i for i in range(num_ts)]

# Add sw dir correction and regrid
comp_time_shadow = []
sw_dir_cor = np.zeros(vec_tilt_enu.shape[:2], dtype=np.float32)

##Load coarse grid
ds_coarse = xr.open_dataset(coarse_static_file)
ds_coarse['mask'] = ds_coarse['MASK'] #prepare for masked regridding

### Build regridder ###
#Create sample dataset to use regridding for
#Create data for first timestep.
ts = load.timescale()
t = ts.from_datetime(ta[0])
astrometric = loc_or.at(t).observe(sun)
alt, az, d = astrometric.apparent().altaz()
x_sun = d.m * np.cos(alt.radians) * np.sin(az.radians)
y_sun = d.m * np.cos(alt.radians) * np.cos(az.radians)
z_sun = d.m * np.sin(alt.radians)
sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)

terrain.sw_dir_cor(sun_position, sw_dir_cor)

## Construct regridder outside of loop - create empty place holder netcdf
result = xr.Dataset()
result.coords['time'] = [pd.to_datetime(ta[0])]
#ix_latmin-11:ix_latmax+11,ix_lonmin-11:ix_lonmax+11
result.coords['lat'] = lat[slice_buffer[0]]
result.coords['lon'] = lon[slice_buffer[1]]
sw_holder = np.zeros(shape=[1, lat[slice_buffer[0]].shape[0], lon[slice_buffer[1]].shape[0]])
sw_holder[0,:,:] = sw_dir_cor[slice_buffer]
mask_crop = mask[slice_buffer]
add_variable_along_timelatlon(result, sw_holder, "sw_dir_cor", "-", "correction factor for direct downward shortwave radiation")
add_variable_along_latlon(result, mask_crop, "mask", "-", "Boolean Glacier Mask")

#build regridder
if regrid == True:
    regrid_mask = xe.Regridder(result, ds_coarse, method="conservative_normed")

result.close()
del result

## Regridder successfully constructed, close and delete place holder

#Iterate over timesteps

datasets = []
for i in range(len(ta)): #loop over timesteps

    t_beg = time.time()

    ts = load.timescale()
    t = ts.from_datetime(ta[i])
    print(i)
    astrometric = loc_or.at(t).observe(sun)
    alt, az, d = astrometric.apparent().altaz()
    x_sun = d.m * np.cos(alt.radians) * np.sin(az.radians)
    y_sun = d.m * np.cos(alt.radians) * np.cos(az.radians)
    z_sun = d.m * np.sin(alt.radians)
    sun_position = np.array([x_sun, y_sun, z_sun], dtype=np.float32)

    terrain.sw_dir_cor(sun_position, sw_dir_cor)

    comp_time_shadow.append((time.time() - t_beg))
    
    ## first create distributed 2d xarray
    result = xr.Dataset()
    result.coords['time'] = [pd.to_datetime(ta[i])]
    #ix_latmin-11:ix_latmax+11,ix_lonmin-11:ix_lonmax+11
    result.coords['lat'] = lat[slice_buffer[0]]
    result.coords['lon'] = lon[slice_buffer[1]]
    sw_holder = np.zeros(shape=[1, lat[slice_buffer[0]].shape[0], lon[slice_buffer[1]].shape[0]])
    sw_holder[0,:,:] = sw_dir_cor[slice_buffer]
    ## this sets the whole small domain to mask == 1 - might have issues in regridding (considers values outside actual mask) but nvm
    mask_crop = mask[slice_buffer]
    add_variable_along_timelatlon(result, sw_holder, "sw_dir_cor", "-", "correction factor for direct downward shortwave radiation")
    # XESMF regridding requires fieldname "mask" to notice it
    add_variable_along_latlon(result, mask_crop, "mask", "-", "Boolean Glacier Mask")
    
    if elevation_profile == True:
        elev_holder = elevation_original[slice_in] #could also use elevation_ortho here
        mask_holder = mask_glacier_original[slice_in]
        add_variable_along_latlon(result,elev_holder[slice_buffer], "HGT", "m asl", "Surface elevation" )
        ## load actual mask
        add_variable_along_latlon(result,mask_holder[slice_buffer], "mask_real", "-", "Actual Glacier Mask" )
        
        full_elev_range = result["HGT"].values[result["mask_real"] == 1]
        bins = np.arange(np.nanmin(full_elev_range), np.nanmax(full_elev_range)+elev_bandsize, elev_bandsize)
        labels = bins[:-1] + elev_bandsize/2
        
        placeholder = {}
        for var in ["SLOPE","ASPECT","lat","lon"]:
            placeholder[var] = calculate_1d_elevationband(ds, "HGT", "MASK", var, elev_bandsize)
        
        for var in ["sw_dir_cor","mask_real"]:
            placeholder[var] = calculate_1d_elevationband(result, "HGT", "mask_real", var, elev_bandsize)
        
        ## construct the dataframe and xarray dataset
        #This is the crudest and most simplest try and here I want to avoid having a 26x26 grid filled with NaNs due to computational time
        mask_elev = np.ones_like(placeholder['lat'][:-1])
        ## Suggest all points on glacier
        df = pd.DataFrame({'lat':placeholder['lat'][:-1],
                           'lon': np.mean(placeholder['lon'][:-1]), #just assign the same value for now for simplicity
                           'time': pd.to_datetime(ta[i]), 
                           'HGT': labels,
                           'ASPECT': placeholder['ASPECT'][:-1],        
                           'SLOPE': placeholder['SLOPE'].data,
                           'MASK': mask_elev,
                           'N_Points': placeholder["mask_real"].data,
                           'sw_dir_cor': placeholder["sw_dir_cor"][0,:].data})
        
        #drop the timezone argument from pandas datetime object to ensure fluent conversion into xarray
        df['time'] = df['time'].dt.tz_localize(None)        
        ##sort values by index vars, just in case
        df.sort_values(by=["time","lat","lon"], inplace=True)
        df.set_index(['time','lat','lon'], inplace=True)
        
        elev_ds = construct_1d_dataset(df)
    
    now = time.time()
    if regrid == True:  
        datasets.append(regrid_mask(result))
        #print("regridding took:", time.time()-now)
    else:
        if elevation_profile == True:
            datasets.append(elev_ds)
            elev_ds.close()
            del elev_ds
            del df
        else:
            datasets.append(result)
    #Close and delete files to free memory
    result.close()
    del result


#Merge single timestep files
now = time.time()
ds_sw_cor = xr.concat(datasets, dim='time')
ds_sw_cor['time'] = pd.to_datetime(ds_sw_cor['time'].values)
if regrid == True:
    ds_sw_cor['MASK'] = ds_coarse['MASK'] #replace with original mask, should have same dimensions
else:
    #dont have same dimensions
    if elevation_profile == True:
        ds_sw_cor['HGT'] = ds_sw_cor['HGT'].isel(time=0)
        ds_sw_cor['ASPECT'] = ds_sw_cor['ASPECT'].isel(time=0)
        ds_sw_cor['SLOPE'] = ds_sw_cor['SLOPE'].isel(time=0)
        ds_sw_cor['MASK'] = ds_sw_cor['MASK'].isel(time=0)
        ds_sw_cor['N_Points'] = ds_sw_cor['N_Points'].isel(time=0)
    else:
        mask_holder = mask_glacier_original[slice_in]
        add_variable_along_latlon(ds_sw_cor,mask_holder[slice_buffer], "MASK", "-", "Actual Glacier Mask" )


ds_sw_cor['MASK'] = (('lat','lon'),np.where(ds_sw_cor['MASK'] == 1, ds_sw_cor['MASK'], np.nan))
if elevation_profile == False:
    ds_sw_cor = ds_sw_cor[['sw_dir_cor','MASK']]
print("concat took:", time.time()-now)

time_tot = np.array(comp_time_shadow).sum()
print("Elapsed time (total / per time step): " + "%.2f" % time_tot
      + " , %.2f" % (time_tot / len(ta)) + " s")

#regrid static ds and merge with sw_dir_cor
if regrid == True:
    regrid_no_mask = xe.Regridder(static_ds, ds_coarse[["HGT"]], method="conservative_normed")
    regrid = regrid_no_mask(static_ds, ds_coarse[["HGT"]])
    combined = xr.merge([ds_sw_cor, regrid])
else:
    if elevation_profile == True:
        combined = ds_sw_cor.copy()
    if elevation_profile == False:
        combined = xr.merge([ds_sw_cor, static_ds])

#BBox script to crop to minimal extent!
if elevation_profile == True:
    combined.to_netcdf(path_out+file_sw_dir_cor)
    combined[['HGT','ASPECT','SLOPE','MASK','N_Points']].to_netcdf(path_out+"HEF_static_30m_elevbands.nc")
else:
    #cropped_combined = crop_file_to_glacier(combined) #results in +2 gridsize somehow
    cropped_combined = combined.where(combined.MASK ==1, drop=True)
    cropped_combined.to_netcdf(path_out+file_sw_dir_cor)


## CURRENT VERSION for the Elevation Profile is taking a longer time to run
## First focus on running at all, then we can check for performance improvements
