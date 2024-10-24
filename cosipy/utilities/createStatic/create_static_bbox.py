"""
 This file reads the DEM of the study site and the shapefile and creates the needed static.nc
"""
import sys
import os
import xarray as xr
import numpy as np
from itertools import product
import richdem as rd
import pandas as pd
import fiona
#from horayzon_domain import curved_grid
import numpy as np
from geographiclib.geodesic import Geodesic
#print(curved_grid)
#Since import doesnt work right now

# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License
def curved_grid(domain, dist_search=50.0, ellps="sphere"):
    """Compute digital elevation model domain (curved grid).
    Computes total required domain of digital elevation model for curved grid.
    Parameters
    ----------
    domain : dict
        Dictionary with domain boundaries (lon_min, lon_max, lat_min, lat_max)
        [degree]
    dist_search : float
        Search distance for horizon [kilometre]
    ellps : str
        Earth's surface approximation (sphere, GRS80 or WGS84)
    Returns
    -------
    domain_outer : dict
        Dictionary with outer domain boundaries (lon_min, lon_max,
        lat_min, lat_max) [degree]
    Notes
    -----
    Source:
    - Geoid parameters a and f: PROJ
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion"""

    # Check arguments
    if ellps not in ("sphere", "GRS80", "WGS84"):
        raise NotImplementedError("ellipsoid " + ellps + " is not supported")
    if ((domain["lon_min"] >= domain["lon_max"])
            or (domain["lat_min"] >= domain["lat_max"])):
        raise ValueError("Invalid domain specification")

    # Initialise geodesic
    if ellps == "sphere":
        a = 6370997.0  # earth radius [m]
        f = 0.0  # flattening [-]
    elif ellps == "GRS80":
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        f = (1.0 / 298.257222101)  # flattening [-]
    else:
        a = 6378137.0  # equatorial radius (semi-major axis) [m]
        f = (1.0 / 298.257223563)  # flattening [-]
    b = a * (1.0 - f)  # polar radius (semi-minor axis) [m]
    e_2 = 1.0 - (b ** 2 / a ** 2)  # squared num. eccentricity [-]
    geod = Geodesic(a, f)

    # Compute outer domain boundaries
    lat_abs_max = np.maximum(np.abs(domain["lat_min"]),
                             np.abs(domain["lat_max"]))
    rad_sph = a / np.sqrt(1.0 - e_2 * np.sin(np.deg2rad(lat_abs_max)) ** 2) \
        * np.cos(np.deg2rad(lat_abs_max))  # sphere radius [m]
    lon_add = 360.0 / (2.0 * np.pi * rad_sph) * (dist_search * 1000.0)  # [deg]
    domain_outer = {"lon_min": domain["lon_min"] - lon_add,
                    "lon_max": domain["lon_max"] + lon_add,
                    "lat_min": geod.Direct(domain["lat_min"], 0.0, 180.0,
                                           dist_search * 1000.0)["lat2"],
                    "lat_max": geod.Direct(domain["lat_max"], 0.0, 0.0,
                                           dist_search * 1000.0)["lat2"]}

    # Check if total domain is within valid range (lon: -/+180.0, lat: -/+90.0)
    if ((domain_outer["lon_min"] < -180.0)
            or (domain_outer["lon_max"] > 180.0)
            or (domain_outer["lat_min"] < -90.0)
            or (domain_outer["lat_max"] > 90.0)):
        raise ValueError("total domain exceeds valid range")

    return domain_outer

static_folder = '../../data/static/'

tile = True
aggregate = True
elevation_test = False

### input digital elevation model (DEM)
dem_path_tif = static_folder + 'DEM/ALOS_N039E071_AVE_DSM.tif'
### input shape of glacier or study area, e.g. from the Randolph glacier inventory
shape_path = static_folder + 'Shapefiles/abramov_rgi6.shp'
### path were the static.nc file is saved
output_path = static_folder + 'Abramov_90m_static_test.nc'

#get bbox
shp = fiona.open(shape_path)
print(shp.bounds) #minlon, minlat, maxlon, maxlat
print(shp.bounds[0],shp.bounds[2])
print(shp.bounds[1],shp.bounds[3])

domain = {"lon_min": shp.bounds[0], "lon_max": shp.bounds[2],
          "lat_min": shp.bounds[1], "lat_max": shp.bounds[3]}

dist_search = 25.0 # search distance for terrain shading [kilometre]
ellps = "WGS84"

# Copyright (c) 2022 ETH Zurich, Christian R. Steger
# MIT License

domain_outer = curved_grid(domain, dist_search, ellps)
print(domain_outer)

### to shrink the DEM use the following lat/lon corners
#for abramov
longitude_upper_left = domain_outer['lon_min']  #71.52 '71.4983'
latitude_upper_left = domain_outer['lat_max'] #39.66 '39.659'
longitude_lower_right = domain_outer['lon_max'] #71.57 '71.6'
latitude_lower_right = domain_outer['lat_min'] #39.57 '39.583'

#convert to str
longitude_upper_left = str(longitude_upper_left)
latitude_upper_left = str(latitude_upper_left)
longitude_lower_right = str(longitude_lower_right)
latitude_lower_right = str(latitude_lower_right)

### to aggregate the DEM to a coarser spatial resolution
aggregate_degree =  '0.000833333' #'0.002777778' #300m

### intermediate files, will be removed afterwards
dem_path_tif_temp = static_folder + 'DEM_temp.tif'
dem_path_tif_temp2 = static_folder + 'DEM_temp2.tif'
dem_path = static_folder + 'dem.nc'
aspect_path = static_folder + 'aspect.nc'
mask_path = static_folder + 'mask.nc'
slope_path = static_folder + 'slope.nc'

### If you do not want to shrink the DEM, comment out the following to three lines
if tile:
    os.system('gdal_translate -projwin ' + longitude_upper_left + ' ' + latitude_upper_left + ' ' +
          longitude_lower_right + ' ' + latitude_lower_right + ' ' + dem_path_tif + ' ' + dem_path_tif_temp)
    dem_path_tif = dem_path_tif_temp

### If you do not want to aggregate DEM, comment out the following to two lines
if aggregate:
    os.system('gdalwarp -tr ' + aggregate_degree + ' ' + aggregate_degree + ' -r average ' + dem_path_tif + ' ' + dem_path_tif_temp2)
    dem_path_tif = dem_path_tif_temp2

### convert DEM from tif to NetCDF
os.system('gdal_translate -of NETCDF ' + dem_path_tif  + ' ' + dem_path)

### calculate slope as NetCDF from DEM
os.system('gdaldem slope -of NETCDF ' + dem_path + ' ' + slope_path + ' -s 111120')

### calculate aspect from DEM
aspect = np.flipud(rd.TerrainAttribute(rd.LoadGDAL(dem_path_tif, no_data= np.nan), attrib = 'aspect'))

### calculate mask as NetCDF with DEM and shapefile
os.system('gdalwarp -of NETCDF  --config GDALWARP_IGNORE_BAD_CUTLINE YES -cutline ' + shape_path + ' ' + dem_path_tif  + ' ' + mask_path)

### open intermediate netcdf files
dem = xr.open_dataset(dem_path)
mask = xr.open_dataset(mask_path)
slope = xr.open_dataset(slope_path)

### set NaNs in mask to -9999 and elevation within the shape to 1
mask=mask.Band1.values
mask[np.isnan(mask)]=-9999
mask[mask>0]=1
print(mask)

## create output dataset
ds = xr.Dataset()
ds.coords['lon'] = dem.lon.values
ds.lon.attrs['standard_name'] = 'lon'
ds.lon.attrs['long_name'] = 'longitude'
ds.lon.attrs['units'] = 'degrees_east'

ds.coords['lat'] = dem.lat.values
ds.lat.attrs['standard_name'] = 'lat'
ds.lat.attrs['long_name'] = 'latitude'
ds.lat.attrs['units'] = 'degrees_north'

### function to insert variables to dataset
def insert_var(ds, var, name, units, long_name):
    ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999

### function to assign attributes to variable
def assign_attrs(ds, name, units, long_name):
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    ds[name].attrs['_FillValue'] = -9999

### insert needed static variables
insert_var(ds, dem.Band1.values,'HGT','meters','meter above sea level')
insert_var(ds, aspect,'ASPECT','degrees','Aspect of slope')
insert_var(ds, slope.Band1.values,'SLOPE','degrees','Terrain slope')
insert_var(ds, mask,'MASK','boolean','Glacier mask')

os.system('rm '+ dem_path + ' ' + mask_path + ' ' + slope_path + ' ' + dem_path_tif_temp + ' '+ dem_path_tif_temp2)

### save combined static file, delete intermediate files and print number of glacier grid points
def check_for_nan(ds,var=None):
    for y,x in product(range(ds.dims['lat']),range(ds.dims['lon'])):
        mask = ds.MASK.isel(lat=y, lon=x)
        if mask==1:
            if var is None:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the static fields')
                    sys.exit()
            else:
                if np.isnan(ds[var].isel(lat=y, lon=x)).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the static fields')
                    sys.exit()
check_for_nan(ds)
ds.to_netcdf(output_path)

### Create 1D Elevation Bins test data ###

if elevation_test:

    if aggregate == True:
        print("Warning aggregation is active. You are not using the best available resolution.")
        print("\nAggregation level: " + str(aggregate_degree))
    else:
        print("Starting calculation of elevation datasets with native resolution of DEM dataset.")
 
#function for mean of circular values strongly inspired by http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%201
    def aspect_mean(aspects):
        mean_sine = np.sum(np.sin(np.radians(aspects)))/len(aspects)
        mean_cosine = np.sum(np.cos(np.radians(aspects)))/len(aspects)
        r = np.sqrt(mean_cosine**2+mean_sine**2)
        cos_mean = mean_cosine/r
        sin_mean = mean_sine/r
        mean_angle = np.arctan2(sin_mean, cos_mean)
        return np.degrees(mean_angle)

    #select only glacier fields
    elev_bandsize = 30 #in m 
    
    elevations = ds.HGT.values.flatten()[ds.MASK.values.flatten() == 1]
    slopes = ds.SLOPE.values.flatten()[ds.MASK.values.flatten() == 1]
    aspects = ds.ASPECT.values.flatten()[ds.MASK.values.flatten() == 1]
    bands = []
    number_points = []
    slope_means = []
    aspect_means = []
    lat_means = []
    lon_means = []

    for i in (np.arange(np.min(elevations),np.max(elevations),elev_bandsize)):
        print(i)
        print(np.max(elevations))
        bands.append(i+elev_bandsize/2)
        greater = elevations[elevations>=i]
        #get mean lat and lon of relevant pixel in glacier mask in elev band
        sub = ds.where((ds.MASK == 1) & (ds.HGT >= i) & (ds.HGT < i + elev_bandsize), drop=True)
        lat_means.append(np.nanmean(sub.lat.values))
        lon_means.append(np.nanmean(sub.lon.values))
        number_points.append(len(greater[greater<i+elev_bandsize]))
        slope_means.append(np.nanmean(slopes[np.logical_and(elevations >= i, elevations < i+elev_bandsize)]))
        aspect_means.append(aspect_mean(aspects[np.logical_and(elevations >= i, elevations < i+elev_bandsize)]))

    full_data = False #This is the crudest and most simplest try and here I want to avoid having a 26x26 grid filled with NaNs due to computational time
    mask_elev = np.ones_like(bands)
    ## Suggest all points on glacier
    if full_data:
        df = pd.DataFrame({'lat': lat_means,
                           'lon': lon_means,
                           'HGT': bands,
                           'ASPECT': aspect_means,
                           'SLOPE': slope_means,
                           'MASK': mask_elev,
                           'N_Points': number_points})
    else:
        df = pd.DataFrame({'lat':lat_means,
                           'lon': np.mean(lon_means), #just assign the same value for now for simplicity
                           'HGT': bands,
                           'ASPECT': aspect_means,
                           'SLOPE': slope_means,
                           'MASK': mask_elev,
                           'N_Points': number_points})

    df.set_index(['lat','lon'], inplace=True)
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
    #Holds a lot of NaNs, is there a way to get rid of them?
    print(elev_bandsize)
    elev_ds.to_netcdf(static_folder+'Abramov_hrzd1D_{}m_elev.nc'.format(elev_bandsize))

print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")
print("Done")

