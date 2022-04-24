"""
 This file reads the DEM of the study site and the shapefile and creates the needed static.nc
"""
import sys
import os
import xarray as xr
import numpy as np
from itertools import product
import richdem as rd

static_folder = '../../data/static/'

tile = True
aggregate = True
elevation_test = True

### input digital elevation model (DEM)
dem_path_tif = static_folder + 'DEM/ALOS_N039E071_AVE_DSM.tif'
### input shape of glacier or study area, e.g. from the Randolph glacier inventory
shape_path = static_folder + 'Shapefiles/abramov_rgi6.shp'
### path were the static.nc file is saved
output_path = static_folder + 'Abramov_600m_static.nc'

### to shrink the DEM use the following lat/lon corners
#for abramov
longitude_upper_left = '71.52' #'71.4983'
latitude_upper_left =  '39.66'#'39.659'
longitude_lower_right = '71.57' #'71.6'
latitude_lower_right =  '39.57' #'39.583'

### to aggregate the DEM to a coarser spatial resolution
aggregate_degree = '0.00555556'

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
    elev_bandsize = 50 #in m 
    
    elevations = ds.HGT.values.flatten()[ds.MASK.values.flatten() == 1]
    slopes = ds.SLOPE.values.flatten()[ds.MASK.values.flatten() == 1]
    aspects = ds.ASPECT.values.flatten()[ds.MASK.values.flatten() == 1]
    bands = []
    number_points = []
    slope_means = []
    aspect_means = []

    for i in (np.arange(np.min(elevations),np.max(elevations),elev_bandsize)):
        bands.append(i+elev_bandsize/2)
        greater = elevations[elevations>=i]
        number_points.append(len(greater[greater<i+elev_bandsize]))
        slope_means.append(np.nanmean(slopes[np.logical_and(elevations >= i, elevations < i+elev_bandsize)]))
        aspect_means.append(aspect_mean(aspects[np.logical_and(elevations >= i, elevations < i+elev_bandsize)]))

    elev_ds = xr.Dataset()
    elev_ds.coords['lon'] = np.arange(len(bands))
    elev_ds.lon.attrs['standard_name'] = 'lon'
    elev_ds.lon.attrs['long_name'] = 'longitude'
    elev_ds.lon.attrs['units'] = 'index'

    elev_ds.coords['lat'] = np.array([1])
    elev_ds.lat.attrs['standard_name'] = 'lat'
    elev_ds.lat.attrs['long_name'] = 'latitude'
    elev_ds.lat.attrs['units'] = 'index'

    mask_elev = np.ones_like(bands)
    ### insert lists into dataset
    insert_var(elev_ds, np.reshape(bands, (1, -1)), 'HGT', 'meters', 'Mean of elevation range per bin as meter above sea level')
    insert_var(elev_ds, np.reshape(aspect_means, (1, -1)), 'ASPECT', 'degrees', 'Mean Aspect of slope')
    insert_var(elev_ds, np.reshape(slope_means, (1, -1)), 'SLOPE', 'degrees', 'Mean Terrain slope')
    insert_var(elev_ds, np.reshape(mask_elev, (1, -1)), 'MASK', 'boolean', 'Glacier mask')
    insert_var(elev_ds, np.reshape(number_points, (1, -1)), 'N_Points', 'count', 'Number of Points in each bin')

    elev_ds.to_netcdf(static_folder+'Abramov_1D_{}m_elev.nc'.format(elev_bandsize))

print("Study area consists of ", np.nansum(mask[mask==1]), " glacier points")
print("Done")

