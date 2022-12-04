import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.metrics import r2_score
import salem

## ----- Load paths and data ----- ##
era_path = "/data/scratch/richteny/ERA5_HMA/"
plot_path = "."
glacier_outline = "../../data/static/Shapefiles/parlung4_rgi6.shp"

g = 9.80665 #m/s**2 gravitational acceleration
#vars to consider
vars_to_consider = ['t2m','d2m','tp-diff']

#1. Get lat lon of nearest ERA5 cell where glacier is in, warning does not account for bordering cells etc.
def select_nearest_latlon():
    era5_gp = salem.open_xr_dataset(era_path+"ERA5_HMA_15N50N_60E125E_1999_2021_z.nc")
    #Assumes files are in same projection and all ERA5 files have same grid
    shape_grid = salem.read_shapefile_to_grid(glacier_outline, grid=salem.grid_from_dataset(era5_gp))
    lon_distance = np.abs(era5_gp.longitude.values-shape_grid.CenLon.values)
    lat_distance = np.abs(era5_gp.latitude.values-shape_grid.CenLat.values)
    idx_lon = np.where(lon_distance == np.nanmin(lon_distance))
    idx_lat = np.where(lat_distance == np.nanmin(lat_distance))
    latitude = float(era5_gp.latitude[idx_lat].values)
    longitude = float(era5_gp.longitude[idx_lon].values)
    return latitude, longitude

lat, lon = select_nearest_latlon()
print(lat,lon)

#Load static data#
ds_static = salem.open_xr_dataset(era_path+"ERA5_HMA_15N50N_60E125E_1999_2021_z.nc")
ds_static = ds_static.sel(latitude=slice(lat+3.5,lat-3.5), longitude=slice(lon-1.5,lon+1.5))
glacier = salem.read_shapefile(glacier_outline)

#Create 1 degree buffer centered around nearest latlon from glacier
#Create plot
fig, ax = plt.subplots(1,1, figsize=(16,9))
ds_static['HGT'] = ds_static['z'] / g
ax = glacier.plot()
print(ds_static['HGT'])
ds_static['HGT'][0,:,:].plot(ax=ax, zorder=-1)
plt.savefig(plot_path+"glacier_in_static_test.png")
