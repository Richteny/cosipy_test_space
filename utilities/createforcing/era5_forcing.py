"""
 This file reads the input data (model forcing) and write the output to netcdf file.  There is the create_1D_input
 (point model) and create_2D_input (distirbuted simulations) function. In case of the 1D input, the function works
 without a static file, in that file the static variables are created. For both cases, lapse rates can be determined
 in the aws2cosipyConfig.py file.
"""
import sys
import xarray as xr
import pandas as pd
import numpy as np
import metpy.calc
import pathlib
import salem
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import matplotlib.pyplot as plt
sys.path.append('../../')
import argparse

## Paths ##
data_path = "/data/scratch/richteny/ERA5_HMA/"
glacier_outline = "../../data/static/Shapefiles/abramov_rgi6.shp"
dem = "../../data/static/DEM/ALOS_N039E071_AVE_DSM.tif"

dic_cosipy_vars = {}

## Constants ##
a1 = 611.21 #Pa
a3 = 17.502
a4 = 32.19 #K
Rdry = 287.0597 #J K^-1 kg^-1
Rvat = 461.5250 #J K^-1 kg^-1
g = 9.80665 #m/s²
M =  0.0289644 #kg/mol Molar mass of air
R = 8.3144598 #J/(mol*K) universal gas constant
lapse_rate = 0.0065 #K/m
T0 = 273.16 #K

def select_nearest_latlon():
    era5_gp = salem.open_xr_dataset(data_path+"ERA5_HMA_15N50N_60E125E_1999_2021_z.nc")
    #Assumes files are in same projecton and all ERA5 files have same grid
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

#Heavily influenced by https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal
def get_glacier_elev(dem):
    gla_shp = salem.read_shapefile(glacier_outline)
    geoms = gla_shp.geometry.values
    geometry = geoms[0]
    geoms = [mapping(geoms[0])]
    with rasterio.open(dem) as src:
        out_image, out_transform = mask(src,geoms, crop=True)
    no_data=src.nodata
    #for ALOS = 0
    no_data = 0
    data = out_image[0,:,:]
    elev = np.extract(data != no_data, data)
    elev = elev[elev != 0]
    mean_alt = np.nanmean(elev)
    return mean_alt

mean_glacier_alt = get_glacier_elev(dem)
print(mean_glacier_alt)

#Load geopotential height
era5_gp = salem.open_xr_dataset(data_path+"ERA5_HMA_15N50N_60E125E_1999_2021_z.nc")
era5_gp = era5_gp.sel(latitude=lat, longitude=lon, method='nearest')

df = era5_gp.to_dataframe().reset_index()
df.drop(['latitude','longitude'], axis=1, inplace=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df['HGT'] = df['z'] / g 
print(df)

#lapse rates from surrounding ERA5 grid cells?
#from .. import .. -> extra script doing that!
lr_t2m = -0.0061 #K/m
lr_d2m = -0.005
#Values should be same, but just in case
era5_alt = np.nanmean(df['HGT'])
height_diff = mean_glacier_alt - era5_alt

for fp in pathlib.Path(data_path).glob('ERA5_HMA*1999_2021*.nc'):
    var = str(fp.stem).split('_')[-1]
    print(var)
    if var != "z":
        ds = xr.open_dataset(fp)
        ds = ds.sel(latitude=lat, longitude=lon, method='nearest')
        df[var] = ds[var].values.flatten()
print(df)

## Adjust temperature according to elevation ##
df['t2m_scaled'] = df['t2m'] + lr_t2m * height_diff
df['d2m_scaled'] = df['d2m'] + lr_d2m * height_diff

## Calculate windspeed ##
df['U10'] = np.sqrt((df['u10']**2) + (df['v10']**2))
#roughness length values, cf. literature mean of firn and snow 2.12 mm / 1000 = 0.00212m
z0 = 2.12/1000
df['U2'] = df['U10'] * ((np.log(2/z0) / np.log(10/z0)))

## Calculate relative humidity ##
#see documentation
df['esat_d2m'] = a1 * np.exp(a3*((df['d2m_scaled']-273.16)/(df['d2m_scaled']-a4)))
df['esat_t2m'] = a1 * np.exp(a3*((df['t2m_scaled']-273.16)/(df['t2m_scaled']-a4)))
df['RH2'] = 100*df['esat_d2m']/df['esat_t2m']
#ensure range between 0 and 100
df['RH2'].loc[df['RH2'] > 100] = 100
df['RH2'].loc[df['RH2'] < 0] = 0


## Calculate pressure at elevation ##
#sp in pascal, convert to hPa
df['sp'] = df['sp']/100
#barometric equation
df['PRES'] = df['sp'] * (1-(-1*lr_t2m*height_diff)/df['t2m']) ** ((g*M)/(R*-1*lr_t2m))

##Adjust accumulated variables of ERA5##
df['tp'] = 1000*df['tp'] #mm from m
df['ssrd'] = df['ssrd']/3600
df['strd'] = df['strd']/3600
#inc. sw rad. can't be below 0
df['ssrd'].loc[df['ssrd'] < 0] = 0 

#Add 5 hours for local time (COSIPY requires this)
df.reset_index(inplace=True)
df['TIMESTAMP'] = df['time']+pd.DateOffset(hours=6) #6 or 5 hours? I think it's 6 

#create final datasets
glac_forc = df.drop(['time','HGT','z', 'sf', 'u10', 'v10', 'strd','t2m','d2m','d2m_scaled','esat_d2m','esat_t2m','U10','sp'], axis=1)
glac_forc.rename(columns={'ssrd':'G','tp':'RRR','tcc':'N','t2m_scaled':'T2'}, inplace=True)
glac_forc.set_index('TIMESTAMP', inplace=True)
print(glac_forc)
fig = glac_forc.plot(subplots=True, figsize=(16,12))
plt.savefig("g.png")
glac_forc.reset_index(inplace=True)
glac_forc.to_csv("Abramov_ERA5_1999_2021.csv")

'''
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Create 2D input file from csv file.')
    parser.add_argument('-c', '-csv_file', dest='csv_file', help='Csv file(see readme for file convention)')
    parser.add_argument('-o', '-cosipy_file', dest='cosipy_file', help='Name of the resulting COSIPY file')
    parser.add_argument('-s', '-static_file', dest='static_file', help='Static file containing DEM, Slope etc.')
    parser.add_argument('-b', '-start_date', dest='start_date', help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', help='End date')
    parser.add_argument('-xl', '-xl', dest='xl', type=float, const=None, help='left longitude value of the subset')
    parser.add_argument('-xr', '-xr', dest='xr', type=float, const=None, help='right longitude value of the subset')
    parser.add_argument('-yl', '-yl', dest='yl', type=float, const=None, help='lower latitude value of the subset')
    parser.add_argument('-yu', '-yu', dest='yu', type=float, const=None, help='upper latitude value of the subset')

    args = parser.parse_args()
    if point_model:
        create_1D_input(args.csv_file, args.cosipy_file, args.static_file, args.start_date, args.end_date) 
    else:
        create_2D_input(args.csv_file, args.cosipy_file, args.static_file, args.start_date, args.end_date, args.xl, args.xr, args.yl, args.yu) 
'''
