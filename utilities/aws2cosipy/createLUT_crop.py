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
import netCDF4 as nc
import time
import dateutil
from itertools import product
import metpy.calc
from metpy.units import units

#np.warnings.filterwarnings('ignore')

sys.path.append('../../')

from utilities.aws2cosipy.aws2cosipyConfig import *
from cosipy.modules.radCor import *

import argparse

def create_LUT_file(cs_file, cosipy_file, static_file, start_date, end_date, x0=None, x1=None, y0=None, y1=None):
    print('-------------------------------------------')
    print('Read input file %s' % (cs_file))

    #-----------------------------------
    # Read data
    #-----------------------------------
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    df = pd.read_csv(cs_file, 
        delimiter=delimiter, index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'],
        na_values='NAN',date_parser=date_parser)

    #-----------------------------------
    # Select time slice
    #-----------------------------------
    if ((start_date != None) & (end_date !=None)): 
        df = df.loc[start_date:end_date]

    #-----------------------------------
    # Load static data
    #-----------------------------------
    print('Read static file %s \n' % (static_file))
    ds = xr.open_dataset(static_file)

    #-----------------------------------
    # Create subset
    #-----------------------------------
    ds = ds.sel(lat=slice(y0,y1), lon=slice(x0,x1))

    if WRF:
        dso = xr.Dataset()
        x, y = np.meshgrid(ds.lon, ds.lat)
        dso.coords['time'] = (('time'), df.index.values)
        dso.coords['lat'] = (('south_north','west_east'), y)
        dso.coords['lon'] = (('south_north','west_east'), x)

    else:
        dso = ds    
        dso.coords['time'] = df.index.values

    # Auxiliary variables
    mask = ds.MASK.values
    # just for regridding set all mask values to 1 to avoid nans
    mask[mask == 0] = 1
    hgt = ds.HGT.values
    slope = ds.SLOPE.values
    aspect = ds.ASPECT.values
    lats = ds.lat.values
    lons = ds.lon.values
    #sw = G.values

    #-----------------------------------
    # Run radiation module 
    #-----------------------------------
    
    if radiationModule == 'Wohlfahrt2016' or radiationModule == 'none':
        print('Run the Radiation Module Wohlfahrt2016 or no Radiation Module.')

        # Change aspect to south==0, east==negative, west==positive
        aspect = ds['ASPECT'].values - 180.0
        ds['ASPECT'] = (('lat', 'lon'), aspect)

        for t in range(len(dso.time)):
            doy = df.index[t].dayofyear
            hour = df.index[t].hour
            for i in range(len(ds.lat)):
                for j in range(len(ds.lon)):
                    if (mask[i, j] == 1):
                        if radiationModule == 'Wohlfahrt2016':
                            G_interp[t, i, j] = np.maximum(0.0, correctRadiation(lats[i], lons[j], timezone_lon, doy, hour, slope[i, j], aspect[i, j], sw[t], zeni_thld))
                        else:
                            G_interp[t, i, j] = sw[t]

    elif radiationModule == 'Moelg2009':
        print('Run the Radiation Module Moelg2009')

        # Calculate solar Parameters
        solPars, timeCorr = solpars(stationLat)

        if LUT == True:
            print('Read in look-up-tables')
            ds_LUT = xr.open_dataset('../../data/static/LUT_Rad.nc')
            shad1yr = ds_LUT.SHADING.values
            svf = ds_LUT.SVF.values

        else:
            print('Build look-up-tables')

            # Sky view factor
            svf = LUTsvf(np.flipud(hgt), np.flipud(mask), np.flipud(slope), np.flipud(aspect), lats[::-1], lons)
            print('Look-up-table sky view factor done')

            # Topographic shading
            shad1yr = LUTshad(solPars, timeCorr, stationLat, np.flipud(hgt), np.flipud(mask), lats[::-1], lons, dtstep, tcart)
            print('Look-up-table topographic shading done')

            # Save look-up tables
            Nt = int(366 * (3600 / dtstep) * 24)  # number of time steps
            Ny = len(lats)  # number of latitudes
            Nx = len(lons)  # number of longitudes

            f = nc.Dataset('../../data/static/LUT_Rad_{}-{}.nc'.format(start_date,end_date), 'w')
            f.createDimension('time', Nt)
            f.createDimension('lat', Ny)
            f.createDimension('lon', Nx)

            LATS = f.createVariable('lat', 'f4', ('lat',))
            LATS.units = 'degree'
            LONS = f.createVariable('lon', 'f4', ('lon',))
            LONS.units = 'degree'

            LATS[:] = lats
            LONS[:] = lons

            shad = f.createVariable('SHADING', float, ('time', 'lat', 'lon'))
            shad.long_name = 'Topographic shading'
            shad[:] = shad1yr

            SVF = f.createVariable('SVF', float, ('lat', 'lon'))
            SVF.long_name = 'sky view factor'
            SVF[:] = svf

            f.close()

        # In both cases, run the radiation model
        #for t in range(len(dso.time)):
         #   doy = df.index[t].dayofyear
         #   hour = df.index[t].hour
         #   G_interp[t, :, :] = calcRad(solPars, timeCorr, doy, hour, stationLat, T_interp[t, ::-1, :], P_interp[t, ::-1, :], RH_interp[t, ::-1, :], N_interp[t, ::-1, :], np.flipud(hgt), np.flipud(mask), np.flipud(slope), np.flipud(aspect), shad1yr, svf, dtstep, tcart)

        # Change aspect to south == 0, east == negative, west == positive
        #aspect2 = ds['ASPECT'].values - 180.0
        #ds['ASPECT'] = (('lat', 'lon'), aspect2)

    else:
        print('Error! Radiation module not available.\nAvailable options are: Wohlfahrt2016, Moelg2009, none.')
        sys.exit()

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

def add_variable_along_timelatlon_point(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('time','lat','lon'), np.reshape(var,(len(var),1,1)))
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def add_variable_along_point(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('lat','lon'), np.reshape(var,(1,1)))
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def check(field, max, min):
    '''Check the validity of the input data '''
    if np.nanmax(field) > max or np.nanmin(field) < min:
        print('\n\nWARNING! Please check the data, its seems they are out of a reasonable range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(field.name), np.nanmax(field), np.nanmin(field)))
     
def check_for_nan(ds):
    if WRF is True:
        for y,x in product(range(ds.dims['south_north']),range(ds.dims['west_east'])):
            mask = ds.MASK.sel(south_north=y, west_east=x)
            if mask==1:
                if np.isnan(ds.sel(south_north=y, west_east=x).to_array()).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                    sys.exit()
    else:
        for y,x in product(range(ds.dims['lat']),range(ds.dims['lon'])):
            mask = ds.MASK.isel(lat=y, lon=x)
            if mask==1:
                if np.isnan(ds.isel(lat=y, lon=x).to_array()).any():
                    print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                    sys.exit()

def check_for_nan_point(ds):
    if np.isnan(ds.to_array()).any():
        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
        sys.exit()

def compute_scale_and_offset(min, max, n):
    # stretch/compress data to the available packed range
    scale_factor = (max - min) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = min + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)


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
    if createLUT:
        create_LUT_file(args.csv_file, args.cosipy_file, args.static_file, args.start_date, args.end_date) 
    else:
        print("Missing argument to create LUT.") 
