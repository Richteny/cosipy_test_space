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

from utilities.aws2cosipy.crop_file_to_glacier import bbox_2d_array

def create_corrected_G(cs_file, cosipy_file, static_file, start_date, end_date, x0=None, x1=None, y0=None, y1=None):
    """ This function creates an input dataset from an offered csv file with input point data
        Here you need to define how to interpolate the data.

        Please note, there should be only one header line in the file.

        Latest update: 
        Tobias Sauter 07.07.2019
	Anselm 01.07.2020
	Franziska Temme 03.08.2021
	"""

    print('-------------------------------------------')
    print('Create input \n')
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
    # Aggregate data to selected value
    #-----------------------------------
    if aggregate:
        if ((N_var in df) and (RRR_var in df) and (LWin_var in df) and (SNOWFALL_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df) and (RRR_var in df) and (LWin_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', LWin_var:'mean'})

        elif ((N_var in df) and (RRR_var in df) and (SNOWFALL_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum', SNOWFALL_var:'sum'})

        elif ((N_var in df) and (LWin_var in df) and (SNOWFALL_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((RRR_var in df) and (LWin_var in df) and (SNOWFALL_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', RRR_var:'sum',
                LWin_var:'mean', SNOWFALL_var:'sum'})

        elif ((N_var in df) and (RRR_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                RRR_var:'sum'})

        elif ((N_var in df) and (SNOWFALL_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', N_var:'mean',
                SNOWFALL_var:'sum'})

        elif ((RRR_var in df) and (LWin_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', RRR_var:'sum',
                LWin_var:'mean'})

        elif ((LWin_var in df) and (SNOWFALL_var in df)):
            df = df.resample(aggregation_step).agg({PRES_var:'mean', T2_var:'mean', RH2_var:'mean', G_var:'mean', U2_var:'mean', LWin_var:'mean',
                SNOWFALL_var:'sum'})

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
        #Crop ds file to minimum extent
        dic_attrs= {'HGT': ('HGT', 'm', 'Elevation'),
                    'ASPECT': ('ASPECT', 'degrees', 'Aspect of slope'),
                    'SLOPE': ('SLOPE', 'degrees', 'Terrain slope'),
                    'MASK': ('MASK', 'boolean', 'Glacier mask'),
                    'T2': ('T2', 'K', 'Temperature at 2 m'),
                    'RH2': ('RH2', '%', 'Relative humidity at 2 m'),
                    'U2': ('U2', 'm s\u207b\xb9', 'Wind velocity at 2 m'),
                    'G': ('G', 'W m\u207b\xb2', 'Incoming shortwave radiation'),
                    'PRES': ('PRES', 'hPa', 'Atmospheric Pressure'),
                    'N_Points': ('N_Points', 'count','Number of Points in each bin'),
                    'RRR': ('RRR', 'mm', 'Total precipitation (liquid+solid)'),
                    'SNOWFALL': ('SNOWFALL', 'm', 'Snowfall'),
                    'LWin': ('LWin', 'W m\u207b\xb2', 'Incoming longwave radiation'),
                    'N': ('N', '%', 'Cloud cover fraction')}

        print(ds.ASPECT.shape)
        dso_mod = xr.Dataset()
        for var in list(ds.variables):
            print(var)
            arr = bbox_2d_array(ds.MASK.values, ds[var].values, var)
            if var in ['lat','latitude','lon','longitude','time','Time']:
                if var == "lat" or var == "latitude":
                    dso_mod.coords['lat'] = arr
                    dso_mod.lat.attrs['standard_name'] = 'lat'
                    dso_mod.lat.attrs['long_name'] = 'latitude'
                    dso_mod.lat.attrs['units'] = 'degrees_north'
                    print(dso_mod.lat, "Success Lat.")
                elif var == 'lon' or var == 'longitude':
                    dso_mod.coords['lon'] = arr
                    dso_mod.lon.attrs['standard_name'] = 'lon'
                    dso_mod.lon.attrs['long_name'] = 'longitude'
                    dso_mod.lon.attrs['units'] = 'degrees_east'
                    print(dso_mod.lon, "Success Lon.")
                else:
                    dso_mod.coords['time'] = arr
                    print("Success time.")
            elif var in ['HGT','ASPECT','SLOPE','MASK','N_Points']:
                print(var)
                add_variable_along_latlon(dso_mod, arr, dic_attrs[var][0], dic_attrs[var][1], dic_attrs[var][2])
                print(var, "success.")
            else:
                add_variable_along_timelatlon(dso_mod, arr, dic_attrs[var][0], dic_attrs[var][1], dic_attrs[var][2])
                print(var, "also success.")
        print(ds.ASPECT.shape)
        print(dso_mod.ASPECT.shape)
        ds = dso_mod
        dso = ds
        dso.coords['time'] = df.index.values

    #-----------------------------------
    # Order variables
    #-----------------------------------
    df[T2_var] = df[T2_var].apply(pd.to_numeric, errors='coerce')
    df[RH2_var] = df[RH2_var].apply(pd.to_numeric, errors='coerce')
    df[U2_var] = df[U2_var].apply(pd.to_numeric, errors='coerce')
    df[G_var] = df[G_var].apply(pd.to_numeric, errors='coerce')
    df[PRES_var] = df[PRES_var].apply(pd.to_numeric, errors='coerce')

    #if (PRES_var not in df):
    #    df[PRES_var] = 660.00

    if (RRR_var in df):
        df[RRR_var] = df[RRR_var].apply(pd.to_numeric, errors='coerce')

    if (LWin_var not in df and N_var not in df):
        print("ERROR no longwave incoming or cloud cover data")
        sys.exit()

    elif (LWin_var in df):
        df[LWin_var] = df[LWin_var].apply(pd.to_numeric, errors='coerce')
        print("LWin in data")

    elif (N_var in df):
        df[N_var] = df[N_var].apply(pd.to_numeric, errors='coerce')

    if (SNOWFALL_var in df):
        df[SNOWFALL_var] = df[SNOWFALL_var].apply(pd.to_numeric, errors='coerce')

    #-----------------------------------
    # Get values from file
    #-----------------------------------
    RH2 = df[RH2_var]       # Relative humidity
    U2 = df[U2_var]         # Wind velocity
    G = df[G_var]           # Incoming shortwave radiation
    PRES = df[PRES_var]     # Pressure

    if (in_K):
        T2 = df[T2_var].values         # Temperature
    else:
        T2 = df[T2_var].values + 273.16      

    if np.nanmax(T2) > 373.16:
        print('Maximum temperature is: %s K please check the input temperature' % (np.nanmax(T2)))
        sys.exit()
    elif np.nanmin(T2) < 173.16:
        print('Minimum temperature is: %s K please check the input temperature' % (np.nanmin(T2)))
        sys.exit()

    #-----------------------------------
    # Create numpy arrays for the 2D fields
    #-----------------------------------
    #Fields too large at 30m res., need to crop to only glacier-extent  ?
    #Crop to glacier extent only and see from there
    
 
    T_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    T_interp_rad = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)]) #new
    RH_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    U_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])
    G_interp = np.full([len(dso.time), len(ds.lat), len(ds.lon)], np.nan)
    P_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if (RRR_var in df):
        RRR = df[RRR_var]       # Precipitation
        RRR_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(SNOWFALL_var in df):
        SNOWFALL = df[SNOWFALL_var]      # Incoming longwave radiation
        SNOWFALL_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(LWin_var in df):
        LW = df[LWin_var]      # Incoming longwave radiation
        LW_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    if(N_var in df):
        N = df[N_var]        # Cloud cover fraction
        N_interp = np.zeros([len(dso.time), len(ds.lat), len(ds.lon)])

    #-----------------------------------
    # Interpolate point data to grid 
    #-----------------------------------
    print('Interpolate CR file to grid')
   
    # Interpolate data (T, RH, RRR, U)  to grid using lapse rates
    rad_tlapse = -0.0065 #new to have default lapse rate for rad. scheme K per m 
    for t in range(len(dso.time)):
        T_interp[t,:,:] = (T2[t]) + (ds.HGT.values-stationAlt)*lapse_T
        T_interp_rad[t,:,:] = (T2[t]) + (ds.HGT.values-stationAlt)*rad_tlapse
        RH_interp[t,:,:] = RH2[t] + (ds.HGT.values-stationAlt)*lapse_RH
        U_interp[t,:,:] = U2[t]

        # Interpolate pressure using the barometric equation
        SLP = PRES[t]/np.power((1-(0.0065*stationAlt)/(288.15)), 5.255)
        P_interp[t,:,:] = SLP * np.power((1-(0.0065*ds.HGT.values)/(288.15)), 5.255)

        if (RRR_var in df):
            RRR_interp[t,:,:] = np.maximum(RRR[t] + (ds.HGT.values-stationAlt)*lapse_RRR, 0.0)
        
        if (SNOWFALL_var in df):
            SNOWFALL_interp[t, :, :] = SNOWFALL[t] + (ds.HGT.values-stationAlt)*lapse_SNOWFALL

        if(LWin_var in df):
            LW_interp[t,:,:] = LW[t]
        
        if(N_var in df):
            N_interp[t,:,:] = N[t]


    print(('Number of glacier cells: %i') % (np.count_nonzero(~np.isnan(ds['MASK'].values))))
    print(('Number of glacier cells: %i') % (np.nansum(ds['MASK'].values)))

    # Auxiliary variables
    mask = ds.MASK.values
    hgt = ds.HGT.values
    slope = ds.SLOPE.values
    aspect = ds.ASPECT.values
    lats = ds.lat.values
    lons = ds.lon.values
    sw = G.values

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
                    if (mask[i, j] == 1) or (mask[i, j] == 0): #added this to allow for regridding
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

            f = nc.Dataset('../../data/static/LUT_Rad.nc', 'w')
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
        for t in range(len(dso.time)):
            doy = df.index[t].dayofyear
            hour = df.index[t].hour
            G_interp[t, :, :] = calcRad(solPars, timeCorr, doy, hour, stationLat, T_interp_rad[t, ::-1, :], P_interp[t, ::-1, :], RH_interp[t, ::-1, :], N_interp[t, ::-1, :], np.flipud(hgt), np.flipud(mask), np.flipud(slope), np.flipud(aspect), shad1yr, svf, dtstep, tcart)
            #changed t_interp field bc with calibration we assign 0 lapse rate by default.
        # Change aspect to south == 0, east == negative, west == positive
        aspect2 = ds['ASPECT'].values - 180.0
        ds['ASPECT'] = (('lat', 'lon'), aspect2)

    else:
        print('Error! Radiation module not available.\nAvailable options are: Wohlfahrt2016, Moelg2009, none.')
        sys.exit()

    #-----------------------------------
    # Check bounds for relative humidity 
    #-----------------------------------
    RH_interp[RH_interp > 100.0] = 100.0
    RH_interp[RH_interp < 0.0] = 0.1

    #-----------------------------------
    # Add variables to file 
    #-----------------------------------
    add_variable_along_latlon(dso, ds.HGT.values, 'HGT', 'm', 'Elevation')
    add_variable_along_latlon(dso, ds.ASPECT.values, 'ASPECT', 'degrees', 'Aspect of slope')
    add_variable_along_latlon(dso, ds.SLOPE.values, 'SLOPE', 'degrees', 'Terrain slope')
    add_variable_along_latlon(dso, ds.MASK.values, 'MASK', 'boolean', 'Glacier mask')
    #add_variable_along_timelatlon(dso, T_interp, 'T2', 'K', 'Temperature at 2 m')
    #add_variable_along_timelatlon(dso, RH_interp, 'RH2', '%', 'Relative humidity at 2 m')
    #add_variable_along_timelatlon(dso, U_interp, 'U2', 'm s\u207b\xb9', 'Wind velocity at 2 m')
    add_variable_along_timelatlon(dso, G_interp, 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
    #add_variable_along_timelatlon(dso, P_interp, 'PRES', 'hPa', 'Atmospheric Pressure')
    
    #if ELEV_model:
    #    add_variable_along_latlon(dso, ds.N_Points.values, 'N_Points', 'count','Number of Points in each bin')

    #if (RRR_var in df):
    #    add_variable_along_timelatlon(dso, RRR_interp, 'RRR', 'mm', 'Total precipitation (liquid+solid)')
    
    #if(SNOWFALL_var in df):
    #    add_variable_along_timelatlon(dso, SNOWFALL_interp, 'SNOWFALL', 'm', 'Snowfall')

    #if(LWin_var in df):
    #    add_variable_along_timelatlon(dso, LW_interp, 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')
    #if(N_var in df):
    #    add_variable_along_timelatlon(dso, N_interp, 'N', '%', 'Cloud cover fraction')

    encoding = dict()
    # for var in IO.get_result().data_vars:
    #     dataMin = IO.get_result()[var].min(skipna=True).values
    #     dataMax = IO.get_result()[var].max(skipna=True).values
    #
    #     dtype = 'int16'
    #     FillValue = -9999
    #     scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
    #     encoding[var] = dict(zlib=True, complevel=2, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)

    #-----------------------------------
    # Write file to disc 
    #-----------------------------------
    #dso.to_netcdf(cosipy_file, encoding=encoding)
    check_for_nan(dso)
    dso.to_netcdf(cosipy_file)

    print('Input file created \n')
    print('-------------------------------------------')

    #-----------------------------------
    # Do some checks
    #-----------------------------------
    #check(dso.T2,316.16,223.16)
    #check(dso.RH2,100.0,0.0)
    #check(dso.U2, 50.0, 0.0)
    check(dso.G,1600.0,0.0)
    #check(dso.PRES,1080.0,200.0)

    #if(RRR_var in df):
    #    check(dso.RRR,25.0,0.0)

    #if (SNOWFALL_var in df):
    #    check(dso.SNOWFALL, 0.05, 0.0)

    #if (LWin_var in df):
    #    check(dso.LWin, 400, 0.0)

    #if (N_var in df):
    #    check(dso.N, 1.0, 0.0)

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
    create_corrected_G(args.csv_file, args.cosipy_file, args.static_file, args.start_date, args.end_date, args.xl, args.xr, args.yl, args.yu) 