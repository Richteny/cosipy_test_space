#!/usr/bin/env python

"""
    This is the main code file of the 'COupled Snowpack and Ice surface energy
    and MAss balance glacier model in Python' (COSIPY). The model was initially written by
    Tobias Sauter. The version is constantly under development by a core developer team.
    
    Core developer team:

    Tobias Sauter
    Anselm Arndt

    You are allowed to use and modify this code in a noncommercial manner and by
    appropriately citing the above mentioned developers.

    The code is available on github. https://github.com/cryotools/cosipy

    For more information read the README and see https://cryo-tools.org/

    The model is written in Python 3.6.3 and is tested on Anaconda3-4.4.7 64-bit.

    Correspondence: tobias.sauter@fau.de

"""
import os
from datetime import datetime
from itertools import product
import itertools

import logging
import yaml

from config import *
from slurm_config import *
from cosipy.cpkernel.cosipy_core import * 
from cosipy.cpkernel.io import *

from distributed import Client, LocalCluster
from dask import compute, delayed
import dask as da
from dask.diagnostics import ProgressBar
from dask.distributed import progress, wait, as_completed
import dask
from tornado import gen
from dask_jobqueue import SLURMCluster

import scipy
import cProfile
#Load constants for default function values
from constants import *
from numba import njit


def main(lr_T=-0.0065, lr_RRR=0, lr_RH=0, RRR_factor=mult_factor_RRR, alb_ice=albedo_ice,
         alb_snow=albedo_fresh_snow,alb_firn=albedo_firn,albedo_aging=albedo_mod_snow_aging,
         albedo_depth=albedo_mod_snow_depth, count=""):

    start_logging()
    times = datetime.now()
    #Initialise dictionary and load Spotpy Params#
    opt_dict = dict()
    opt_dict['mult_factor_RRR'] = RRR_factor
    opt_dict['albedo_ice'] = alb_ice
    opt_dict['albedo_fresh_snow'] = alb_snow
    opt_dict['albedo_firn'] = alb_firn
    opt_dict['albedo_mod_snow_aging'] = albedo_aging
    opt_dict['albedo_mod_snow_depth'] = albedo_depth
    lapse_T = float(lr_T)
    lapse_RRR = float(lr_RRR)
    lapse_RH = float(lr_RH)
    print("Time required to load in opt_dic: ", datetime.now()-times)
    #additionally initial snowheight constant, snow_layer heights, temperature bottom, albedo aging and depth
    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------
    #setup IO with new values from dictionary 
    times = datetime.now()
    IO = IOClass(opt_dict=opt_dict)
    start_time = datetime.now() 
    if (restart == True):
        DATA = IO.create_data_file(suffix="_num{}_lrT_{}_lrRRR_{}_prcp_{}".format(count, round(abs(lapse_T),7),round(lapse_RRR,7),round(opt_dict['mult_factor_RRR'],5)))
    else:
        DATA = IO.create_data_file()
    # Create global result and restart datasets
    RESULT = IO.create_result_file() 
    RESTART = IO.create_restart_file()
    print("Time required to init IO, DATA, RESULT, RESTART: ", datetime.now()-times)
    # Auxiliary variables for futures
    futures= []
    #adjust lapse rates
    print("#--------------------------------------#") 
    print("\nStarting run with lapse rates:", lapse_T, "and:", lapse_RRR) 
    print("\nAlbedo ice, snow and firn:", opt_dict['albedo_ice'],",",opt_dict['albedo_fresh_snow'],"and", opt_dict['albedo_firn'])
    print("\nRRR mult factor is:", opt_dict['mult_factor_RRR'])
    print("\n#--------------------------------------#")
    
    start2 = datetime.now()
    t2 = DATA.T2.values
    rh2 = DATA.RH2.values
    rrr = DATA.RRR.values
    hgt = DATA.HGT.values
    print(np.nanmean(t2))
    t2, rh2, rrr = online_lapse_rate(t2,rh2,rrr,hgt,lapse_T,lapse_RH, lapse_RRR)
    print(np.nanmean(t2))
    print("Assigning values back to DATA")
    DATA['T2'] = (('time','lat','lon'), t2)
    DATA['RH2'] = (('time','lat','lon'), rh2)
    DATA['RRR'] = (('time','lat','lon'), rrr)
    print(np.nanmean(DATA.T2.values))
    print("Seconds needed for lapse rate:", datetime.now()-start2)
    #-----------------------------------------------
    # Create a client for distributed calculations
    #-----------------------------------------------
    if (slurm_use):
        with SLURMCluster(scheduler_port=port, cores=cores, processes=processes, memory=memory, shebang=shebang, name=name, job_extra=slurm_parameters, local_directory='logs/dask-worker-space') as cluster:
            cluster.scale(processes * nodes)   
            print(cluster.job_script())
            print("You are using SLURM!\n")
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures, opt_dict=opt_dict)

    else:
        with LocalCluster(scheduler_port=local_port, n_workers=workers, local_dir='logs/dask-worker-space', threads_per_worker=1, silence_logs=True) as cluster:
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures,opt_dict=opt_dict)

    print('\n')
    print('--------------------------------------------------------------')
    print('Write results ...')
    print('-------------------------------------------------------------- \n')
    start_writing = datetime.now()

    #-----------------------------------------------
    # Write results and restart files
    #-----------------------------------------------
    timestamp = pd.to_datetime(str(IO.get_restart().time.values)).strftime('%Y-%m-%dT%H-%M')
   
    encoding = dict()
    for var in IO.get_result().data_vars:
        dataMin = IO.get_result()[var].min(skipna=True).values
        dataMax = IO.get_result()[var].max(skipna=True).values
        dtype = 'int16'
        FillValue = -9999 
        scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
        #encoding[var] = dict(zlib=True, complevel=compression_level, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)
        encoding[var] = dict(zlib=True, complevel=compression_level)
                    
    results_output_name = output_netcdf.split('.nc')[0]+'_num{}_lrT_{}_lrRRR_{}_prcp_{}.nc'.format(count, round(abs(lapse_T),7), round(lapse_RRR,7),round(opt_dict['mult_factor_RRR'],5))  
    IO.get_result().to_netcdf(os.path.join(data_path,'output',results_output_name), encoding=encoding, mode = 'w')
    
    encoding = dict()
    for var in IO.get_restart().data_vars:
        dataMin = IO.get_restart()[var].min(skipna=True).values
        dataMax = IO.get_restart()[var].max(skipna=True).values
        dtype = 'int16'
        FillValue = -9999 
        scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
        #encoding[var] = dict(zlib=True, complevel=compression_level, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)
        encoding[var] = dict(zlib=True, complevel=compression_level)
                        
    IO.get_restart().to_netcdf(os.path.join(data_path,'restart','restart_'+timestamp+'_num{}_lrT_{}_lrRRR_{}_prcp_{}.nc'.format(count,round(abs(lapse_T),7), round(lapse_RRR,7),round(opt_dict['mult_factor_RRR'],5))), encoding=encoding)
    
    #----------------------------------------------
    # Implement TSL Extraction
    #----------------------------------------------
    #if (restart == True) and (merge == True):
    #    print("Trying to concatenate files. Requires some time.")
    #    #Get name of files 
    #    previous_output_name = results_output_name.replace(time_start_str, time_start_old_file).replace(time_end_str, time_start_str)
    #    merged_output_name = results_output_name.replace(time_start_str, time_start_old_file)
    #    print("Merging with :", previous_output_name)
    #    previous_output = xr.open_dataset(os.path.join(data_path,'output',previous_output_name))
        #Get variables to concat on
    #    list_vars = list(IO.get_result().keys())
    #    [list_vars.remove(x) for x in ['HGT','SLOPE','ASPECT','MASK','MB']]
        #To prevent OOM-Kill Event split into multiple datasets and add variable
        #sub_lists = [list_vars[i:i+2] for i in range(0, len(list_vars),2)]
        #do subset to only those variables for now to avoid memory error
    #    list_vars = [x for x in list_vars if x in ['surfM','surfMB']]
    #    print(list_vars)
    #    ds_merged = xr.concat([previous_output[['MB','SNOWHEIGHT']],IO.get_result()[['MB','SNOWHEIGHT']]], dim='time')
    #    for var in list_vars:
    #        print(var)
            #this produces memory error sometimes, why?
            #Reconstruct by hand?
    #        var_concat = np.concatenate((previous_output[var].values,IO.get_result()[var].values))
    #        ds_merged[var] = (('time','lat','lon'), var_concat)
    #        ds_merged[var].attrs['units'] = IO.get_result()[var].attrs['units']
    #        ds_merged[var].attrs['long_name'] = IO.get_result()[var].attrs['long_name']
    #        ds_merged[var].encoding['_FillValue'] = -9999
    #        del var_concat
    #    print("Part 1/2 of concat done.")
                        
    #    for var in ['HGT','MASK','SLOPE','ASPECT']:
    #        ds_merged[var] = IO.get_result()[var]
    #        print("Part 2/2 of concat done.")
    #        ds_merged.to_netcdf(os.path.join(data_path,'output',merged_output_name))    
    #        print("Concat successful.")
    times = datetime.now()
    if tsl_evaluation is True:
        print("Starting TSL eval.")
        #times = datetime.now()
        tsla_observations = pd.read_csv(tsl_data_file)
        
        if (restart == True) and (merge == True):
            tsl_csv_name = 'tsla_'+merged_output_name.split('.nc')[0].lower()+'.csv'
            resampled_out = resample_output(ds_merged)
            tsl_out = calculate_tsl(resampled_out, min_snowheight)
            tsla_stats = eval_tsl(tsla_observations, tsl_out, time_col_obs, tsla_col_obs)
            print("TSLA Observed vs Modelled RMSE: " + str(tsla_stats[0]) + "; R-squared: " + str(tsla_stats[1]))
            tsl_out.to_csv(os.path.join(data_path,'output',tsl_csv_name))
            del ds_merged
        else:
            tsl_csv_name = 'tsla_'+results_output_name.split('.nc')[0].lower()+'.csv'    
            tsla_observations = pd.read_csv(tsl_data_file)
            #resampled_out = resample_output(IO.get_result())
            for var in ['SNOWHEIGHT','HGT']:
                vals = resample_array_style(IO.get_result()[var].values)
            print(IO.get_result()['HGT'])
            resampled_out = construct_resampled_ds(IO.get_result(),vals)
            resampled_out['HGT'] = (('lat','lon'), IO.get_result()['HGT'])

            tsl_out = calculate_tsl(resampled_out, min_snowheight)
            tsla_stats = eval_tsl(tsla_observations,tsl_out, time_col_obs, tsla_col_obs)
            print("TSLA Observed vs. Modelled RMSE: " + str(tsla_stats[0])+ "; R-squared: " + str(tsla_stats[1]))
            tsl_out.to_csv(os.path.join(data_path,'output',tsl_csv_name))
        print("Time required for TSL EVAL: ", datetime.now()-times)
    #-----------------------------------------------
    # Stop time measurement
    #-----------------------------------------------
    duration_run = datetime.now() - start_time
    duration_run_writing = datetime.now() - start_writing

    #-----------------------------------------------
    # Print out some information
    #-----------------------------------------------
    print("\t Time required to write restart and output files: %4g minutes %2g seconds \n" % (duration_run_writing.total_seconds()//60.0,duration_run_writing.total_seconds()%60.0))
    print("\t Total run duration: %4g minutes %2g seconds \n" % (duration_run.total_seconds()//60.0,duration_run.total_seconds()%60.0))
    print('--------------------------------------------------------------')
    print('\t SIMULATION WAS SUCCESSFUL')
    print('--------------------------------------------------------------')

    return tsl_out
    

def run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures, opt_dict=None):

    with Client(cluster) as client:
        print('--------------------------------------------------------------')
        print('\t Starting clients and submit jobs ... \n')
        print('-------------------------------------------------------------- \n')

        print(cluster)
        print(client)

        # Get dimensions of the whole domain
        ny = DATA.dims[northing]
        nx = DATA.dims[easting]

        cp = cProfile.Profile()

        # Get some information about the cluster/nodes
        total_grid_points = DATA.dims[northing]*DATA.dims[easting]
        if slurm_use is True:
            total_cores = processes*nodes
            points_per_core = total_grid_points // total_cores
            print(total_grid_points, total_cores, points_per_core)

        # Check if evaluation is selected:
        if stake_evaluation is True:
            # Read stake data (data must be given as cumulative changes)
            df_stakes_loc = pd.read_csv(stakes_loc_file, delimiter='\t', na_values='-9999')
            df_stakes_data = pd.read_csv(stakes_data_file, delimiter='\t', index_col='TIMESTAMP', na_values='-9999')
            df_stakes_data.index = pd.to_datetime(df_stakes_data.index)

            # Uncomment, if stake data is given as changes between measurements
            # df_stakes_data = df_stakes_data.cumsum(axis=0)

            # Init dataframes to store evaluation statistics
            df_stat = pd.DataFrame()
            df_val = df_stakes_data.copy()

            # reshape and stack coordinates
            if WRF:
                coords = np.column_stack((DATA.lat.values.ravel(), DATA.lon.values.ravel()))
            else:
                # in case lat/lon are 1D coordinates
                lons, lats = np.meshgrid(DATA.lon,DATA.lat)
                coords = np.column_stack((lats.ravel(),lons.ravel()))

            # construct KD-tree, in order to get closes grid cell
            ground_pixel_tree = scipy.spatial.cKDTree(transform_coordinates(coords))

            # Check for stake data
            stakes_list = []
            for index, row in df_stakes_loc.iterrows():
                index = ground_pixel_tree.query(transform_coordinates((row['lat'], row['lon'])))
                if WRF:
                    index = np.unravel_index(index[1], DATA.lat.shape)
                else:
                    index = np.unravel_index(index[1], lats.shape)
                stakes_list.append((index[0][0], index[1][0], row['id']))

        else:
            stakes_loc = None
            df_stakes_data = None


        # Distribute data and model to workers
        start_res = datetime.now()
        for y,x in product(range(DATA.dims[northing]),range(DATA.dims[easting])):
            if stake_evaluation is True:
                stake_names = []
                # Check if the grid cell contain stakes and store the stake names in a list
                for idx, (stake_loc_y, stake_loc_x, stake_name) in enumerate(stakes_list):
                    if ((y == stake_loc_y) & (x == stake_loc_x)):
                        stake_names.append(stake_name)
            else:
                stake_names = None

            if WRF is True:
                mask = DATA.MASK.sel(south_north=y, west_east=x)
	        # Provide restart grid if necessary
                if ((mask==1) & (restart==False)):
                    if np.isnan(DATA.sel(south_north=y, west_east=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, stake_names=stake_names, stake_data=df_stakes_data, opt_dict=opt_dict))
                elif ((mask==1) & (restart==True)):
                    if np.isnan(DATA.sel(south_north=y, west_east=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, 
                                             GRID_RESTART=IO.create_grid_restart().sel(south_north=y, west_east=x), 
                                             stake_names=stake_names, stake_data=df_stakes_data, opt_dict=opt_dict))
            else:
                mask = DATA.MASK.isel(lat=y, lon=x)
	        # Provide restart grid if necessary
                if ((mask==1) & (restart==False)):
                    if np.isnan(DATA.isel(lat=y,lon=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.isel(lat=y, lon=x), y, x, stake_names=stake_names, stake_data=df_stakes_data, opt_dict=opt_dict))
                elif ((mask==1) & (restart==True)):
                    if np.isnan(DATA.isel(lat=y,lon=x).to_array()).any():
                        print('ERROR!!!!!!!!!!! There are NaNs in the dataset')
                        sys.exit()
                    futures.append(client.submit(cosipy_core, DATA.isel(lat=y, lon=x), y, x, 
                                             GRID_RESTART=IO.create_grid_restart().isel(lat=y, lon=x), 
                                             stake_names=stake_names, stake_data=df_stakes_data, opt_dict=opt_dict))
        # Finally, do the calculations and print the progress
        #progress(futures)

        #---------------------------------------
        # Guarantee that restart file is closed
        #---------------------------------------
        if (restart==True):
            IO.get_grid_restart().close()
      
        # Create numpy arrays which aggregates all local results
        IO.create_global_result_arrays()

        # Create numpy arrays which aggregates all local results
        IO.create_global_restart_arrays()

        #---------------------------------------
        # Assign local results to global 
        #---------------------------------------
        for future in as_completed(futures):

                # Get the results from the workers
                indY,indX,local_restart,RAIN,SNOWFALL,LWin,LWout,H,LE,B,QRR,MB,surfMB,Q,SNOWHEIGHT,TOTALHEIGHT,TS,ALBEDO,NLAYERS, \
                                ME,intMB,EVAPORATION,SUBLIMATION,CONDENSATION,DEPOSITION,REFREEZE,subM,Z0,surfM,MOL, \
                                LAYER_HEIGHT,LAYER_RHO,LAYER_T,LAYER_LWC,LAYER_CC,LAYER_POROSITY,LAYER_ICE_FRACTION, \
                                LAYER_IRREDUCIBLE_WATER,LAYER_REFREEZE,stake_names,stat,df_eval = future.result()
               
                IO.copy_local_to_global(indY,indX,RAIN,SNOWFALL,LWin,LWout,H,LE,B,QRR,MB,surfMB,Q,SNOWHEIGHT,TOTALHEIGHT,TS,ALBEDO,NLAYERS, \
                                ME,intMB,EVAPORATION,SUBLIMATION,CONDENSATION,DEPOSITION,REFREEZE,subM,Z0,surfM,MOL,LAYER_HEIGHT,LAYER_RHO, \
                                LAYER_T,LAYER_LWC,LAYER_CC,LAYER_POROSITY,LAYER_ICE_FRACTION,LAYER_IRREDUCIBLE_WATER,LAYER_REFREEZE)

                IO.copy_local_restart_to_global(indY,indX,local_restart)

                # Write results to file
                IO.write_results_to_file()
                
                # Write restart data to file
                IO.write_restart_to_file()

                if stake_evaluation is True:
                    # Store evaluation of stake measurements to dataframe
                    stat = stat.rename('rmse')
                    df_stat = pd.concat([df_stat, stat])

                    for i in stake_names:
                        if (obs_type == 'mb'):
                            df_val[i] = df_eval.mb
                        if (obs_type == 'snowheight'):
                            df_val[i] = df_eval.snowheight

        # Measure time
        end_res = datetime.now()-start_res 
        print("\t Time required to do calculations: %4g minutes %2g seconds \n" % (end_res.total_seconds()//60.0,end_res.total_seconds()%60.0))
      
        if stake_evaluation is True:
            # Save the statistics and the mass balance simulations at the stakes to files
            df_stat.to_csv(os.path.join(data_path,'output','stake_statistics.csv'),sep='\t', float_format='%.2f')
            df_val.to_csv(os.path.join(data_path,'output','stake_simulations.csv'),sep='\t', float_format='%.2f')


def start_logging():
    ''' Start the python logging'''

    if os.path.exists('./cosipy.yaml'):
        with open('./cosipy.yaml', 'rt') as f:
            config = yaml.load(f.read(),Loader=yaml.SafeLoader)
        logging.config.dictConfig(config)
    else:
       logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info('COSIPY simulation started')    


def transform_coordinates(coords):
    """ Transform coordinates from geodetic to cartesian
    an array of tuples)
    """
    # WGS 84 reference coordinate system parameters
    A = 6378.137 # major axis [km]   
    E2 = 6.69437999014e-3 # eccentricity squared    
    
    coords = np.asarray(coords).astype(np.float)
                                                      
    # is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])
    
    # convert to radiants
    lat_rad = np.radians(coords[:,0])
    lon_rad = np.radians(coords[:,1]) 
    
    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)
    
    return np.column_stack((x, y, z))


def compute_scale_and_offset(min, max, n):
    # stretch/compress data to the available packed range
    scale_factor = (max - min) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = min + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)

@njit
def online_lapse_rate(t2,rh2,rrr,hgt,lapse_T,lapse_RH,lapse_RRR):
    print(t2.shape)
    for t in range(t2.shape[0]):
        t2[t,:,:] = t2[t,:,:]+ (hgt - station_altitude)*lapse_T
        rh2[t,:,:] = rh2[t,:,:]+ (hgt - station_altitude)*lapse_RH
        rrr[t,:,:] = np.maximum(rrr[t,:,:]+ (hgt - station_altitude)*lapse_RRR, 0.0)
    return t2,rh2,rrr

def construct_resampled_ds(input_ds,vals):
    times = [str(x) for x in input_ds.time.values]
    time_vals = pd.daterange(times[0], times[-2], freq="1d")
    data_vars = {'SNOWHEIGHT':(['time','lat','lon'], [vals],
                               {'units': "m",
                                'long_name': "snowheight"})}
     

    # define coordinates
    coords = {'time': (['time'], [time_vals]),
              'lat': (['lat'], [input_ds.lat.values]),
              'lon': (['lon'], [input_ds.lon.values])}
     
    ds = xr.Dataset(data_vars=data_vars,coords=coords)
    return ds

@gen.coroutine
def close_everything(scheduler):
    yield scheduler.retire_workers(workers=scheduler.workers, close_workers=True)
    yield scheduler.close()



''' MODEL EXECUTION '''
if __name__ == "__main__":
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats=pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats()
    
