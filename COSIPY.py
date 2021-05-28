#!/usr/bin/env python

"""
This is the main code file of the 'COupled Snowpack and Ice surface
energy and MAss balance glacier model in Python' (COSIPY). The model was
initially written by Tobias Sauter. The version is constantly under
development by a core developer team.

Core developer team:

Tobias Sauter
Anselm Arndt

You are allowed to use and modify this code in a noncommercial manner
and by appropriately citing the above mentioned developers.

The code is available on github. https://github.com/cryotools/cosipy

For more information read the README and see https://cryo-tools.org/

The model is written in Python 3.9 and is tested on Anaconda3-4.4.7 64-bit.

Correspondence: tobias.sauter@fau.de
"""
import cProfile
import logging
import os
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import scipy
import yaml
# from dask import compute, delayed
# from dask.diagnostics import ProgressBar
from dask.distributed import as_completed, progress
from dask_jobqueue import SLURMCluster
from distributed import Client, LocalCluster
# import dask
from tornado import gen

from cosipy.config import Config, SlurmConfig
from cosipy.constants import Constants
from cosipy.cpkernel.cosipy_core import cosipy_core
from cosipy.cpkernel.io import IOClass


def main():
    Config()
    Constants()

    start_logging()

    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------
    IO = IOClass()
    DATA = IO.create_data_file()

    # Create global result and restart datasets
    RESULT = IO.create_result_file()
    RESTART = IO.create_restart_file()

    #----------------------------------------------
    # Calculation - Multithreading using all cores
    #----------------------------------------------

    # Auxiliary variables for futures
    #futures = []

    # Measure time
    start_time = datetime.now()
    if lapse_rate_config:
        #here lapse rate changes
        #if run_multiple_lapse_rates:
        df_tsl_stats = pd.DataFrame(columns={'LR_T2':[],
                                             'LR_RRR':[],
                                             'RMSE':[],
                                             'R2':[],
                                             'MBE':[],
                                             'MAE':[]})
        #check if lapse rate in aws2cosipy - 
        from utilities.aws2cosipy.aws2cosipyConfig import lapse_T, lapse_RRR
        if lapse_T != 0:
            print("Attention! AWS2cosipy contains a temperature lapse rate that is not zero. \nPlease be sure the input file is not constructed with a lapse rate.")
        else:
            print("No temperature lapse rate used in AWS2cosipy.")    

        if lapse_RRR != 0:
            print("Attention! AWS2cosipy contains a precipitation lapse rate that is not zero. \nPlease be sure the input file is not constructed with a lapse rate.")
        else:
            print("No precipitation lapse rate used in AWS2cosipy.") 
    
        for lapse_T in lapse_T_range:
    
            for lapse_RRR in lapse_RRR_range:
                #------------------------------------------
                # Create input and output dataset
                #------------------------------------------ 
                if (restart == True):
                    DATA = IO.create_data_file(suffix="_lrT_{}_lrRRR_{}".format(abs(lapse_T),lapse_RRR))
                else:
                    DATA = IO.create_data_file()
                # Create global result and restart datasets
                RESULT = IO.create_result_file() 
                RESTART = IO.create_restart_file()
                # Auxiliary variables for futures
                futures= []
                #adjust lapse rates 
                print("Starting run with lapse rates:", lapse_T, "and:", lapse_RRR) 
                for t in range(len(DATA.time)):
                    DATA.T2[t,:,:] = DATA.T2[t,:,:]+ (DATA.HGT - station_altitude)*lapse_T
                    DATA.RRR[t,:,:] = np.maximum(DATA.RRR[t,:,:]+ (DATA.HGT - station_altitude)*lapse_RRR, 0.0)
        
    #-----------------------------------------------
    # Create a client for distributed calculations
    #-----------------------------------------------

    if Config.slurm_use:
        SlurmConfig()
        with SLURMCluster(
            job_name=SlurmConfig.name,
            cores=SlurmConfig.cores,
            processes=SlurmConfig.cores,
            memory=SlurmConfig.memory,
            account=SlurmConfig.account,
            job_extra_directives=SlurmConfig.slurm_parameters,
            local_directory=SlurmConfig.local_directory,
        ) as cluster:
            cluster.scale(SlurmConfig.nodes * SlurmConfig.cores)
            print(cluster.job_script())
            print("You are using SLURM!\n")
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

    else:
        with LocalCluster(scheduler_port=Config.local_port, n_workers=Config.workers, local_directory='logs/dask-worker-space', threads_per_worker=1, silence_logs=True) as cluster:
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

    print("\n")
    print_notice(msg="Write results ...")
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
            
                results_output_name = output_netcdf.split('.nc')[0]+'_lrT_{}_lrRRR_{}.nc'.format(abs(lapse_T), lapse_RRR)  
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
                
                IO.get_restart().to_netcdf(os.path.join(data_path,'restart','restart_'+timestamp+'_lrT_{}_lrRRR_{}.nc'.format(abs(lapse_T),lapse_RRR)), encoding=encoding)
    
        #----------------------------------------------
        # Implement TSL Extraction
        #----------------------------------------------
                if (restart == True) and (merge == True):
                    print("Trying to concatenate files. Requires some time.")
                    #Get name of files 
                    previous_output_name = results_output_name.replace(time_start_str, time_start_old_file).replace(time_end_str, time_start_str)
                    merged_output_name = results_output_name.replace(time_start_str, time_start_old_file)
                    print("Merging with :", previous_output_name)
                    previous_output = xr.open_dataset(os.path.join(data_path,'output',previous_output_name))
                    #Get variables to concat on
                    list_vars = list(IO.get_result().keys())
                    [list_vars.remove(x) for x in ['HGT','SLOPE','ASPECT','MASK','MB']]
                    #To prevent OOM-Kill Event split into multiple datasets and add variable
                    #sub_lists = [list_vars[i:i+2] for i in range(0, len(list_vars),2)]
                    #do subset to only those variables for now to avoid memory error
                    list_vars = [x for x in list_vars if x in ['surfM','surfMB']]
                    print(list_vars)
                    ds_merged = xr.concat([previous_output[['MB','SNOWHEIGHT']],IO.get_result()[['MB','SNOWHEIGHT']]], dim='time')
                    for var in list_vars:
                        print(var)
                        #this produces memory error sometimes, why?
                        #Reconstruct by hand?
                        var_concat = np.concatenate((previous_output[var].values,IO.get_result()[var].values))
                        ds_merged[var] = (('time','lat','lon'), var_concat)
                        ds_merged[var].attrs['units'] = IO.get_result()[var].attrs['units']
                        ds_merged[var].attrs['long_name'] = IO.get_result()[var].attrs['long_name']
                        ds_merged[var].encoding['_FillValue'] = -9999
                        del var_concat
                        print("Part 1/2 of concat done.")
                
                    for var in ['HGT','MASK','SLOPE','ASPECT']:
                         ds_merged[var] = IO.get_result()[var]
                    print("Part 2/2 of concat done.")
                    ds_merged.to_netcdf(os.path.join(data_path,'output',merged_output_name))    
                    print("Concat successful.")
                if tsl_evaluation is True:
                    tsla_observations = pd.read_csv(tsl_data_file)
                    if (restart == True) and (merge == True):
                        tsl_csv_name = 'tsla_'+merged_output_name.split('.nc')[0].lower()+'.csv'
                        resampled_out = resample_output(ds_merged)
                        tsl_out = calculate_tsl(resampled_out, min_snowheight)
                        tsla_stats = eval_tsl(tsla_observations, tsl_out)
                        print("TSLA Observed vs Modelled RMSE: " + str(tsla_stats[0]) + "; R-squared: " + str(tsla_stats[1]))
                        tsl_out.to_csv(os.path.join(data_path,'output',tsl_csv_name))
                        df_tsl_stats = df_tsl_stats.append({'LR_T2':lapse_T,
                                                            'LR_RRR':lapse_RRR,
                                                            'RMSE':tsla_stats[0],
                                                            'R2':tsla_stats[1],
                                                            'MBE':tsla_stats[2],
                                                            'MAE':tsla_stats[3]}, ignore_index=True)
                        del ds_merged
                    else:
                        tsl_csv_name = 'tsla_'+results_output_name.split('.nc')[0].lower()+'.csv'    
                        tsla_observations = pd.read_csv(tsl_data_file)
                        resampled_out = resample_output(IO.get_result())
                        tsl_out = calculate_tsl(resampled_out, min_snowheight)     
                        tsla_stats = eval_tsl(tsla_observations,tsl_out)
                        print("TSLA Observed vs. Modelled RMSE: " + str(tsla_stats[0])+ "; R-squared: " + str(tsla_stats[1]))
                        tsl_out.to_csv(os.path.join(data_path,'output',tsl_csv_name))
                        df_tsl_stats = df_tsl_stats.append({'LR_T2':lapse_T,
                                                            'LR_RRR':lapse_RRR,
                                                            'RMSE':tsla_stats[0],
                                                            'R2':tsla_stats[1],
                                                            'MBE':tsla_stats[2],
                                                            'MAE':tsla_stats[3]}, ignore_index=True)
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
         
        df_tsl_stats.to_csv(os.path.join(data_path,'output','tsla_statistics.csv'))

    # if run without lapse rate config    
    else:
        IO = IOClass()
        DATA = IO.create_data_file()
        # Create global result and restart datasets
        RESULT = IO.create_result_file()
        RESTART = IO.create_restart_file()
        futures = []
        start_time = datetime.now()
    #-----------------------------------------------
    # Create a client for distributed calculations
    #-----------------------------------------------
        if (slurm_use):

            with SLURMCluster(scheduler_port=port, cores=cores, processes=processes, memory=memory, shebang=shebang, name=name, job_extra=slurm_parameters, local_directory='logs/dask-worker-space') as cluster:
                cluster.scale(processes * nodes)
                print(cluster.job_script())
                print("You are using SLURM!\n")
                print(cluster)
                run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

        else:
            with LocalCluster(scheduler_port=local_port, n_workers=workers, local_dir='logs/dask-worker-space', threads_per_worker=1, silence_logs=True) as cluster:
                print(cluster)
                run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures)

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
        # dataMin = IO.get_result()[var].min(skipna=True).values
        # dataMax = IO.get_result()[var].max(skipna=True).values
        # dtype = 'int16'
        # FillValue = -9999
        # scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
        #encoding[var] = dict(zlib=True, complevel=compression_level, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)
        encoding[var] = dict(zlib=True, complevel=Config.compression_level)
    output_netcdf = set_output_netcdf_path()
    output_path = create_data_directory(path='output')
    IO.get_result().to_netcdf(os.path.join(output_path,output_netcdf), encoding=encoding, mode='w')

    encoding = dict()
    for var in IO.get_restart().data_vars:
        # dataMin = IO.get_restart()[var].min(skipna=True).values
        # dataMax = IO.get_restart()[var].max(skipna=True).values
        # dtype = 'int16'
        # FillValue = -9999
        # scale_factor, add_offset = compute_scale_and_offset(dataMin, dataMax, 16)
        #encoding[var] = dict(zlib=True, complevel=compression_level, dtype=dtype, scale_factor=scale_factor, add_offset=add_offset, _FillValue=FillValue)
        encoding[var] = dict(zlib=True, complevel=Config.compression_level)

    restart_path = create_data_directory(path='restart')
    IO.get_restart().to_netcdf(os.path.join(restart_path,f'restart_{timestamp}.nc'), encoding=encoding)

    # Stop time measurement
    #-----------------------------------------------
        duration_run = datetime.now() - start_time
        duration_run_writing = datetime.now() - start_writing

    #-----------------------------------------------
    # Print out some information
    #-----------------------------------------------

    get_time_required(
        action="write restart and output files", times=duration_run_writing
    )
    run_time = duration_run.total_seconds()
    print(f"\tTotal run duration: {run_time // 60.0:4g} minutes {run_time % 60.0:2g} seconds\n")
    print_notice(msg="\tSIMULATION WAS SUCCESSFUL")


def run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures):
    Config()
    Constants()

    with Client(cluster) as client:
        print_notice(msg="\tStarting clients and submitting jobs ...")
        print(cluster)
        print(client)

        # Get dimensions of the whole domain
        # ny = DATA.sizes[Config.northing]
        # nx = DATA.sizes[Config.easting]

        # cp = cProfile.Profile()

        # Get some information about the cluster/nodes
        total_grid_points = DATA.sizes[Config.northing]*DATA.sizes[Config.easting]
        if Config.slurm_use:
            total_cores = SlurmConfig.cores * SlurmConfig.nodes
            points_per_core = total_grid_points // total_cores
            print(total_grid_points, total_cores, points_per_core)

        # Check if evaluation is selected:
        if Config.stake_evaluation:
            # Read stake data (data must be given as cumulative changes)
            df_stakes_loc = pd.read_csv(Config.stakes_loc_file, delimiter='\t', na_values='-9999')
            df_stakes_data = pd.read_csv(Config.stakes_data_file, delimiter='\t', index_col='TIMESTAMP', na_values='-9999')
            df_stakes_data.index = pd.to_datetime(df_stakes_data.index)

            # Uncomment, if stake data is given as changes between measurements
            # df_stakes_data = df_stakes_data.cumsum(axis=0)

            # Init dataframes to store evaluation statistics
            df_stat = pd.DataFrame()
            df_val = df_stakes_data.copy()

            # reshape and stack coordinates
            if Config.WRF:
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
                if Config.WRF:
                    index = np.unravel_index(index[1], DATA.lat.shape)
                else:
                    index = np.unravel_index(index[1], lats.shape)
                stakes_list.append((index[0][0], index[1][0], row['id']))

        else:
            stakes_loc = None
            df_stakes_data = None

        # Distribute data and model to workers
        start_res = datetime.now()
        for y,x in product(range(DATA.sizes[Config.northing]),range(DATA.sizes[Config.easting])):
            if Config.stake_evaluation:
                stake_names = []
                # Check if the grid cell contain stakes and store the stake names in a list
                for idx, (stake_loc_y, stake_loc_x, stake_name) in enumerate(stakes_list):
                    if (y == stake_loc_y) and (x == stake_loc_x):
                        stake_names.append(stake_name)
            else:
                stake_names = None

            if Config.WRF:
                mask = DATA.MASK.sel(south_north=y, west_east=x)
                # Provide restart grid if necessary
                if (mask == 1) and (not Config.restart):
                    check_for_nan(data=DATA.sel(south_north=y, west_east=x))
                    futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, stake_names=stake_names, stake_data=df_stakes_data))
                elif (mask == 1) and (Config.restart):
                    check_for_nan(data=DATA.sel(south_north=y, west_east=x))
                    futures.append(
                        client.submit(
                            cosipy_core,
                            DATA.sel(south_north=y, west_east=x),
                            y,
                            x,
                            GRID_RESTART=IO.create_grid_restart().sel(
                                south_north=y,
                                west_east=x,
                            ),
                            stake_names=stake_names,
                            stake_data=df_stakes_data
                        )
                    )
            else:
                mask = DATA.MASK.isel(lat=y, lon=x)
                # Provide restart grid if necessary
                if (mask == 1) and (not Config.restart):
                    check_for_nan(data=DATA.isel(lat=y,lon=x))
                    futures.append(client.submit(cosipy_core, DATA.isel(lat=y, lon=x), y, x, stake_names=stake_names, stake_data=df_stakes_data))
                elif (mask == 1) and (Config.restart):
                    check_for_nan(data=DATA.isel(lat=y,lon=x))
                    futures.append(
                        client.submit(
                            cosipy_core,
                            DATA.isel(lat=y, lon=x),
                            y,
                            x,
                            GRID_RESTART=IO.create_grid_restart().isel(
                                lat=y, lon=x
                            ),
                            stake_names=stake_names,
                            stake_data=df_stakes_data
                        )
                    )
        # Finally, do the calculations and print the progress
        #progress(futures)

        #---------------------------------------
        # Guarantee that restart file is closed
        #---------------------------------------
        if Config.restart:
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
            indY, indX, local_restart, RAIN, SNOWFALL, LWin, LWout, H, LE, B, \
                QRR, MB, surfMB, Q, SNOWHEIGHT, TOTALHEIGHT, TS, ALBEDO, \
                NLAYERS, ME, intMB, EVAPORATION, SUBLIMATION, CONDENSATION, \
                DEPOSITION, REFREEZE, subM, Z0, surfM, new_snow_height, new_snow_timestamp, old_snow_timestamp, MOL, LAYER_HEIGHT, \
                LAYER_RHO, LAYER_T, LAYER_LWC, LAYER_CC, LAYER_POROSITY, \
                LAYER_ICE_FRACTION, LAYER_IRREDUCIBLE_WATER, LAYER_REFREEZE, \
                stake_names, stat, df_eval = future.result()

            IO.copy_local_to_global(
                indY, indX, RAIN, SNOWFALL, LWin, LWout, H, LE, B, QRR, MB, surfMB, Q,
                SNOWHEIGHT, TOTALHEIGHT, TS, ALBEDO, NLAYERS, ME, intMB, EVAPORATION,
                SUBLIMATION, CONDENSATION, DEPOSITION, REFREEZE, subM, Z0, surfM, MOL,
                LAYER_HEIGHT, LAYER_RHO, LAYER_T, LAYER_LWC, LAYER_CC, LAYER_POROSITY,
                LAYER_ICE_FRACTION, LAYER_IRREDUCIBLE_WATER, LAYER_REFREEZE)

            IO.copy_local_restart_to_global(indY,indX,local_restart)

            # Write results to file
            IO.write_results_to_file()

            # Write restart data to file
            IO.write_restart_to_file()

            if Config.stake_evaluation:
                # Store evaluation of stake measurements to dataframe
                stat = stat.rename('rmse')
                df_stat = pd.concat([df_stat, stat])

                for i in stake_names:
                    if Config.obs_type == 'mb':
                        df_val[i] = df_eval.mb
                    if Config.obs_type == 'snowheight':
                        df_val[i] = df_eval.snowheight

        # Measure time
        end_res = datetime.now()-start_res
        get_time_required(action="do calculations", times=end_res)

        if Config.stake_evaluation:
            # Save the statistics and the mass balance simulations at the stakes to files
            output_path = create_data_directory(path='output')
            df_stat.to_csv(os.path.join(output_path,'stake_statistics.csv'),sep='\t', float_format='%.2f')
            df_val.to_csv(os.path.join(output_path,'stake_simulations.csv'),sep='\t', float_format='%.2f')


def create_data_directory(path: str) -> str:
    """Create a directory in the configured data folder.

    Returns:
        Path to the created directory.
    """
    dir_path = os.path.join(Config.data_path, path)
    os.makedirs(dir_path, exist_ok=True)

    return dir_path


def get_timestamp_label(timestamp: str) -> str:
    """Get a formatted label from a timestring.

    Args:
        An ISO 8601 timestamp.

    Returns:
        Formatted timestamp with hyphens and time removed.
    """
    return (timestamp[0:10]).replace("-", "")


def set_output_netcdf_path() -> str:
    """Set the file path for the output netCDF file.

    Returns:
        The path to the output netCDF file.
    """
    time_start = get_timestamp_label(timestamp=Config.time_start)
    time_end = get_timestamp_label(timestamp=Config.time_end)
    output_path = f"{Config.output_prefix}_{time_start}-{time_end}.nc"

    return output_path

def start_logging():
    """Start the python logging"""

    if os.path.exists('./cosipy.yaml'):
        with open('./cosipy.yaml', 'rt') as f:
            config = yaml.load(f.read(),Loader=yaml.SafeLoader)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info('COSIPY simulation started')


def transform_coordinates(coords):
    """Transform geodetic coordinates to cartesian."""
    # WGS 84 reference coordinate system parameters
    A = 6378.137  # major axis [km]
    E2 = 6.69437999014e-3  # eccentricity squared

    coords = np.asarray(coords).astype(float)

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


@gen.coroutine
def close_everything(scheduler):
    yield scheduler.retire_workers(workers=scheduler.workers, close_workers=True)
    yield scheduler.close()


def print_notice(msg:str):
    print(f"{'-'*72}\n{msg}\n{'-'*72}\n")


def check_for_nan(data):
    if np.isnan(data.to_array()).any():
        raise SystemExit('ERROR! There are NaNs in the dataset.')

def get_time_required(action:str, times):
    run_time = get_time_elapsed(times)
    print(f"\tTime required to {action}: {run_time}")


def get_time_elapsed(times) -> str:
    run_time = times.total_seconds()
    time_elapsed = f"{run_time//60.0:4g} minutes {run_time % 60.0:2g} seconds\n"
    return time_elapsed


""" MODEL EXECUTION """
if __name__ == "__main__":
    main()
