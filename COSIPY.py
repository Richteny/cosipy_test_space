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
#for name in [
#    "distributed",
#    "distributed.scheduler",
#    "distributed.worker",
#    "distributed.core",
#    "distributed.comm",
#]:
#    logging.getLogger(name).setLevel(logging.WARNING)
#logging.getLogger("distributed").setLevel(logging.WARNING) #silence the print outs which overcrowd the .err files
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
from cosipy.modules.evaluation import evaluate, resample_output, create_tsl_df, eval_tsl, resample_by_hand
from numba import njit
import xarray as xr

def main(lr_T=0.0, lr_RRR=0.0, lr_RH=0.0, RRR_factor=Constants.mult_factor_RRR, alb_ice=Constants.albedo_ice,
         alb_snow= Constants.albedo_fresh_snow, alb_firn=Constants.albedo_firn, albedo_aging= Constants.albedo_mod_snow_aging,
         albedo_depth= Constants.albedo_mod_snow_depth, center_snow_transfer_function= Constants.center_snow_transfer_function,
         spread_snow_transfer_function= Constants.spread_snow_transfer_function, roughness_fresh_snow= Constants.roughness_fresh_snow,
         roughness_ice= Constants.roughness_ice,roughness_firn= Constants.roughness_firn, aging_factor_roughness= Constants.aging_factor_roughness,
         LWIN_factor = Constants.mult_factor_LWin, WS_factor = Constants.mult_factor_WS, summer_bias_t2 = Constants.bias_T2,
         t_wet = Constants.t_star_wet, t_dry = Constants.t_star_dry, t_K = Constants.t_star_K,
         count=""):

    Config()
    Constants()

    start_logging()
    #Load count variable 
    if isinstance(count, int):
        count = count + 1

    # target geodetic 2000-2010 = -1.0425, unc= 0.26 == roughly -1.3 to -0.7825
    #RRR_factor = float(0.741) #0.97 
    #alb_ice = float(0.2153) #range LHS after satellite 0.115  to 0.233
    #alb_snow = float(0.916) #range LHS after satellite 0.887 to 0.93
    #alb_firn = float(0.566) #range LHS after satellite 0.506 to 0.692
    #albedo_aging = float(15.6) #range LHS 1 to 25
    #albedo_depth = float(3.0) #range  LHS 1 to 15 the lower the more positive MB
    #roughness_fresh_snow = float(0.24) #0.03 (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JD037032) to max 1.6 from Brock et al. 2006
    #roughness_ice = float(1.7) #range LHS 0.7 to 20 the lower the more positive MB
    #roughness_firn = float(4.0)
    #aging_factor_roughness = float(0.0026)
    opt_dict = (RRR_factor, alb_ice, alb_snow, alb_firn, albedo_aging, albedo_depth, center_snow_transfer_function,
                spread_snow_transfer_function, roughness_fresh_snow, roughness_ice, roughness_firn, aging_factor_roughness,
                LWIN_factor, WS_factor, summer_bias_t2, t_wet, t_dry, t_K)
    #0 to 5 - base, 6 center snow , 7 spreadsnow, 8 to 10 roughness length 
    #opt_dict=None
    lapse_T = float(lr_T)
    lapse_RRR = float(lr_RRR)
    lapse_RH = float(lr_RH)
    print("#--------------------------------------#")
    print("Starting simulations with the following parameters.")
    print(opt_dict)
    print("\n#--------------------------------------#")
    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------
    IO = IOClass(opt_dict=opt_dict)
    DATA = IO.create_data_file()

    # Create global result and restart datasets
    RESULT = IO.create_result_file(opt_dict=opt_dict) 
    RESTART = IO.create_restart_file()

    #----------------------------------------------
    # Calculation - Multithreading using all cores
    #----------------------------------------------

    # Auxiliary variables for futures
    futures = []


    # Measure time
    start_time = datetime.now()
    t2 = DATA.T2.values
    rh2 = DATA.RH2.values
    rrr = DATA.RRR.values
    hgt = DATA.HGT.values
    station_altitude = Config.station_altitude #define outside for numba
    t2, rh2, rrr = online_lapse_rate(t2,rh2,rrr,hgt,station_altitude,lapse_T,lapse_RH, lapse_RRR)

    print("Assigning values back to DATA")
    DATA['T2'] = (('time','lat','lon'), t2)
    DATA['RH2'] = (('time','lat','lon'), rh2)
    DATA['RRR'] = (('time','lat','lon'), rrr)
    print(np.nanmax(DATA.RH2.values))
    print(np.min(DATA.RH2.values))
    print(rh2.shape)
    print("Seconds needed for lapse rate:", datetime.now()-start_time)
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
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures, opt_dict=opt_dict)

    else:
        with LocalCluster(scheduler_port=Config.local_port, n_workers=Config.workers, local_directory='logs/dask-worker-space', threads_per_worker=1, silence_logs=True) as cluster:
            print(cluster)
            run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures, opt_dict=opt_dict)

    print("\n")
    print_notice(msg="Write results ...")
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
    #version for parsed floats by hand here
    if Constants.albedo_method == "Oerlemans98":
        try:
            results_output_name = output_netcdf.split('.nc')[0] + f"_RRR-{round(RRR_factor,4)}_{round(alb_snow,4)}_{round(alb_ice,4)}_{round(alb_firn,4)}"\
                                                                  f"_{round(albedo_aging,4)}_{round(albedo_depth,4)}_{round(roughness_fresh_snow,4)}"\
                                                                  f"_{round(roughness_ice,4)}_{round(roughness_firn,4)}_{round(aging_factor_roughness,6)}"\
                                                                  f"_{round(LWIN_factor,4)}_{round(WS_factor,4)}_{round(summer_bias_t2,4)}_{round(center_snow_transfer_function,4)}_num{count}.nc"
        #item below only works when objects are arrays and not given by hand, parameters not taken from pymc or sorts are floats
        except:
            results_output_name = output_netcdf.split('.nc')[0] + f"_RRR-{round(RRR_factor.item(),4)}_{round(alb_snow.item(),4)}_{round(alb_ice.item(),4)}_{round(alb_firn.item(),4)}"\
                                                                  f"_{round(albedo_aging.item(),4)}_{round(albedo_depth.item(),4)}_{round(roughness_fresh_snow,4)}"\
                                                                  f"_{round(roughness_ice,4)}_{round(roughness_firn,4)}_{round(aging_factor_roughness,6)}"\
                                                                  f"_{round(LWIN_factor,4)}_{round(WS_factor,4)}_{round(summer_bias_t2,4)}_{round(center_snow_transfer_function,4)}_num{count}.nc"
    else:
        try:
            results_output_name = output_netcdf.split('.nc')[0] + f"_RRR-{round(RRR_factor,4)}_{round(alb_snow,4)}_{round(alb_ice,4)}_{round(alb_firn,4)}"\
                                                                  f"_{round(t_wet,4)}_{round(t_dry,4)}_{round(t_K,4)}_{round(albedo_depth,4)}_{round(roughness_fresh_snow,4)}"\
                                                                  f"_{round(roughness_ice,4)}_{round(roughness_firn,4)}_{round(aging_factor_roughness,6)}"\
                                                                  f"_{round(LWIN_factor,4)}_{round(WS_factor,4)}_{round(summer_bias_t2,4)}_{round(center_snow_transfer_function,4)}_num{count}.nc"
        #item below only works when objects are arrays and not given by hand, parameters not taken from pymc or sorts are floats
        except:
            results_output_name = output_netcdf.split('.nc')[0] + f"_RRR-{round(RRR_factor.item(),4)}_{round(alb_snow.item(),4)}_{round(alb_ice.item(),4)}_{round(alb_firn.item(),4)}"\
                                                                  f"_{round(t_wet.item(),4)}_{round(t_dry.item(),4)}_{round(t_K.item(),4)}_{round(albedo_depth.item(),4)}_{round(roughness_fresh_snow,4)}"\
                                                                  f"_{round(roughness_ice,4)}_{round(roughness_firn,4)}_{round(aging_factor_roughness,6)}"\
                                                                  f"_{round(LWIN_factor,4)}_{round(WS_factor,4)}_{round(summer_bias_t2,4)}_{round(center_snow_transfer_function,4)}_num{count}.nc"

    IO.get_result().to_netcdf(os.path.join(output_path,results_output_name), encoding=encoding, mode='w')
    
    #print(np.nanmax(IO.get_result().ALBEDO))
    #print(np.nanmin(IO.get_result().ALBEDO))
    #print(np.nanmax(IO.get_result().Z0))

    #Check if 1D or 2D
    times = datetime.now()
    if Config.tsl_evaluation is True:
        if 'N_Points' in list(IO.get_result().keys()):
            dsmb = IO.get_result().sel(time=slice(Config.time_start_mb, Config.time_end_mb))
            if 'time' not in IO.get_result()['N_Points'].dims:
                print("Compute area weighted MB for 1D case.")
                dsmb['weighted_mb'] = dsmb['MB'] * dsmb['N_Points'] / np.sum(dsmb['N_Points'])
                spatial_mean = dsmb[['weighted_mb']].sum(dim=['lat','lon'])
                dfmb = spatial_mean['weighted_mb'].to_dataframe()
                mean_annual_df =  dfmb.resample("1YE").sum() #resample to fixed year to match geodetic
                geod_mb = np.nanmean(mean_annual_df['weighted_mb'].values)
            else:
                n_points_fixed = IO.get_result()['N_Points'].sel(time=Config.time_start_mb, method='nearest')
                ref_area_total = n_points_fixed.sum()
                #total_mass_change = (dsmb['MB'] * dsmb['N_Points']).sum(dim=['lat','lon']) time-varying 
                total_mass_change = (dsmb['MB']  *  n_points_fixed).sum(dim=['lat','lon'])
                dsmb['weighted_mb'] = total_mass_change / ref_area_total
                dfmb = dsmb['weighted_mb'].to_dataframe()
                mean_annual_df = dfmb.resample("1YE").sum()
                geod_mb = np.nanmean(mean_annual_df['weighted_mb'].values)
        else:
            print("2D case.")
            spatial_mean = IO.get_result()['MB'].mean(dim=['lat','lon'], keep_attrs=True)
            geod_df = spatial_mean.sel(time=slice(Config.time_start_mb,Config.time_end_mb)).to_dataframe()
            mean_annual_df = geod_df.resample("1YE").sum()
            geod_mb = np.nanmean(mean_annual_df.MB.values)
        print("Geod. MB test.") 
        print(geod_mb)
        print("Time it took to calculate geod. MB ", datetime.now()-times) 

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
    #IO.get_restart().to_netcdf(os.path.join(restart_path,f'restart_{timestamp}.nc'), encoding=encoding)
    
    
    times = datetime.now()
    if Config.tsl_evaluation is True:
        print("Starting TSL eval.")
        tsla_observations = pd.read_csv(Config.tsl_data_file) 

        tsl_csv_name = 'tsla_'+results_output_name.split('.nc')[0].lower()+'.csv'    
        tsla_observations = pd.read_csv(Config.tsl_data_file)
        ds_slice = IO.get_result().sel(time=slice(Config.time_start_cali, Config.time_end_cali))
        dates,clean_day_vals,secs = prereq_res(ds_slice)
        resampled_snow = resample_by_hand(ds_slice.SNOWHEIGHT.values, secs, clean_day_vals).copy()
        resampled_out = construct_resampled_ds(ds_slice,resampled_snow,dates.values)

        if 'N_Points' in ds_slice:
            if 'time' in ds_slice['N_Points'].dims:
                print("Resampling dynamic N_Points for TSL masking.")
                resampled_np = resample_by_hand(ds_slice.N_Points.values, secs, clean_day_vals)
                resampled_out['N_Points'] = (('time','lat','lon'), resampled_np)
                # mask snowheight
                resampled_out['SNOWHEIGHT'] = resampled_out['SNOWHEIGHT'].where(resampled_out['N_Points'] > 0)
            else:
                resampled_out['N_Points'] = (('lat','lon'), ds_slice['N_Points'].values)
                resampled_out['SNOWHEIGHT'] = resampled_out['SNOWHEIGHT'].where(resampled_out['N_Points'] > 0)
        print("Time required for resampling of output: ", datetime.now()-times)
        #Need HGT values as 2D, ensured with following line of code.
        resampled_out['HGT'] = (('lat','lon'), IO.get_result()['HGT'].data)
        resampled_out['MASK'] = (('lat','lon'), IO.get_result()['MASK'].data)

        if "N_Points" in resampled_out:
            if len(resampled_out["N_Points"].shape) == 3:
                n_points_arg = resampled_out["N_Points"].values
            else:
                n_points_arg = None
        else:
            n_points_arg = None
        tsl_out = create_tsl_df(resampled_out, Config.min_snowheight, Config.tsl_method, Config.tsl_normalize, n_points_arg)
        print("Max. TSLA:", np.nanmax(tsl_out['Med_TSL'].values))
        tsl_out.to_csv(os.path.join(output_path, tsl_csv_name))
        tsla_stats = eval_tsl(tsla_observations,tsl_out, Config.time_col_obs, Config.tsla_col_obs)
        print("TSLA Observed vs. Modelled RMSE: " + str(tsla_stats[0])+ "; R-squared: " + str(tsla_stats[1]))
        ## Match to observation dates for pymc routine
        tsl_out_match = tsl_out.loc[tsl_out['time'].isin(tsla_observations['LS_DATE'])]
    
        print("Time required for full TSL EVAL: ", datetime.now()-times)

        ## Create DF that holds params to save ##
        if Config.write_csv_status:
            try:
                param_df = pd.read_csv(f"./simulations/{Config.csv_filename}", index_col=0)
                curr_df = pd.DataFrame( np.concatenate((np.array(opt_dict, dtype=float),np.array([geod_mb]),
                                        tsl_out_match.Med_TSL.values)) ).transpose()
                curr_df.columns = ['rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn', 'albedo_aging',
                                   'albedo_depth', 'center_snow_transfer', 'spread_snow_transfer',
                                   'roughness_fresh_snow', 'roughness_ice', 'roughness_firn',
                                   'aging_factor_roughness', 'lwin_factor', 'ws_factor','t2_factor', 'mb'] +\
                                  [f'sim{i+1}' for i in range(tsl_out_match.shape[0])]

                param_df = pd.concat([param_df, curr_df], ignore_index=True)
            except:
                #print(opt_dict)
                param_df = pd.DataFrame( np.concatenate((np.array(opt_dict, dtype=float), np.array([geod_mb]),
                                         tsl_out_match.Med_TSL.values)) ).transpose()
                param_df.columns =   ['rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn', 'albedo_aging',
                                      'albedo_depth', 'center_snow_transfer', 'spread_snow_transfer',
                                      'roughness_fresh_snow', 'roughness_ice', 'roughness_firn',
                                      'aging_factor_roughness','lwin_factor', 'ws_factor','t2_factor', 'mb'] +\
                                     [f'sim{i+1}' for i in range(tsl_out_match.shape[0])]
            param_df.to_csv(f"./simulations/{Config.csv_filename}")

    #-----------------------------------------------
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

    return (geod_mb,tsl_out_match)

def run_cosipy(cluster, IO, DATA, RESULT, RESTART, futures, opt_dict=None):
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
                    futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, stake_names=stake_names, stake_data=df_stakes_data, opt_dict=opt_dict))
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
                            stake_data=df_stakes_data,
                            opt_dict=opt_dict
                        )
                    )
            else:
                mask = DATA.MASK.isel(lat=y, lon=x)
                # Provide restart grid if necessary
                if (mask == 1) and (not Config.restart):
                    check_for_nan(data=DATA.isel(lat=y,lon=x))
                    futures.append(client.submit(cosipy_core, DATA.isel(lat=y, lon=x), y, x, stake_names=stake_names, stake_data=df_stakes_data, opt_dict=opt_dict))
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
                            stake_data=df_stakes_data,
                            opt_dict=opt_dict
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

@njit
def online_lapse_rate(t2,rh2,rrr,hgt,station_altitude,lapse_T,lapse_RH,lapse_RRR):
    print(t2.shape)
    for t in range(t2.shape[0]):
        t2[t,:,:] = t2[t,:,:]+ (hgt - station_altitude)*lapse_T
        rh2[t,:,:] = rh2[t,:,:]+ (hgt - station_altitude)*lapse_RH
        rh2[t,:,:] = np.where(rh2[t,:,:] > 100, 100, rh2[t,:,:])
        rh2[t,:,:] = np.where(rh2[t,:,:] < 0, 0, rh2[t,:,:])
        rrr[t,:,:] = np.maximum(rrr[t,:,:]+ (hgt - station_altitude)*lapse_RRR, 0.0)
    return t2,rh2,rrr

def construct_resampled_ds(input_ds,vals,time_vals):
    data_vars = {'SNOWHEIGHT':(['time','lat','lon'], vals,
                               {'units': "m",
                                'long_name': "snowheight"})}
     

    # define coordinates
    coords = {'time': (['time'], time_vals),
              'lat': (['lat'], input_ds.lat.values),
              'lon': (['lon'], input_ds.lon.values)}
     
    ds = xr.Dataset(data_vars=data_vars,coords=coords)
    return ds

@njit
def resample_annual_mb(holder,vals,secs,time_vals):
    i=0
    for ts in time_vals:
        idx = np.argwhere((secs>=ts) & (secs<ts+24*3600000000000)).ravel()
        subset = vals[idx[0]:idx[-1]+1,:,:]
        latlon = np.zeros((vals.shape[1],vals.shape[2]))
        j = np.arange(0,vals.shape[1])
        k = np.arange(0,vals.shape[2])
        for j in np.arange(0,vals.shape[1]):
            for k in np.arange(0,vals.shape[2]):
                latlon[j,k] = np.nanmean(subset[:,j,k])
        holder[i,:,:] = latlon
        i+=1
    return holder

def prereq_res(ds):
    time_vals = pd.to_datetime(ds.time.values)
    #holder = np.zeros((len(np.unique(time_vals.date)), ds.SNOWHEIGHT.values.shape[1], ds.SNOWHEIGHT.values.shape[2]))
    # Integer seconds since epoch for numba
    secs = np.array([time_vals.astype('int64')]).ravel()
    dates = pd.to_datetime(np.unique(time_vals.date))
    clean_day_vals = np.array(dates.astype('int64'))

    return (dates,clean_day_vals,secs) #,holder)

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
