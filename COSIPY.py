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
import sys
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

from numba import njit, typeof

import xarray as xr

def main(lr_T=0.0, lr_RRR=0.0, lr_RH=0.0, RRR_factor=Constants.mult_factor_RRR, alb_ice=Constants.albedo_ice,
         alb_snow= Constants.albedo_fresh_snow, alb_firn=Constants.albedo_firn, albedo_aging= Constants.albedo_mod_snow_aging,
         albedo_depth= Constants.albedo_mod_snow_depth, center_snow_transfer_function= Constants.center_snow_transfer_function,
         spread_snow_transfer_function= Constants.spread_snow_transfer_function, roughness_fresh_snow= Constants.roughness_fresh_snow,
         roughness_ice= Constants.roughness_ice,roughness_firn= Constants.roughness_firn, aging_factor_roughness= Constants.aging_factor_roughness,
         count=""):

    Config()
    Constants()

    start_logging()
    times = datetime.now()
    #Load count variable 
    if isinstance(count, int):
        count = count + 1

    '''
    TEST TO PARSE A TUPLE OF PARAM VALUES AND NOT CALL IN DICTIONARY
    '''
    # these values crashed previously [array(2.74724189), array(0.25), array(0.84), array(0.555), array(1.1), array(1.1)]
    #RRR_factor = float(2.2)
    #alb_ice = float(0.2)
    #alb_snow = float(0.94)
    #alb_firn = float(0.555)
    #albedo_aging = float(23)
    #albedo_depth = float(3)
    #roughness_fresh_snow = np.array([1.0], dtype=float)
    opt_dict = (RRR_factor, alb_ice, alb_snow, alb_firn, albedo_aging, albedo_depth, center_snow_transfer_function,
                spread_snow_transfer_function, roughness_fresh_snow, roughness_ice, roughness_firn, aging_factor_roughness)
    #0 to 5 - base, 6 center snow , 7 spreadsnow, 8 to 10 roughness length 
    #Initialise dictionary and load Params#
    #opt_dict = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    #opt_dict['mult_factor_RRR'] = RRR_factor
    #opt_dict['albedo_ice'] = alb_ice
    #opt_dict['albedo_fresh_snow'] = alb_snow
    #opt_dict['albedo_firn'] = alb_firn
    #opt_dict['albedo_mod_snow_aging'] = albedo_aging
    #opt_dict['albedo_mod_snow_depth'] = albedo_depth
    #opt_dict['center_snow_transfer_function'] = center_snow_transfer_function
    #opt_dict['spread_snow_transfer_function'] = spread_snow_transfer_function
    #opt_dict['roughness_fresh_snow'] = roughness_fresh_snow
    #opt_dict['roughness_ice'] = roughness_ice
    #opt_dict['roughness_firn'] = roughness_firn
    #print(opt_dict)
    #print(typeof(opt_dict))
    lapse_T = float(lr_T)
    lapse_RRR = float(lr_RRR)
    lapse_RH = float(lr_RH)
    #print("Lapse rates are:", lapse_T, lapse_RRR, lapse_RH)
    #print("Time required to load in opt_dic: ", datetime.now()-times)
    print("#--------------------------------------#")
    print("Starting simulations with the following parameters.")
    print(opt_dict)
    #[print("Parameter ", x,"=",opt_dict[x]) for x in opt_dict.keys()]
    #for key in opt_dict.keys():
    #    print("Parameter ", key,"=",opt_dict[key])
    print("\n#--------------------------------------#")
    #additionally initial snowheight constant, snow_layer heights, temperature bottom, albedo aging and depth
    #------------------------------------------
    # Create input and output dataset
    #------------------------------------------
    #setup IO with new values from dictionary 
    times = datetime.now()
    #test_dict(opt_dict)
    IO = IOClass(opt_dict=opt_dict)
    start_time = datetime.now() 
    if Config.restart:
        #DATA = IO.create_data_file(suffix="_num{}_lrT_{}_lrRRR_{}_prcp_{}".format(count, round(abs(lapse_T),7),round(lapse_RRR,7),round(opt_dict['mult_factor_RRR'],5)))
         DATA = IO.create_data_file(suffix="_num{}_lrT_{}_lrRRR_{}_prcp_{}".format(count, round(abs(lapse_T),6),round(lapse_RRR, 6),round(opt_dict[0],5)))
    else:
        DATA = IO.create_data_file()
    # Create global result and restart datasets
    RESULT = IO.create_result_file(opt_dict=opt_dict) 
    RESTART = IO.create_restart_file()
    print("Time required to init IO, DATA, RESULT, RESTART: ", datetime.now()-times)
    #----------------------------------------------
    # Calculation - Multithreading using all cores  
    #----------------------------------------------


    # Auxiliary variables for futures
    futures= []
    #adjust lapse rates
    #print("#--------------------------------------#") 
    #print("\nStarting run with lapse rates:", lapse_T, "and:", lapse_RRR) 
    #print("\nAlbedo ice, snow and firn:", opt_dict['albedo_ice'],",",opt_dict['albedo_fresh_snow'],"and", opt_dict['albedo_firn'])
    #print("\nRRR mult factor is:", opt_dict['mult_factor_RRR'])
    #print("\n#--------------------------------------#")
    
    start2 = datetime.now()
    t2 = DATA.T2.values
    rh2 = DATA.RH2.values
    rrr = DATA.RRR.values
    hgt = DATA.HGT.values
    #print(np.nanmean(t2))
    station_altitude = Config.station_altitude #define outside for numba
    t2, rh2, rrr = online_lapse_rate(t2,rh2,rrr,hgt,station_altitude,lapse_T,lapse_RH, lapse_RRR)
    #print(np.nanmean(t2))
    print("Assigning values back to DATA")
    DATA['T2'] = (('time','lat','lon'), t2)
    DATA['RH2'] = (('time','lat','lon'), rh2)
    DATA['RRR'] = (('time','lat','lon'), rrr)
    print(np.nanmax(DATA.RH2.values))
    print(np.min(DATA.RH2.values))
    print(rh2.shape)
    #print(np.nanmean(DATA.T2.values))
    print("Seconds needed for lapse rate:", datetime.now()-start2)
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
    #item below only works when objects are arrays and not given by hand
    results_output_name = output_netcdf.split('.nc')[0] + f"_RRR-{round(RRR_factor.item(),4)}_{round(alb_snow.item(),4)}_{round(alb_ice.item(),4)}_{round(alb_firn.item(),4)}_num{count}.nc"
    #IO.get_result().to_netcdf(os.path.join(output_path,results_output_name), encoding=encoding, mode = 'w')
    
    print(np.nanmax(IO.get_result().ALBEDO))
    print(np.nanmin(IO.get_result().ALBEDO))
    print(np.nanmax(IO.get_result().Z0))
    #dataset = IO.get_result()
    #calculate MB for geod. reference
    #Check if 1D or 2D
    times = datetime.now()
    #this takes 1min, make it faster! CHANGE IT BACK ONCE DONE WITH TESTS!
    if Config.tsl_evaluation is True:
        if 'N_Points' in list(IO.get_result().keys()):
            print("Compute area weighted MB for 1D case.")
            dsmb = IO.get_result().sel(time=slice("2000-01-01", "2009-12-31"))
            dsmb['weighted_mb'] = dsmb['MB'] * dsmb['N_Points'] / np.sum(dsmb['N_Points'])
            #print("time 1:", datetime.now()-times)
            #time_vals = pd.to_datetime(dsmb.time.values)
            #secs = np.array([time_vals.astype('int64')]).ravel()
            #years = np.unique(time_vals.year)
            #clean_year_vals = np.array([np.datetime64(pd.datetime(x,1,1,0,0,0)) for x in years])
            #clean_year_vals = clean_year_vals.astype('int64')
            #print(secs)
            #print(clean_year_vals)
            #sum over glacier - all grid cell
            spatial_mean = dsmb[['weighted_mb']].sum(dim=['lat','lon'])
            dfmb = spatial_mean['weighted_mb'].to_dataframe()
            #dfmb.reset_index(inplace=True)
            #dfmb['FY'] =  dfmb.apply(lambda x: pd.datetime(x.time.year,1,1).year, axis=1)
            mean_annual_df =  dfmb.resample("1Y").sum() #resample to fixed year to match geodetic
            geod_mb = np.nanmean(mean_annual_df['weighted_mb'].values)
        else:
            print("2D case.")
            spatial_mean = IO.get_result()['MB'].mean(dim=['lat','lon'], keep_attrs=True)
            #mean glacier-wide MB
            #select timeframe from 2010 to 2020 (do not include first day of 2020)
            geod_df = spatial_mean.sel(time=slice("2000-01-01","2009-12-31")).to_dataframe()
            #geod_df = spatial_mean.sel(time=slice("2010-01-01","2019-12-31")).to_dataframe()
            #geod_df['FY'] = geod_df.apply(lambda x: str(pd.to_datetime(str(x.time.year)+'-01-01').year), axis=1)
            mean_annual_df = geod_df.resample("1Y").sum()
            geod_mb = np.nanmean(mean_annual_df.MB.values)
        print("Geod. MB test.") 
        print(geod_mb)
        print("Time it took to calculate geod. MB ", datetime.now()-times)
    #else:
    #    geod_mb = np.array([np.nan])
    #cmb_spatial_mean_cum = np.cumsum(cmb_spatial_mean)    

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
    #IO.get_restart().to_netcdf(os.path.join(restart_path,f'restart_'{timestamp}+'_num{}_lrT_{}_lrRRR_{}_prcp_{}.nc'.format(count,round(abs(lapse_T),7), round(lapse_RRR,7),round(opt_dict['mult_factor_RRR'],5))), encoding=encoding)
    
    #----------------------------------------------
    # Implement TSL Extraction
    #----------------------------------------------
    #if (Config.restart == True) and (Config.merge == True):
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
    if Config.tsl_evaluation is True:
        print("Starting TSL eval.")
        #times = datetime.now()
        tsla_observations = pd.read_csv(Config.tsl_data_file)
        
        if (Config.restart == True) and (Config.merge == True):
            tsl_csv_name = 'tsla_'+merged_output_name.split('.nc')[0].lower()+'.csv'
            resampled_out = resample_output(ds_merged)
            tsl_out = calculate_tsl(resampled_out, Config.min_snowheight)
            tsla_stats = eval_tsl(tsla_observations, tsl_out, Config.time_col_obs, Config.tsla_col_obs)
            print("TSLA Observed vs Modelled RMSE: " + str(tsla_stats[0]) + "; R-squared: " + str(tsla_stats[1]))
            #tsl_out.to_csv(os.path.join(output_path,tsl_csv_name))
            del ds_merged
        else:
            tsl_csv_name = 'tsla_'+results_output_name.split('.nc')[0].lower()+'.csv'    
            tsla_observations = pd.read_csv(Config.tsl_data_file)
            #a_resampled_out = resample_output(IO.get_result())
            times = datetime.now()
            #dates,clean_day_vals,secs,holder = prereq_res(IO.get_result().sel(time=slice("2010-01-01","2019-12-31")))
            dates,clean_day_vals,secs,holder = prereq_res(IO.get_result().sel(time=slice("2000-01-01","2009-12-31")))
            resampled_array = resample_by_hand(holder, IO.get_result().sel(time=slice("2000-01-01","2009-12-31")).SNOWHEIGHT.values, secs, clean_day_vals)
            resampled_out = construct_resampled_ds(IO.get_result().sel(time=slice("2000-01-01","2009-12-31")),resampled_array,dates.values)
            #print(resampled_out)
            print("Time required for resampling of output: ", datetime.now()-times)
            #Need HGT values as 2D, ensured with following line of code.
            resampled_out['HGT'] = (('lat','lon'), IO.get_result()['HGT'].data)
            resampled_out['MASK'] = (('lat','lon'), IO.get_result()['MASK'].data)
            #print("Time required for resampling: ", datetime.now()-times)
            #a_tsl_out = create_tsl_df(a_resampled_out, min_snowheight, tsl_method, tsl_normalize)
            tsl_out = create_tsl_df(resampled_out, Config.min_snowheight, Config.tsl_method, Config.tsl_normalize)
            #tsl_out = calculate_tsl(resampled_out, min_snowheight)
            #print(tsla_observations)
            #print(tsl_out)
            #print(np.nanmedian(tsl_out['Med_TSL']))
            tsla_stats = eval_tsl(tsla_observations,tsl_out, Config.time_col_obs, Config.tsla_col_obs)
            print("TSLA Observed vs. Modelled RMSE: " + str(tsla_stats[0])+ "; R-squared: " + str(tsla_stats[1]))
            #tsl_out.to_csv(os.path.join(output_path,tsl_csv_name))
            ## Match to observation dates for pymc routine
            tsl_out_match = tsl_out.loc[tsl_out['time'].isin(tsla_observations['LS_DATE'])]
            #print(np.array(tsl_out_match['Med_TSL']))
            #print(tsl_out_match['Med_TSL'].values.shape)
            #tsla_obs_v2 = tsla_observations.loc[tsla_observations['LS_DATE'].isin(tsl_out_match['time'])]
            
            #if tsl_normalize:
            #    tsla_obs_v2['SC_stdev'] = (tsla_obs_v2['SC_stdev']) / (tsla_obs_v2['glacier_DEM_max'] - tsla_obs_v2['glacier_DEM_min'])
            #a_tsl_out.to_csv(os.path.join(data_path,'output','test_for_resample.csv'))
            ### PUTTING TEST FOR COMB COST FUNCTION SCORE HERE ###
            #eval_tsla = np.delete(tsla_obs_v2.TSL_normalized.values, np.argwhere(np.isnan(tsl_out_match.Med_TSL.values)))
            #path_to_geod = "/data/scratch/richteny/Hugonnet_21_MB/"
            #rgi_id = "RGI60-11.00897"
            #rgi_region = rgi_id.split('-')[-1][:2]
            #geod_ref = pd.read_csv(path_to_geod+"dh_{}_rgi60_pergla_rates.csv".format(rgi_region))
            #geod_ref = geod_ref.loc[geod_ref['rgiid'] == rgi_id]
            #geod_ref = geod_ref.loc[geod_ref['period'] == "2000-01-01_2010-01-01"]
            #geod_mb_ref = geod_ref[['dmdtda','err_dmdtda']]
            #eval_mb = geod_mb_ref['dmdtda'].values
            #sigma_mb = geod_mb_ref['err_dmdtda'].values
            #sigma_tsla = np.delete(tsla_obs_v2.SC_stdev.values, np.argwhere(np.isnan(tsl_out_match.Med_TSL.values)))
            #sim_tsla = tsl_out_match.Med_TSL.values[~np.isnan(tsl_out_match.Med_TSL.values)]
            #mbe_tsla = (((eval_tsla - sim_tsla)**2) / (sigma_tsla**2)).mean() 
            #mbe = ((eval_mb - geod_mb)**2) / (sigma_mb**2)
            #cost = -(1*mbe_tsla + 1*mbe)
            #print("Full cost function value is: ", cost)
        print("Time required for full TSL EVAL: ", datetime.now()-times)
        
        ## Create DF that holds params to save ##
        if Config.write_csv_status:
            try:
                param_df = pd.read_csv("./simulations/cosipy_synthetic_params.csv", index_col=0)
                curr_df = pd.DataFrame( np.concatenate((np.array(opt_dict, dtype=float),np.array([geod_mb]),
                                                        tsl_out_match.Med_TSL.values)) ).transpose()
                curr_df.columns = ['rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn', 'albedo_aging',
                                   'albedo_depth', 'center_snow_transfer', 'spread_snow_transfer',
                                   'roughness_fresh_snow', 'roughness_ice', 'roughness_firn', 'aging_factor_roughness', 'mb'] +\
                                  [f'sim{i+1}' for i in range(tsl_out_match.shape[0])]
                #print("\n--------------------------------")
                #print(curr_df)
                #curr_df.to_csv("./simulations/curr_df.csv")
                param_df = pd.concat([param_df, curr_df], ignore_index=True)
                #print(param_df)
                #print("\n----------------------------------")
            except:
                #param_df does not exist yet, create
                #print(tsl_out_match)
                #print(tsl_out_match.shape[1])
                print(opt_dict)
                #dtype float in np.array(opt_dict, dtype=float) not working when using prescribed parameters in earlier instance
                #test = np.concatenate((np.array(opt_dict, dtype=object), np.array([geod_mb]), tsl_out_match.Med_TSL.values))
                #print(test)
                param_df = pd.DataFrame( np.concatenate((np.array(opt_dict, dtype=float), np.array([geod_mb]),
                                                         tsl_out_match.Med_TSL.values)) ).transpose()
                #print(param_df)
                param_df.columns =   ['rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn', 'albedo_aging',
                                      'albedo_depth', 'center_snow_transfer', 'spread_snow_transfer',
                                      'roughness_fresh_snow', 'roughness_ice', 'roughness_firn', 'aging_factor_roughness', 'mb'] +\
                                     [f'sim{i+1}' for i in range(tsl_out_match.shape[0])]
            param_df.to_csv("./simulations/cosipy_synthetic_params.csv")

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
                    if np.isnan(DATA.sel(south_north=y, west_east=x).to_array()).any():
                        print_nan_error()
                    futures.append(client.submit(cosipy_core, DATA.sel(south_north=y, west_east=x), y, x, stake_names=stake_names, stake_data=df_stakes_data))
                elif (mask == 1) and (Config.restart):
                    if np.isnan(DATA.sel(south_north=y, west_east=x).to_array()).any():
                        print_nan_error()
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
                    if np.isnan(DATA.isel(lat=y,lon=x).to_array()).any():
                        print_nan_error()
                    futures.append(client.submit(cosipy_core, DATA.isel(lat=y, lon=x), y, x, stake_names=stake_names, stake_data=df_stakes_data, opt_dict=opt_dict))
                elif (mask == 1) and (Config.restart):
                    if np.isnan(DATA.isel(lat=y,lon=x).to_array()).any():
                        print_nan_error()
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
                DEPOSITION, REFREEZE, subM, Z0, surfM, MOL, LAYER_HEIGHT, \
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
    holder = np.zeros((len(np.unique(time_vals.date)), ds.SNOWHEIGHT.values.shape[1], ds.SNOWHEIGHT.values.shape[2]))
    # Integer seconds since epoch for numba
    secs = np.array([time_vals.astype('int64')]).ravel()
    dates = pd.to_datetime(np.unique(time_vals.date))
    clean_day_vals = np.array(dates.astype('int64'))

    return (dates,clean_day_vals,secs,holder)

@gen.coroutine
def close_everything(scheduler):
    yield scheduler.retire_workers(workers=scheduler.workers, close_workers=True)
    yield scheduler.close()


def print_notice(msg:str):
    print(f"{'-'*72}\n{msg}\n{'-'*72}\n")


def print_nan_error():
    print('ERROR! There are NaNs in the dataset.')
    sys.exit()


def get_time_required(action:str, times):
    run_time = get_time_elapsed(times)
    print(f"\tTime required to {action}: {run_time}")


def get_time_elapsed(times) -> str:
    run_time = times.total_seconds()
    time_elapsed = f"{run_time//60.0:4g} minutes {run_time % 60.0:2g} seconds\n"
    return time_elapsed


""" MODEL EXECUTION """
if __name__ == "__main__":
    #import pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()
    #stats=pstats.Stats(profiler).sort_stats("tottime")
    #stats.print_stats()
    
