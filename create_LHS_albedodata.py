import pandas as pd
from numba import njit
import numpy as np
import pathlib
import xarray as xr

path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/Halji/LHS-narrow/"
#path = "/data/scratch/richteny/thesis/io/data/output/bestfiles/"
albpath = "/data/scratch/richteny/Ren_21_Albedo/"
outpath = "/data/scratch/richteny/thesis/io/data/output/albedo_files/LHS/Halji/"
outpath_me = "/data/scratch/richteny/thesis/io/data/output/sens_me/"

#Load albedo observations
obs_albedo = xr.open_dataset(albpath+"Halji_hrz-merged_mean-albedos.nc")
obs_albedo = obs_albedo.sortby("time")
#obs_landsat = xr.open_dataset(albpath+"Halji_LS_filtered_albedos.nc")
#obs_sentinel = xr.open_dataset(albpath+"Halji_S2_filtered_albedos.nc")

#t_landsat = obs_landsat.time.values
#t_sentinel = obs_sentinel.time.values
#combined_obs_times = np.union1d(t_landsat, t_sentinel)
combined_obs_times = obs_albedo.time.values
calc_albedo_only = True

if calc_albedo_only:
    outpath = "/data/scratch/richteny/thesis/io/data/output/albedo_files/LHS/Halji/NN/"

### Helper functions ###
def prereq_res(ds):
    time_vals = pd.to_datetime(ds.time.values)
    unique_dates = np.unique(time_vals.date)
    holder = np.zeros(len(unique_dates))
    # Integer seconds since epoch for numba
    secs = ds.time.values.astype('int64')
    dates_pd = pd.to_datetime(unique_dates)
    clean_day_vals = dates_pd.astype('int64').values

    return (dates_pd,clean_day_vals,secs,holder)

@njit
def resample_by_hand(holder,vals,secs,day_starts):
    day_len = 86400000000000 #Nanoseconds in a day (24*60*60*1e9)
    n_days = len(day_starts)
    n_inputs = len(secs)
    i=0
    for i in range(n_days):
        ts = day_starts[i]
        next_ts = ts + day_len
        
        current_sum = 0.0
        current_count = 0
        
        for j in range(n_inputs):
            if secs[j] >= ts and secs[j] < next_ts:
                val = vals[j]
                if not np.isnan(val):
                    current_sum += val
                    current_count += 1
        if current_count > 0:
            holder[i] = current_sum / current_count
        else:
            holder[i] = np.nan
    return holder

### Processing ###

i=0
for fp in pathlib.Path(path).glob('*.nc'):
    
    print(fp.stem)
    if i % 100 == 0:
        print(f"Processing file {i}/2500")
    #ds = xr.open_dataset(fp).sel(time=albobs.time)[['ALBEDO','HGT','N_Points']]
    ds = xr.open_dataset(fp)[['ALBEDO','HGT','MB','N_Points','ME']]
    area_dynamic_total = ds['N_Points'].sum(dim=['lat','lon'])
    area_static_total = ds['N_Points'].isel(time=0).sum(dim=['lat','lon'])
    #create weighted mean 
    alb_total = (ds['ALBEDO'] * ds['N_Points']).sum(dim=['lat','lon'])
    me_total = (ds['ME'] * ds['N_Points']).sum(dim=['lat','lon'])

    weighted_alb = alb_total / area_dynamic_total
    if calc_albedo_only == False:
        weighted_melt = me_total / area_dynamic_total

        mb_total_vol = (ds['MB'] * ds['N_Points']).sum(dim=['lat','lon'])
        weighted_mb = mb_total_vol / area_static_total #calc on RGI area

        #melt per season
        time_da = weighted_melt['time']
        hydroyear = time_da.dt.year.where(time_da.dt.month < 10, time_da.dt.year +1)
        ds_w = weighted_melt.assign_coords(hydroyear=hydroyear)

        mask_ablation = time_da.dt.month.isin([5,6,7,8,9])

        ds_abl = (ds_w.where(mask_ablation).groupby('hydroyear').mean(dim='time'))
        ds_annual = ds_w.groupby('hydroyear').mean(dim='time')
        #rename so merge is possible
        ds_abl_rename = ds_abl.to_dataset(name="ME_abl")
        ds_annual_rename = ds_annual.to_dataset(name="ME_ann")
        ds_merge = xr.merge([ds_abl_rename, ds_annual_rename])

        # Combine into a new Dataset
        result = xr.Dataset({
            'ALBEDO_weighted': weighted_alb,
            'MB_weighted': weighted_mb
        })
    else:
        dates,clean_day_vals,secs,holder = prereq_res(weighted_alb)
        resampled_alb_vals = resample_by_hand(holder, weighted_alb.data, secs, clean_day_vals).copy()

        resampled_alb = xr.DataArray(resampled_alb_vals, coords={'time':dates}, dims=['time'], name="ALBEDO_weighted")
        result = resampled_alb.sel(time=obs_albedo.time)
        result = result.sortby("time")
    # Save to file
    #weighted_alb.to_netcdf(outpath + str(fp.name))
    result.to_netcdf(outpath + str(fp.name))
    #ds_merge.to_netcdf(outpath_me + str(fp.name))
    i += 1
