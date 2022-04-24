import numpy as np
from config import eval_method, obs_type
from cosipy.utils.options import read_opt
import pandas as pd
from scipy import stats
from datetime import datetime
from numba import njit

def evaluate(stake_names, stake_data, df_, opt_dict=None):
    """ This methods evaluates the simulation with the stake measurements
        stake_name  ::  """
            
    # Read and set options
    read_opt(opt_dict, globals())
    if eval_method == 'rmse':
        stat = rmse(stake_names, stake_data, df_)
    else:
        stat = None
       
    return stat


def rmse(stake_names, stake_data, df_):
    if (obs_type=='mb'):
        rmse = ((stake_data[stake_names].subtract(df_['mb'],axis=0))**2).mean()**.5
    if (obs_type=='snowheight'):
        rmse = ((stake_data[stake_names].subtract(df_['snowheight'],axis=0))**2).mean()**.5
    return rmse



def resample_output(cos_output):
    times = datetime.now()
    ds = cos_output
    ds = ds[['SNOWHEIGHT','HGT']]
    #ds_daily = ds.resample(time='1d', keep_attrs=True).mean(dim='time')
    df_daily = ds.to_dataframe().groupby([pd.Grouper(level='time',freq='1d'),
                                          pd.Grouper(level='lat'),
                                          pd.Grouper(level='lon')]).mean()
    ds_daily = df_daily.to_xarray()
    print("Required time for resample only: ", datetime.now()-times)
    return ds_daily


def resample_array_style(data):
    times = datetime.now()
    freq=24
    if data.ndim == 3:
        lats = data.shape[1]
        lons = data.shape[2]
        res = data[:-1,:,:].reshape(freq,-1,lats,lons)
    else:
        point = data.shape[1]
        res = data[:-1,:].reshape(freq,-1,points)
    res = np.nanmean(res,axis=0)
    return res


    
def calculate_tsl(cos_output, min_snowheight):
    times = datetime.now()
    tsl_df = pd.DataFrame({'time': [],
                           'Med_TSL': [],
                           'Mean_TSL': [],
                           'Std_TSL': [],
                           'Max_TSL': [],
                           'Min_TSL': []})

    snow_ds = cos_output.where(cos_output.SNOWHEIGHT > min_snowheight)
    #drop = true discards a lot of values, where there were NaNs because min snowheight is arbitrarily chosen 
    for timestep in snow_ds.time.values:
        subset = snow_ds.sel(time=timestep)
        two_perc = np.nanpercentile(subset.HGT,2)
        snowline_range = subset.where(subset.HGT < two_perc)
        
        try:
            tsl_med = np.nanmedian(snowline_range.HGT)
            tsl_mean = np.nanmean(snowline_range.HGT)
            tsl_std = np.nanstd(snowline_range.HGT)
            tsl_max = np.nanmax(snowline_range.HGT)
            tsl_min = np.nanmin(snowline_range.HGT)
        except:
            print("Forced NaNs. Timestamp:", timestep)
            tsl_med = np.nan
            tsl_mean = np.nan
            tsl_std = np.nan
            tsl_max = np.nan
            tsl_min = np.nan

        tsl_df = tsl_df.append({'time': timestep,
                                'Med_TSL': tsl_med,
                                'Mean_TSL': tsl_mean,
                                'Std_TSL': tsl_std,
                                'Max_TSL': tsl_max,
                                'Min_TSL': tsl_min}, ignore_index=True)
        tsl_df['time'] = pd.to_datetime(tsl_df['time'])
    print("Time required for calculating TSL only :", datetime.now()-times)
    return tsl_df

def mbe_score(y_obs, y_pred):
    diff = (y_obs-y_pred)
    mbe = np.mean(diff)
    return mbe
 
def eval_tsl(tsl_obs, tsl_mod, time_col_obs, tsla_col_obs):
    times= datetime.now()
    tsl_obs[time_col_obs] = pd.to_datetime(tsl_obs[time_col_obs])
    #first get only modelled values where observation is present
    tsl_modelled = tsl_mod[tsl_mod['time'].isin(tsl_obs[time_col_obs])]
    #second subset observations where modelled values are present
    tsl_observed = tsl_obs[tsl_obs[time_col_obs].isin(tsl_mod['time'])]
    #Drop NaNs if NaNs in dataset, assuming same size
    nan_indices = np.argwhere(np.isnan(tsl_modelled.Med_TSL.values)).flatten().tolist()
    tsl_observed = tsl_observed.drop(tsl_observed.index[nan_indices])
    tsl_modelled = tsl_modelled.drop(tsl_modelled.index[nan_indices])
    rmse = ((tsl_observed[tsla_col_obs].values - tsl_modelled['Med_TSL'].values)**2).mean()**.5 
    slope, intercept, r_value, p_value, std_err = stats.linregress(tsl_observed[tsla_col_obs].values,tsl_modelled['Med_TSL'].values)    
    r2 = r_value**2 #other options to be filled in or print out multiple metrics
    mbe = mbe_score(tsl_observed[tsla_col_obs].values,tsl_modelled['Med_TSL'].values)
    mae = (1/len(tsl_observed[tsla_col_obs].values)) * sum(abs(tsl_observed[tsla_col_obs].values - tsl_modelled['Med_TSL'].values))
    print("Time required for calculating TSL stats: ", datetime.now()-times)
    return rmse,r2,mbe,mae

