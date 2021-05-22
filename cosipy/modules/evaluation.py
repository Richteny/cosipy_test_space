from cosipy.config import Config

eval_method = Config.eval_method
obs_type = Config.obs_type
tsl_method = Config.tsl_method
tsl_normalize = Config.tsl_normalize

import numpy as np
import pandas as pd
from scipy import stats



def evaluate(stake_names, stake_data, df):
    """Evaluate the simulation using stake measurements.

    Implemented stake evaluation methods:

        - **rmse**: RMSE of simulated mass balance.

    Args:
        stake_names (list): Stake IDs.
        stake_data (pd.Dataframe): Stake measurements.
        df (pd.Dataframe): Simulated mass balance and snow height.

    Returns:
        float or None: Statistical evaluation.
    """

    if Config.eval_method == "rmse":
        stat = rmse(stake_names, stake_data, df)
    else:
        stat = None

    return stat


def rmse(stake_names: list, stake_data, df) -> float:
    """Get RMSE of simulated stake measurements.

    Args:
        stake_names: Stake IDs.
        stake_data (pd.Dataframe): Stake measurements.
        df (pd.Dataframe): Simulated mass balance and snow height.

    Returns:
        RMSE of simulated measurements.
    """
    if Config.obs_type not in ["mb", "snowheight"]:
        msg = f'RMSE not implemented for obs_type="{Config.obs_type}" in config.toml.'
        raise NotImplementedError(msg)
    else:
        rmse = (
            (stake_data[stake_names].subtract(df[Config.obs_type], axis=0))
            ** 2
        ).mean() ** 0.5

    return rmse

def resample_output(cos_output):

    ds = cos_output
    ds = ds[['SNOWHEIGHT','HGT']]
    ds_daily = ds.resample(time='1d', keep_attrs=True).mean(dim='time')
    return ds_daily
    
def calculate_tsl(cos_output, min_snowheight):
    
    tsl_df = pd.DataFrame({'time': [],
                           'Med_TSL': [],
                           'Mean_TSL': [],
                           'Std_TSL': [],
                           'Max_TSL': [],
                           'Min_TSL': []})

    snow_ds = cos_output.where(cos_output.SNOWHEIGHT > min_snowheight, drop=True)
     
    for timestep in snow_ds.time.values:
        subset = snow_ds.sel(time=timestep)
        two_perc = np.nanpercentile(subset.HGT,2)
        snowline_range = subset.where(subset.HGT < two_perc)
        tsl_med = np.nanmedian(snowline_range.HGT)
        tsl_mean = np.nanmean(snowline_range.HGT)
        tsl_std = np.nanstd(snowline_range.HGT)
        tsl_max = np.nanmax(snowline_range.HGT)
        tsl_min = np.nanmin(snowline_range.HGT)
        
        tsl_df = tsl_df.append({'time': timestep,
                                'Med_TSL': tsl_med,
                                'Mean_TSL': tsl_mean,
                                'Std_TSL': tsl_std,
                                'Max_TSL': tsl_max,
                                'Min_TSL': tsl_min}, ignore_index=True)
        tsl_df['time'] = pd.to_datetime(tsl_df['time'])
        
    return tsl_df

def mbe_score(y_obs, y_pred):
    diff = (y_obs-y_pred)
    mbe = np.mean(diff)
    return mbe
 
def eval_tsl(tsl_obs, tsl_mod):
    
    print(time_col_obs)
    tsl_obs[time_col_obs] = pd.to_datetime(tsl_obs[time_col_obs])
    #first get only modelled values where observation is present
    tsl_modelled = tsl_mod[tsl_mod['time'].isin(tsl_obs[time_col_obs])]
    #second subset observations where modelled values are present
    tsl_observed = tsl_obs[tsl_obs[time_col_obs].isin(tsl_mod['time'])]
    rmse = ((tsl_observed[tsla_col_obs].values - tsl_modelled['Med_TSL'].values)**2).mean()**.5 
    slope, intercept, r_value, p_value, std_err = stats.linregress(tsl_observed[tsla_col_obs].values,tsl_modelled['Med_TSL'].values)    
    r2 = r_value**2 #other options to be filled in or print out multiple metrics
    mbe = mbe_score(tsl_observed[tsla_col_obs].values,tsl_modelled['Med_TSL'].values)
    mae = (1/len(tsl_observed[tsla_col_obs].values)) * sum(abs(tsl_observed[tsla_col_obs].values - tsl_modelled['Med_TSL'].values))

    return rmse,r2,mbe,mae
