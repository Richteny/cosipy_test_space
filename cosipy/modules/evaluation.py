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

@njit
def resample_by_hand(holder,vals,secs,time_vals):
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

@njit
def calculate_tsl_byhand(snowheights,hgts,min_snowheight):
    
    amed = np.zeros(snowheights.shape[0])
    amean = np.zeros(snowheights.shape[0])
    astd = np.zeros(snowheights.shape[0])
    amax = np.zeros(snowheights.shape[0])
    amin = np.zeros(snowheights.shape[0])
    for i in np.arange(0,snowheights.shape[0]):

        filtered_elev_vals = np.where(snowheights[i,:,:]>min_snowheight,hgts,np.nan).ravel()
        #filtered_elev_vals = filtered_elev_vals[~np.isnan(filtered_elev_vals)]
        snowline_range = filtered_elev_vals[filtered_elev_vals < np.nanpercentile(filtered_elev_vals,2)]
        
        try:
            amed[i] = np.nanmedian(snowline_range)
            amean[i]= np.nanmean(snowline_range)
            astd[i] = np.nanstd(snowline_range)
            amax[i] = np.nanmax(snowline_range)
            amin[i] = np.nanmin(snowline_range)
        except:
            amed[i] = np.nan
            amean[i] = np.nan
            astd[i] = np.nan
            amax[i] = np.nan
            amin[i] = np.nan
        
    return (amed,amean,astd,amax,amin)

def create_tsl_df(cos_output,min_snowheight):
    times = datetime.now()
    amed,amean,astd,amax,amin = calculate_tsl_byhand(cos_output.SNOWHEIGHT.values, cos_output.HGT.values, min_snowheight)        
    tsl_df = pd.DataFrame({'time':pd.to_datetime(cos_output.time.values),
                            'Med_TSL':amed,
                            'Mean_TSL':amean,
                            'Std_TSL':astd,
                            'Max_TSL':amax,
                            'Min_TSL':amin})
    print("Time required for calculating TSL only :", datetime.now()-times)
    return tsl_df

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

