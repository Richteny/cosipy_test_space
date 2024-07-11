import numpy as np
from cosipy.config import Config

eval_method = Config.eval_method
obs_type = Config.obs_type
tsl_method = Config.tsl_method
tsl_normalize = Config.tsl_normalize

import pandas as pd
from scipy import stats
from datetime import datetime
from cosipy.utils.options import read_opt
from numba import njit
from numba.core import types
from numba.typed import Dict

def evaluate(stake_names, stake_data, df_):
    """Evaluate the simulation using stake measurements.

    Args:
        stake_names (list): Stake IDs.
        stake_data (pd.Dataframe): Stake measurements.
        df (pd.Dataframe): Simulated mass balance and snow height.
    
    Returns:
        Statistical evaluation.
    """

    if Config.eval_method == 'rmse':
        stat = rmse(stake_names, stake_data, df_)
    else:
        stat = None
       
    return stat


def rmse(stake_names, stake_data, df_):
    if (Config.obs_type=='mb'):
        rmse = ((stake_data[stake_names].subtract(df_['mb'],axis=0))**2).mean()**.5
    elif (Config.obs_type=='snowheight'):
        rmse = ((stake_data[stake_names].subtract(df_['snowheight'],axis=0))**2).mean()**.5
    else:
        msg = f'RMSE not implemented for obs_type="{Config.obs_type}" in config.py.'
        raise NotImplementedError(msg)
    return rmse



def resample_output(cos_output):
    times = datetime.now()
    ds = cos_output
    ds = ds[['SNOWHEIGHT','HGT','MASK']]
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
def tsl_method_mantra(snowheights, hgts, mask, min_snowheight):
    amed = np.zeros(snowheights.shape[0])
    amean = np.zeros(snowheights.shape[0])
    astd = np.zeros(snowheights.shape[0])
    amax = np.zeros(snowheights.shape[0])
    amin = np.zeros(snowheights.shape[0])
    flag = np.zeros(snowheights.shape[0])
    #print("Starting loop.")
    for n in np.arange(0, snowheights.shape[0]):
        filtered_elev_vals = np.where(snowheights[n,:,:]>min_snowheight, hgts, np.nan).ravel()
        filtered_elev_vals = filtered_elev_vals[~np.isnan(filtered_elev_vals)]
        # had to change to <= instead of < -> implications?
        snowline_range = filtered_elev_vals[filtered_elev_vals <= np.nanpercentile(filtered_elev_vals, 2)]
        #print("Calculated percentiles.")
        #check if all array values are True / 1
        flat = snowheights[n,:,:].ravel()
        flat = flat[~np.isnan(flat)]
        #Convert to boolean mask and check if all true
        test = flat > min_snowheight
        if snowline_range.size == 0:
            print("Snowline range is empty. Glacier is most likely snow-free. Marking timestep as flagged.")
            print("\n Assigning maximum elevation of glacier.")
            snowline_range = np.array([np.nanmax(np.where(mask==1, hgts, np.nan))])
            print(snowline_range)
            flag[n] = 1
        elif np.nanmin(snowline_range) == np.nanmin(np.where(mask==1, hgts, np.nan)):
            #print("Glacier is most likely fully snow-covered. Marking timestep", n, "out of ", snowheights.shape[0])
            #print("Checking if all grid cells snow-covered.", test.all())
            if test.all() == True:
                snowline_range = np.array([np.nanmin(snowline_range)])
            #assign minimum value if all values are snow-covered
            flag[n] = 1
            

        amed[n] = np.nanmedian(snowline_range)
        amean[n] = np.nanmean(snowline_range)
        astd[n] = np.nanstd(snowline_range)
        amax[n] = np.nanmax(snowline_range)
        amin[n] = np.nanmin(snowline_range)


    return (amed, amean, astd, amax, amin, flag)

#Needs work? Method is not really reliable for a 2D distributed simulation due to large differences
@njit 
def tsl_method_conservative(snowheights, hgts, mask, min_snowheight):
    amed = np.zeros(snowheights.shape[0])
    amean = np.zeros(snowheights.shape[0])
    astd = np.zeros(snowheights.shape[0])
    amax = np.zeros(snowheights.shape[0])
    amin = np.zeros(snowheights.shape[0])
    flag = np.zeros(snowheights.shape[0])

    for n in np.arange(0, snowheights.shape[0]):
        #min altitude of snow
        #print("Processing ", n, "out of ", snowheights.shape[0])
        filtered_elev_snow = np.nanmin(np.where(snowheights[n,:,:]>=min_snowheight, hgts, np.nan).ravel())
        #print(filtered_elev_snow)
        #this line is basically redudant, ensures that snowline altitude cannot fall below glacier altitude
        filtered_elev_snow = np.nanmax(np.append(filtered_elev_snow, np.nanmin(np.where(mask==1, hgts, np.nan)))) #numba does not support np.maximum
        #print(filtered_elev_snow)
        #now for snow-free surfaces
        filtered_elev_nosnow = np.nanmax(np.where(snowheights[n,:,:]<min_snowheight, hgts, np.nan))
        #print(filtered_elev_nosnow)
        if np.isnan(filtered_elev_nosnow):
            #print("Glacier seems to be fully snow-covered. Assigning minimum elevation.")
            filtered_elev_nosnow = np.nanmin(np.where(mask==1, hgts, np.nan))
            #print(filtered_elev_nosnow,"m a.s.l.")
            flag[n] = 1
        else:
            #technically also redudant, ascertain that max. elevation of glacier is not exceeded
            filtered_elev_nosnow = np.nanmin(np.append(filtered_elev_nosnow, np.nanmax(np.where(mask==1, hgts, np.nan)))) #numba does not support minimum/maximum of numpy
        amed[n] = np.nanmedian(np.append(filtered_elev_snow, filtered_elev_nosnow))
        amean[n] = np.nanmean(np.append(filtered_elev_snow, filtered_elev_nosnow))
        astd[n] = np.nanstd(np.append(filtered_elev_snow, filtered_elev_nosnow)) #might be a bit large.. std of all values essentially function of resolution  
        amax[n] = np.nanmax(np.append(filtered_elev_snow, filtered_elev_nosnow))
        amin[n] = np.nanmin(np.append(filtered_elev_snow, filtered_elev_nosnow))

    return (amed, amean, astd, amax, amin, flag)

@njit
def tsl_method_gridsearch(snowheights, hgts, mask, min_snowheight):
    amed = np.zeros(snowheights.shape[0])
    amean = np.zeros(snowheights.shape[0])
    astd = np.zeros(snowheights.shape[0])
    amax = np.zeros(snowheights.shape[0])
    amin = np.zeros(snowheights.shape[0])
    flag = np.zeros(snowheights.shape[0])

    for n in np.arange(0,snowheights.shape[0]):
        snowmask = np.where(snowheights[n,:,:] > min_snowheight, 1, 0)
        #Mask to glacier extent. Values out of mask are set to NaN
        sno_arr = np.where(mask==1, snowmask, np.nan)
        #Check if snow-free or fully snow-covered
        #array slicing via [] does not seem to work, np.delete also does not
        #sno_arr_check = sno_arr[~np.isnan(sno_arr)]
        #sno_arr_check = np.delete(sno_arr, np.argwhere(np.isnan(sno_arr)))
        #print(sno_arr_check)
        if np.nanmin(sno_arr) == 0 and np.nanmax(sno_arr) == 0:
            #all values are 0 -> snow-free. Set maximum elevation
            elev_list = np.array(np.nanmax(np.where(mask==1, hgts, np.nan)))
            elev_list_snow = np.array(np.nanmax(np.where(mask==1, hgts, np.nan)))
            flag[n] = 1
        elif np.nanmin(sno_arr) == 1 and np.nanmax(sno_arr) == 1:
            #all values are 1 -> fully snow-covered
            elev_list = np.array(np.nanmin(np.where(mask==1, hgts, np.nan)))
            elev_list_snow = np.array(np.nanmin(np.where(mask==1, hgts, np.nan)))
            flag[n] = 1
            #TODO: List uncertainties here.. maybe 1 grid cell in elev. ?
        else:
            ilist = []
            ilist_snow = []
            for i in np.arange(0, sno_arr.shape[0]):
                for j in np.arange(0, sno_arr.shape[1]):
                    if sno_arr[i,j] == 1:
                        #get slice of grid cell to the left and right but also ceck down!
                        slices = sno_arr[i:i+2, j-1:j+2]
                        #first check left
                        if slices[0,0] == 0:
                            #create unique identifier
                            idx = i*1000 + (j-1)
                            #print(idx)
                            ilist.append(idx)
                            ilist_snow.append(i*1000 + (j))
                        #then check right
                        if slices[0,2] == 0:
                            idx = i*1000 + (j+1)
                            #print(idx)
                            ilist.append(idx)
                            ilist_snow.append(i*1000 + (j))
                        #then check below
                        if slices[1,1] == 0:
                            idx = (i+1)*1000 + (j)
                            #print(idx)
                            ilist.append(idx)
                            ilist_snow.append(i*1000+(j))
            #Values in list describe border values that are in glacier mask, where next grid cell is snow-covered
            #Can take metrics from these grid points, what happens if there is a "hole" in the data? Have to ignore it for now
            #np.array append is slow because it copies an array each time
            #this numba version does not support string lists to array .. workaround with numbers?
            ilist = np.unique(np.array(ilist))
            ilist_snow = np.unique(np.array(ilist_snow))
            elev_list = []
            for ix in ilist:
                i = int(round(ix, -3) / 1000)
                j = int(ix - i*1000)
                elev_list.append(hgts[i,j])
            elev_list_snow = [] 
            for ix in ilist_snow:
                i = int(round(ix, -3) / 1000)
                j= int(ix - i*1000)
                elev_list_snow.append(hgts[i,j])

            elev_list = np.array(elev_list)
            elev_list_snow = np.array(elev_list_snow)
            #There are multiple options now on how to implement the end result. Lazy one chosen for now.
            amed[n] = np.nanmedian(np.append(elev_list, elev_list_snow))
            amean[n] = np.nanmean(np.append(elev_list, elev_list_snow))
            astd[n] = np.nanstd(np.append(elev_list, elev_list_snow))
            amax[n] = np.nanmax(np.append(elev_list, elev_list_snow))
            amin[n] = np.nanmin(np.append(elev_list, elev_list_snow))
    return (amed, amean, astd, amax, amin, flag)


def calculate_tsl_byhand(snowheights,hgts,mask, min_snowheight,tsl_method, tsl_normalize):
    max_elev = np.nanmax(np.where(mask==1, hgts, np.nan))
    min_elev = np.nanmin(np.where(mask==1, hgts, np.nan))
    print("Calculating TSLA using {}. Normalization is set to {}.".format(tsl_method, tsl_normalize))
    print("Max elev.", max_elev,".\n Min elev.", min_elev)
    if tsl_method == 'mantra':
        amed, amean, astd, amax, amin, flag = tsl_method_mantra(snowheights, hgts, mask, min_snowheight)
    elif tsl_method == 'conservative':
        amed, amean, astd, amax, amin, flag = tsl_method_conservative(snowheights, hgts, mask, min_snowheight)
    else:
        amed, amean, astd, amax, amin, flag = tsl_method_gridsearch(snowheights, hgts, mask, min_snowheight)
    if tsl_normalize:
        #Start normalizing:
        amed_norm = (amed - min_elev) / (max_elev - min_elev)
        amean_norm = (amean - min_elev) / (max_elev - min_elev)
        astd_norm = (astd - min_elev) / (max_elev - min_elev)
        amax_norm = (amax - min_elev) / (max_elev - min_elev)
        amin_norm = (amin - min_elev) / (max_elev - min_elev)
        return (amed_norm, amean_norm, astd_norm, amax_norm, amin_norm, flag)
    else:
        return (amed, amean, astd, amax, amin, flag)
        

def create_tsl_df(cos_output,min_snowheight, tsl_method, tsl_normalize):
    times = datetime.now()
    amed,amean,astd,amax,amin,flag = calculate_tsl_byhand(cos_output.SNOWHEIGHT.values, cos_output.HGT.values, cos_output.MASK.values, min_snowheight, tsl_method, tsl_normalize)
    print(amed)
    tsl_df = pd.DataFrame({'time':pd.to_datetime(cos_output.time.values),
                            'Med_TSL':amed,
                            'Mean_TSL':amean,
                            'Std_TSL':astd,
                            'Max_TSL':amax,
                            'Min_TSL':amin,
                            'Flag': flag})
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

