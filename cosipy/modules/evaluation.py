import numpy as np
from cosipy.config import Config

eval_method = Config.eval_method
obs_type = Config.obs_type
tsl_method = Config.tsl_method
tsl_normalize = Config.tsl_normalize

import pandas as pd
from scipy import stats
from datetime import datetime
from numba import njit

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
    times = datetime.now()
    ds = cos_output[['SNOWHEIGHT','HGT','MASK']]
    #ds_daily = ds.resample(time='1d', keep_attrs=True).mean(dim='time')
    df_daily = ds.to_dataframe().groupby([pd.Grouper(level='time',freq='1d'),
                                          pd.Grouper(level='lat'),
                                          pd.Grouper(level='lon')]).mean()
    ds_daily = df_daily.to_xarray()
    print("Required time for resample only: ", datetime.now()-times)
    return ds_daily
""" archived!
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
"""
@njit
def resample_by_hand(vals,secs,time_vals):
    ntime, nlat, nlon = vals.shape
    ndays = len(time_vals)
    
    day_next = 24*3600*1e9 #nanoseconds

    out = np.zeros((ndays, nlat, nlon))
    count = np.zeros((ndays, nlat, nlon))

    day_idx = 0
    day_end = time_vals[0] + day_next

    for t in range(ntime):
        ts = secs[t]

        while day_idx < ndays -1 and ts >= day_end:
            day_idx += 1
            day_end = time_vals[day_idx] + day_next

        for j in range(nlat):
            for k in range(nlon):
                v = vals[t, j, k]
                if not np.isnan(v):
                    out[day_idx, j, k] += v
                    count[day_idx, j, k] += 1

    #mean
    for d in range(ndays):
        for j in range(nlat):
            for k in range(nlon):
                if count[d, j, k] > 0:
                    out[d, j, k] /= count[d, j, k]
                else:
                    out[d, j, k] = np.nan
    return out

@njit
def tsl_method_mantra(albedos, hgts, mask, min_albedo, min_elevs_t, max_elevs_t):
    """
    Use pre-calculated dynamic min/max elevations and finds 2th percentile of snow-covered pixels.
    We follow a bottom-up logic to handle snow-scoured areas.

    Args:
        albedos (3d): time lat lon
        hgts (2d) static elevations
        mask (2d) static mask
        min_albedo (float): threshold
        min_elevs_t (1D): dynamic min elev
        max_elevs_t (1D): dynamic max elev
    """
    ntime = albedos.shape[0]
    rows, cols = hgts.shape

    amed = np.full(ntime, np.nan)
    amean = np.full(ntime, np.nan)
    astd = np.full(ntime, np.nan)
    amax = np.full(ntime, np.nan)
    amin = np.full(ntime, np.nan)
    flag = np.zeros(ntime)
    #print("Starting loop.")

    for n in range(ntime):

        # Get dynamic geometry
        current_min = min_elevs_t[n]
        current_max = max_elevs_t[n]

        # if glacier gone skip
        if np.isnan(current_min):
            flag[n] = 3
            continue

        snow_elevs = np.zeros(rows * cols)
        snow_count = 0
        total_valid_pixels = 0

        for r in range(rows):
            for c in range(cols):
                #check static mask
                if mask[r,c] == 1:
                    h = hgts[r,c]

                    #is pixel within current elevation range
                    if h >= current_min and h <= current_max:
                        total_valid_pixels += 1

                        alb = albedos[n,r,c]
                        if not np.isnan(alb) and alb >= min_albedo:
                            snow_elevs[snow_count] = h
                            snow_count += 1
        # Handle no snow and all snow
        if snow_count == 0:
            #bare ice
            #tsla technically above glacier, we set to max elev.
            res_val = current_max
            flag[n] = 2
            amed[n] = res_val; amean[n] = res_val
            amin[n] = res_val; amax[n] = res_val
            continue

        elif snow_count == total_valid_pixels:
            #full snow cover, tsla at bottom
            res_val = current_min
            flag[n] = 1
            amed[n] = res_val; amean[n] = res_val
            amin[n] = res_val; amax[n] = res_val
            continue
    
        #Calculate TSLA
        valid_snow = snow_elevs[:snow_count]
        valid_snow.sort()

        #percentiles
        p_idx = int(snow_count * 0.02)
        if p_idx < 0:
            p_idx = 0
        if p_idx >= snow_count:
            p_idx = snow_count -1

        tsla_final = valid_snow[p_idx]
    
        transition_zone = valid_snow[:p_idx+1]
        amed[n] = tsla_final
        amean[n] = np.mean(transition_zone)
        amin[n] = np.min(transition_zone)
        amax[n] = np.max(transition_zone)
        astd[n] = np.std(transition_zone)

    return (amed, amean, astd, amax, amin, flag)

#Needs work? Method is not really reliable for a 2D distributed simulation due to large differences
@njit 
def tsl_method_conservative(albedos, hgts, mask, min_albedo, min_elevs_t, max_elevs_t):
    """
    Finds the absolute lowest snow pixel. Finds the absolute highest ice pixel that is BELOW snow.
    Calculates the average of the two as TSLA.
    Args:
        albedos(3D): time lat lon
        hgts (2D): elevations
        mask (2D): static_mask
        min_albedo (float): threshold
        min_elevs_t (1D): dynamic min elev
        max_elevs_t (1D): dynamic max elev
    """
    ntime = albedos.shape[0]
    rows, cols = hgts.shape

    amed = np.full(ntime, np.nan)
    amean = np.full(ntime, np.nan)
    astd = np.full(ntime, np.nan)
    amax = np.full(ntime, np.nan)
    amin = np.full(ntime, np.nan)
    flag = np.full(ntime)

    for n in range(ntime):
        current_min = min_elevs_t[n]
        current_max = max_elevs_t[n]
        if np.isnan(current_min):
            flag[n] = 3
            continue
        # collect valid pxiels, flatten for speed
        # collect ice to filter later
        snow_elevs = np.zeros(rows*cols)
        ice_elevs = np.zeros(rows*cols)
        ice_count = 0
        snow_count = 0

        for r in range(rows):
            for c in range(cols):
                if mask[r,c] == 1:
                    h = hgts[r,c]
                    if h>= current_min and h <= current_max:
                        alb = albedos[n,r,c]

                        if not np.isnan(alb):
                            if alb >= min_albedo:
                                #track minimum snow
                                snow_elevs[snow_count] = h
                                snow_count +=1
                            else:
                                #store ice
                                ice_elevs[ice_count] = h
                                ice_count += 1

        #handle extremes
        if snow_count == 0:
            #bare ice - tsla is top
            res_val = current_max
            flag[n] = 2
            amed[n] = res_val; amean[n] = res_val
            amin[n] = res_val; amax[n] = res_val
            continue

        if ice_count == 0:
            #full snow - tsla is bottom
            res_val = current_min
            flag[n] = 1
            amed[n] = res_val; amean[n] = res_val
            amin[n] = res_val; amax[n] = res_val
            continue

        #filter ice (wind scour)
        valid_snow = snow_elevs[:snow_count]
        valid_snow.sort()

        snow_min = valid_snow[0]
        snow_median = valid_snow[int(snow_count*0.5)]
        
        valid_ice = ice_elevs[:ice_count]
        ice_top_filtered = -999999.0
        found_valid_ice = False

        for i in range(ice_count):
            val = valid_ice[i]
            if val < snow_median:
                if val > ice_top_filtered:
                    ice_top_filtered = val
                found_valid_ice = True

        if found_valid_ice:
            tsla_final = (ice_top_filtered + snow_min) / 2.0
        else:
            tsla_final = snow_min
            if np.abs(tsla_final - current_min) < 1:
                flag[n] = 1

        if tsla_final < current_min:
            tsla_final = current_min
        if tsla_final > current_max:
            tsla_final = current_max

        amed[n] = tsla_final
        amean[n] = tsla_final
        amin[n] = tsla_final
        amax[n] = tsla_final
        astd[n] = 0.0

    return (amed, amean, astd, amax, amin, flag)

""" ARCHIVED and not functional!
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
"""

def calculate_tsl_byhand(albedos, hgts, mask, min_albedo, tsl_method, tsl_normalize, n_points_3d=None):
    
    timesteps = albedos.shape[0]
    min_elevs_t = np.full(timesteps, np.nan)
    max_elevs_t = np.full(timesteps, np.nan)

    if n_points_3d is not None:
        for t in range(timesteps):
            current_mask = n_points_3d[t,:,:]
            valid_mask = (mask == 1) & (current_mask > 0) & (~np.isnan(current_mask))
            if np.any(valid_mask):
                valid_hgts = hgts[valid_mask]
                min_elevs_t[t] = np.nanmin(valid_hgts)
                max_elevs_t[t] = np.nanmax(valid_hgts)
    else:
        static_min = np.nanmin(np.where(mask==1, hgts, np.nan))
        static_max = np.nanmax(np.where(mask==1, hgts, np.nan))

        max_elevs_t[:] = static_max
        min_elevs_t[:] = static_min
    print("Calculating TSLA using {}. Normalization is set to {}.".format(tsl_method, tsl_normalize))
    if tsl_method == 'mantra':
        amed, amean, astd, amax, amin, flag = tsl_method_mantra(albedos, hgts, mask, min_albedo, min_elevs_t, max_elevs_t)
    elif tsl_method == 'conservative':
        amed, amean, astd, amax, amin, flag = tsl_method_conservative(albedos, hgts, mask, min_albedo, min_elevs_t, max_elevs_t)
    else:
        print("Warning. No method passed. Script will terminate with error.")
    if tsl_normalize:
        #Start normalizing:
        elev_range = max_elevs_t - min_elevs_t
        #elev_range[elev_range == 0] = 1.0 #safety

        amed_norm = (amed - min_elevs_t) / elev_range
        amean_norm = (amean - min_elevs_t) / elev_range
        astd_norm = (astd - min_elevs_t) / elev_range
        amax_norm = (amax - min_elevs_t) / elev_range
        amin_norm = (amin - min_elevs_t) / elev_range
        return (amed_norm, amean_norm, astd_norm, amax_norm, amin_norm, flag)
    else:
        return (amed, amean, astd, amax, amin, flag)
        

def create_tsl_df(var_to_check, cos_output, min_albedo, tsl_method, tsl_normalize, n_points_arg):
    times = datetime.now()
    if var_to_check == "ALBEDO":
        amed,amean,astd,amax,amin,flag = calculate_tsl_byhand(
            cos_output.ALBEDO.values,
            cos_output.HGT.values,
            cos_output.MASK.values,
            min_albedo,
            tsl_method,
            tsl_normalize,
            n_points_arg)
    elif var_to_check == "SNOWHEIGHT":
        amed,amean,astd,amax,amin,flag = calculate_tsl_byhand(
            cos_output.SNOWHEIGHT.values,
            cos_output.HGT.values,
            cos_output.MASK.values,
            min_albedo,
            tsl_method,
            tsl_normalize,
            n_points_arg)
    #print(amed)
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

