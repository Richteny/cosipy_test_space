import pandas as pd
import pathlib
import numpy as np
import xarray as xr
from numba import njit

path = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/"

alb_obs_data = xr.open_dataset("/data/scratch/richteny/Ren_21_Albedo/Halji_hrz-merged_mean-albedos.nc")
alb_obs_data = alb_obs_data.sortby("time")

tsla_obs = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/data/input/Halji/snowlines/Halji_TSLA_fixed-1990-2025.csv", parse_dates=True, index_col="LS_DATE")
tsla_obs = tsla_obs.loc["1990-01-01":"2022-12-31"]

df = pd.read_csv("/data/scratch/richteny/for_emulator/Halji/LHS-narrow/LHS_Posterior_Design_Buffered.csv")
param_cols = df.columns
df_params = df.copy()
#df_params[param_cols] = df_params[param_cols].round(4)

n = len(df_params)

n_tsla = len(tsla_obs.index)
n_alb = len(alb_obs_data.time)

tsla_cols = pd.DataFrame(np.nan, index=df_params.index,
                         columns=[f"tsla{i}" for i in range(1, n_tsla+1)])

alb_cols = pd.DataFrame(np.nan, index=df_params.index,
                        columns=[f"alb{i}" for i in range(1, n_alb+1)])

df_params = pd.concat([df_params, tsla_cols, alb_cols], axis=1)
df_params["mb"] = np.nan

df_params["filename_tolerance_match"] = np.nan
df_params["param_key"] = list(map(tuple, df_params[param_cols].values))

def find_row(df, cols, values, atol=5e-5):
    rounded_df = df[cols].round(4)
    rounded_vals = np.round(values, 4)
    mask_exact = (rounded_df.values == rounded_vals).all(axis=1)
    idxs = np.where(mask_exact)[0]
    if len(idxs) == 1:
        return idxs[0], "exact"

    arr = df[cols].values
    mask_tol = np.all(np.isclose(arr, values, atol=atol), axis=1)
    idxs = np.where(mask_tol)[0]
    if len(idxs) == 1:
        return idxs[0], "tolerance"
    else:
        print(idxs)

    return None, None

def parse_param_key_from_filename(fname):
    tail = fname.split("_RRR-")[1]
    tail = tail.split("_num")[0]
    vals = [float(v) for v in tail.split("_")]

    return tuple([
        round(vals[0], 4), #rrr_factor
        round(vals[2], 4), #alb_ice
        round(vals[1], 4), #alb_snow
        round(vals[3], 4), #alb_firn
        round(vals[4], 4), #alb_aging
        round(vals[5], 4), #alb_depth
        round(vals[7], 4), #roughness ice
        round(vals[10], 4), #LWin factor
        round(vals[11], 4), #WS factor
        round(vals[13], 4), #center snow
    ])


def prereq_res(ds):
    time_vals = pd.to_datetime(ds.time.values)
    unique_dates = np.unique(time_vals.date)
    holder = np.zeros(len(unique_dates))
    secs = ds.time.values.astype("int64")
    dates_pd = pd.to_datetime(unique_dates)
    clean_day_vals = dates_pd.astype("int64").values
    return (dates_pd, clean_day_vals, secs, holder)

@njit
def resample_by_hand(holder,vals,secs,day_starts):
    day_len = 24*60*60*1e9
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

def compute_glacier_mean(ncfile, target_var, time_start_mb, albobs):
    if target_var == "MB":
        ref_weights = ncfile["N_Points"].sel(time=time_start_mb, method="nearest")
        ref_area_total = ref_weights.sum()
        total_mass_change = (ncfile["MB"] * ref_weights).sum(dim=["lat","lon"])
        weighted_mb = total_mass_change / ref_area_total
        dfmb = weighted_mb.to_dataframe(name="weighted_mb")
        annual_mb = dfmb.resample("1Y").sum()
        geod_mb = np.nanmean(annual_mb["weighted_mb"].values)
        return geod_mb
    else:
        ref_weights = ncfile["N_Points"].sum(dim=["lat","lon"])
        alb_total = (ncfile["ALBEDO"] * ncfile["N_Points"]).sum(dim=["lat","lon"])
        weighted_alb = alb_total / ref_weights
        dates,clean_day_vals,secs,holder = prereq_res(weighted_alb)
        resampled_alb_vals = resample_by_hand(holder, weighted_alb.data, secs, clean_day_vals).copy()
        resampled_alb = xr.DataArray(resampled_alb_vals, coords={"time":dates}, dims=["time"], name="ALBEDO_weighted")
        result = resampled_alb.sortby("time")
        result = result.sel(time=albobs.time)
        return result

for fp in pathlib.Path(path).glob('*.nc'):
    name = str(fp.stem)
    #print(name)
    csv_name = "tsla_" + name.lower() + ".csv"
    try:
        param_vals = parse_param_key_from_filename(name)
    except Exception:
        print("EXCEPTION FOUND.")
        continue

    idx, method = find_row(df_params, param_cols, param_vals)
    if idx is None:
        print("NO MATCH:", fp.name)
        continue

    if method == "tolerance":
        df_params.at[idx, "filename_tolerance_match"] = name

    ds = xr.open_dataset(fp).sel(time=slice("1990-01-01",None))
    mb = compute_glacier_mean(ds.sel(time=slice("2000-01-01T00:00","2020-01-01T00:00")),"MB", "2000-01-01T01:00", None)
    albsim = compute_glacier_mean(ds,"ALBEDO",None,alb_obs_data)

    snowlinesim = pd.read_csv(path+csv_name, parse_dates=True, index_col="time")
    tsla_sim = snowlinesim.loc[snowlinesim.index.isin(tsla_obs.index)]
     
    df_params.loc[idx, "mb"] = mb
    df_params.loc[idx, [f"tsla{i}" for i in range(1, n_tsla+1)]] = tsla_sim["Med_TSL"].values[:n_tsla]
    df_params.loc[idx, [f"alb{i}" for i in range(1, n_alb+1)]] = albsim.data[:n_alb]

df_params.to_csv("/data/scratch/richteny/thesis/cosipy_test_space/LHS-narrow_filled_params.csv") 
