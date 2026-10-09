import pandas as pd
import pathlib
import numpy as np
import xarray as xr
from numba import njit

glacier = "Dongkemadi"
path = f"/data/scratch/richteny/thesis/cosipy_test_space/data/output/{glacier}/"

#alb_obs_data = xr.open_dataset("/data/scratch/richteny/Ren_21_Albedo/Halji_hrz-merged_mean-albedos.nc")
alb_obs_data = xr.open_dataset(f"/data/scratch/richteny/Ren_21_Albedo/{glacier}_hrz-merged_mean-albedos.nc")
alb_obs_data = alb_obs_data.sortby("time")
alb_obs_data = alb_obs_data.sel(time=slice("1990-01-01","2023-12-31"))

#tsla_obs = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/data/input/Halji/snowlines/Halji_TSLA_fixed-1990-2025.csv", parse_dates=True, index_col="LS_DATE")
tsla_obs = pd.read_csv(f"/data/scratch/richteny/thesis/cosipy_test_space/data/input/{glacier}/snowlines/{glacier}_TSLA_fixed-1990-2024.csv", parse_dates=True, index_col="LS_DATE")
tsla_obs = tsla_obs.loc["1990-01-01":"2023-12-31"]

#df = pd.read_csv("/data/scratch/richteny/for_emulator/Halji/LHS-narrow/LHS_Posterior_Design_Buffered.csv")
df = pd.read_csv(f"/data/scratch/richteny/thesis/cosipy_test_space/{glacier}_LHS-wide-master.csv", index_col=0)
df['rrr_factor_summer'] = np.exp(df['rrr_factor_summer'])
df['rrr_factor_winter'] = np.exp(df['rrr_factor_winter'])
df['ws_factor'] = np.exp(df['ws_factor'])

param_cols = df.columns
param_cols = [x for x in param_cols if 'global_id' not in x]
print(param_cols)
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
#order must match order below

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

    """
    Filename if Bougamont: 
    0 RRR_factor, 1 alb_snow, 2 alb_ice, 3 alb_firn, 4 t wet
    5 t dry, 6 t K, 7 alb depth, 8 roughness fresh snow, 9 roughness ice, 10 roughness firn, 11 aging factor roughness
    12 bias LWin, 13 WS_factor, 14 bias T2, 15 center_snow_transfer, 16 min snowfall, 17 rrr summer, 18 rrr winter

    # Order in CSV: rrr-factor-summer, alb-snow, alb-firn, alb-depth, bias-LWin, ws-factor, bias-t2, t-wet, rrr-factor-winter, global-id
    """

    return tuple([
        round(vals[17], 4), #rrr_factor summer
        #round(vals[2], 4), #alb_ice now fixed
        round(vals[1], 4), #alb_snow
        round(vals[3], 4), #alb_firn
        round(vals[7], 4), #alb_depth
        round(vals[12], 4), #bias lwin
        round(vals[13], 4), #ws factor
        round(vals[14], 4), #bias t2
        round(vals[4], 4), #t wet
        round(vals[18], 4), #rrr_factor winter
    ])
"""
    return tuple([
        round(vals[0], 4), #rrr_factor
        round(vals[2], 4), #alb_ice
        round(vals[1], 4), #alb_snow
        round(vals[3], 4), #alb_firn
        round(vals[4], 4), #alb_aging
        round(vals[5], 4), #alb_depth
        round(vals[13], 4), #center snow
        round(vals[7], 4), #roughness ice
        round(vals[10], 4), #LWin factor
        round(vals[11], 4), #WS factor
        round(vals[12], 4), #bias t2
        #round(vals[13], 4), #center snow
    ])
"""

def prereq_res(ds):
    t = np.asarray(ds.time.values, dtype="datetime64[ns]")
    secs = t.astype("int64")
    dates = pd.to_datetime(np.unique(t.astype("datetime64[D]")))
    clean_day_vals = dates.values.astype("datetime64[ns]").astype("int64")
    assert abs(np.log10(max(clean_day_vals[0], 1)) - np.log10(max(secs[0], 1))) < 0.5, "time units don't match"
    return dates, clean_day_vals, secs


@njit
def resample_by_hand(vals, secs, time_vals):
    # vals is 1D, so shape is just (ntime,)
    ntime = vals.shape[0] 
    ndays = len(time_vals)

    day_next = np.int64(86_400_000_000_000)   # ns, integer nanoseconds

    # Arrays only need to be 1D now
    out = np.zeros(ndays)
    count = np.zeros(ndays)

    day_idx = 0
    day_end = time_vals[0] + day_next

    for t in range(ntime):
        ts = secs[t]

        while day_idx < ndays - 1 and ts >= day_end:
            day_idx += 1
            day_end = time_vals[day_idx] + day_next

        # No more j, k loops. Just grab the 1D value.
        v = vals[t]
        if not np.isnan(v):
            out[day_idx] += v
            count[day_idx] += 1

    # Mean calculation for 1D
    for d in range(ndays):
        if count[d] > 0:
            out[d] /= count[d]
        else:
            out[d] = np.nan
            
    return out

def compute_glacier_mean(ncfile, target_var, time_start_mb, albobs):
    if target_var == "MB":
        try:
            ref_weights = ncfile["N_Points"].sel(time=time_start_mb, method="nearest")
        except:
            ref_weights = ncfile["N_Points"]
        ref_area_total = ref_weights.sum()
        total_mass_change = (ncfile["MB"] * ref_weights).sum(dim=["lat","lon"])
        weighted_mb = total_mass_change / ref_area_total
        dfmb = weighted_mb.to_dataframe(name="weighted_mb")
        annual_mb = dfmb.resample("1YE").sum()
        geod_mb = np.nanmean(annual_mb["weighted_mb"].values)
        return geod_mb
    else:
        ref_weights = ncfile["N_Points"].sum(dim=["lat","lon"])
        alb_total = (ncfile["ALBEDO"] * ncfile["N_Points"]).sum(dim=["lat","lon"])
        weighted_alb = alb_total / ref_weights
        dates,clean_day_vals,secs = prereq_res(weighted_alb)
        resampled_alb_vals = resample_by_hand(weighted_alb.data, secs, clean_day_vals).copy()
        resampled_alb = xr.DataArray(resampled_alb_vals, coords={"time":dates}, dims=["time"], name="ALBEDO_weighted")
        result = resampled_alb.sortby("time")
        result = result.sel(time=albobs.time)
        return result

i = 0
for fp in pathlib.Path(path).glob('*.nc'):
    if i % 100 == 0:
        print(f"Processed {i} files.")
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
        print("TOLERANCE:", fp.name)
        df_params.at[idx, "filename_tolerance_match"] = name

    ds = xr.open_dataset(fp).sel(time=slice("1990-01-01",None))
    mb = compute_glacier_mean(ds.sel(time=slice("2000-01-01T00:00","2019-12-31T23:00")),"MB", "2000-01-01T01:00", None)
    albsim = compute_glacier_mean(ds,"ALBEDO",None,alb_obs_data)

    snowlinesim = pd.read_csv(path+csv_name, parse_dates=True, index_col="time")
    tsla_sim = snowlinesim.loc[snowlinesim.index.isin(tsla_obs.index)]
     
    df_params.loc[idx, "mb"] = mb
    df_params.loc[idx, [f"tsla{i}" for i in range(1, n_tsla+1)]] = tsla_sim["Med_TSL"].values[:n_tsla]
    df_params.loc[idx, [f"alb{i}" for i in range(1, n_alb+1)]] = albsim.data[:n_alb]
    i +=1

df_params.to_csv(f"/data/scratch/richteny/thesis/cosipy_test_space/{glacier}_LHS-wide_filled_params.csv") 
