"""
Summary of the round-2 (narrow) LHS: same output as create_summary_lhs-wide.py.

Differences from the round-1 script, all marked with # R2:
  1. reads {glacier}_LHS_Round2_Design.csv, which already holds LINEAR values
     -> no np.exp() on rrr_factor_summer/_winter and ws_factor
  2. that CSV was written with index=False -> no index_col=0
  3. rows are matched primarily via the _num suffix in the filename
     (run_id = ID_OFFSET + global_id), with the parameter values used only as a
     cross-check. Round 2 resamples from ~100 behavioural points, so several
     rows can be near-identical and parameter-only matching may be ambiguous.

Resume mode (marked with # RESUME):
  If the output CSV already exists and RESUME = True, the rows that are already
  filled (mb not NaN) are taken over and the matching .nc files are skipped.
  Only empty rows are (re)computed. With RESUME = False, or if no output file
  exists yet, the script behaves exactly as before.
"""

import pandas as pd
import pathlib
import numpy as np
import xarray as xr
from numba import njit

glacier = "Abramov"

# R2: point this at the folder that holds ONLY the round-2 .nc files
path = f"/data/scratch/richteny/thesis/cosipy_test_space/data/output/{glacier}/"

# R2: must match ID_OFFSET in cosipy_lhs_round2.py
ID_OFFSET = 0

# RESUME: output file, resume switch and intermediate saving
out_path = f"/data/scratch/richteny/thesis/cosipy_test_space/{glacier}_LHS-narrowB_design_filled.csv"
RESUME = False        # False -> always compute everything from scratch
SAVE_EVERY = 0     # write intermediate results every N newly filled files (0 = only at the end)

alb_obs_data = xr.open_dataset(f"/data/scratch/richteny/Ren_21_Albedo/{glacier}_hrz-merged_mean-albedos.nc")
alb_obs_data = alb_obs_data.sortby("time")
alb_obs_data = alb_obs_data.sel(time=slice("1990-01-01", "2023-12-31"))

tsla_obs = pd.read_csv(f"/data/scratch/richteny/thesis/cosipy_test_space/data/input/{glacier}/snowlines/{glacier}_TSLA_fixed-1990-2024.csv",
                       parse_dates=True, index_col="LS_DATE")
tsla_obs = tsla_obs.loc["1990-01-01":"2023-12-31"]

# R2: design file, already linear, written without index
df = pd.read_csv(f"/data/scratch/richteny/thesis/cosipy_test_space/{glacier}_LHS-narrowB_design.csv")
if "global_id" not in df.columns:
    df["global_id"] = range(len(df))
df = df.set_index("global_id", drop=False)

# R2: fixed order -- must match the tuple order in parse_param_key_from_filename.
# Defined explicitly rather than taken from df.columns, so a reordered CSV
# cannot silently break the cross-check.
param_cols = ['rrr_factor_summer', 'alb_snow', 'alb_firn', 'albedo_depth',
              'bias_LWin', 'ws_factor', 'bias_T2', 't_wet', 'rrr_factor_winter']
print(param_cols)

df_params = df.copy()
n = len(df_params)

n_tsla = len(tsla_obs.index)
n_alb = len(alb_obs_data.time)

tsla_names = [f"tsla{i}" for i in range(1, n_tsla + 1)]
alb_names = [f"alb{i}" for i in range(1, n_alb + 1)]

tsla_cols = pd.DataFrame(np.nan, index=df_params.index, columns=tsla_names)
alb_cols = pd.DataFrame(np.nan, index=df_params.index, columns=alb_names)

df_params = pd.concat([df_params, tsla_cols, alb_cols], axis=1)
df_params["mb"] = np.nan
df_params["filename"] = pd.Series(np.nan, index=df_params.index, dtype=object)
df_params["param_mismatch"] = np.nan

result_cols = tsla_names + alb_names + ["mb", "filename"]


# RESUME: take over already filled rows from an existing output file
def load_previous(fpath):
    prev = pd.read_csv(fpath, index_col=0)
    # the old file has global_id both as index and as column -> pandas names
    # the second one "global_id.1" when reading it back
    if "global_id.1" in prev.columns:
        prev = prev.rename(columns={"global_id.1": "global_id"})
    prev.index = prev.index.astype(int)
    prev.index.name = "global_id"
    return prev


already_filled = set()
if RESUME and pathlib.Path(out_path).exists():
    prev = load_previous(out_path)
    missing_cols = [c for c in result_cols + param_cols if c not in prev.columns]
    if missing_cols:
        print(f"RESUME: existing file has a different structure ({len(missing_cols)} columns missing, "
              f"e.g. {missing_cols[:3]}) -> ignoring it and starting from scratch.")
    else:
        common = prev.index[prev["mb"].notna()].intersection(df_params.index)
        # only take over rows whose parameters still agree with the design file
        same = np.isclose(prev.loc[common, param_cols].values.astype(float),
                          df_params.loc[common, param_cols].values.astype(float),
                          atol=1e-8).all(axis=1)
        if (~same).any():
            print(f"RESUME: {int((~same).sum())} filled rows differ from the design file "
                  f"and will be recomputed.")
        keep = common[same]
        df_params.update(prev.loc[keep, result_cols])
        already_filled = set(int(g) for g in keep)
        print(f"RESUME: took over {len(already_filled)} filled rows from {out_path}")
else:
    print("No previous output used -> computing everything.")


def parse_param_key_from_filename(fname):
    tail = fname.split("_RRR-")[1]
    tail = tail.split("_num")[0]
    vals = [float(v) for v in tail.split("_")]
    """
    Filename (Bougamont):
    0 RRR_factor, 1 alb_snow, 2 alb_ice, 3 alb_firn, 4 t wet
    5 t dry, 6 t K, 7 alb depth, 8 roughness fresh snow, 9 roughness ice,
    10 roughness firn, 11 aging factor roughness, 12 bias LWin, 13 WS_factor,
    14 bias T2, 15 center_snow_transfer, 16 min snowfall, 17 rrr summer, 18 rrr winter
    """
    return np.array([
        vals[17],  # rrr_factor summer
        vals[1],   # alb_snow
        vals[3],   # alb_firn
        vals[7],   # alb_depth
        vals[12],  # bias lwin
        vals[13],  # ws factor
        vals[14],  # bias t2
        vals[4],   # t wet
        vals[18],  # rrr_factor winter
    ])


def parse_run_id_from_filename(fname):
    """R2: COSIPY counts from 1, so _num{k} belongs to run_id k-1."""
    return int(fname.split("_num")[-1]) - 1


def prereq_res(ds):
    t = np.asarray(ds.time.values, dtype="datetime64[ns]")
    secs = t.astype("int64")
    dates = pd.to_datetime(np.unique(t.astype("datetime64[D]")))
    clean_day_vals = dates.values.astype("datetime64[ns]").astype("int64")
    assert abs(np.log10(max(clean_day_vals[0], 1)) - np.log10(max(secs[0], 1))) < 0.5, "time units don't match"
    return dates, clean_day_vals, secs


@njit
def resample_by_hand(vals, secs, time_vals):
    ntime = vals.shape[0]
    ndays = len(time_vals)
    day_next = np.int64(86_400_000_000_000)   # ns
    out = np.zeros(ndays)
    count = np.zeros(ndays)
    day_idx = 0
    day_end = time_vals[0] + day_next
    for t in range(ntime):
        ts = secs[t]
        while day_idx < ndays - 1 and ts >= day_end:
            day_idx += 1
            day_end = time_vals[day_idx] + day_next
        v = vals[t]
        if not np.isnan(v):
            out[day_idx] += v
            count[day_idx] += 1
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
        except Exception:
            ref_weights = ncfile["N_Points"]
        ref_area_total = ref_weights.sum()
        total_mass_change = (ncfile["MB"] * ref_weights).sum(dim=["lat", "lon"])
        weighted_mb = total_mass_change / ref_area_total
        dfmb = weighted_mb.to_dataframe(name="weighted_mb")
        annual_mb = dfmb.resample("1YE").sum()
        return np.nanmean(annual_mb["weighted_mb"].values)
    else:
        ref_weights = ncfile["N_Points"].sum(dim=["lat", "lon"])
        alb_total = (ncfile["ALBEDO"] * ncfile["N_Points"]).sum(dim=["lat", "lon"])
        weighted_alb = alb_total / ref_weights
        dates, clean_day_vals, secs = prereq_res(weighted_alb)
        resampled_alb_vals = resample_by_hand(weighted_alb.data, secs, clean_day_vals).copy()
        resampled_alb = xr.DataArray(resampled_alb_vals, coords={"time": dates},
                                     dims=["time"], name="ALBEDO_weighted")
        return resampled_alb.sortby("time").sel(time=albobs.time)


n_done = n_nomatch = n_mismatch = n_skipped = n_error = 0
for fp in pathlib.Path(path).glob('*.nc'):
    name = str(fp.stem)
    csv_name = "tsla_" + name.lower() + ".csv"

    # R2: primary match via the run id in the filename
    try:
        run_id = parse_run_id_from_filename(name)
        param_vals = parse_param_key_from_filename(name)
    except Exception:
        print("EXCEPTION FOUND:", fp.name)
        continue

    gid = run_id - ID_OFFSET
    if gid not in df_params.index:
        print("NO MATCH (id not in design):", fp.name)
        n_nomatch += 1
        continue

    # RESUME: row already filled in a previous run -> skip
    if gid in already_filled:
        n_skipped += 1
        continue

    # R2: cross-check the parameters. The filename rounds to 4 decimals.
    design_vals = df_params.loc[gid, param_cols].values.astype(float)
    if not np.allclose(np.round(design_vals, 4), param_vals, atol=1.5e-4):
        diff = np.abs(np.round(design_vals, 4) - param_vals)
        worst = param_cols[int(np.argmax(diff))]
        print(f"PARAM MISMATCH {fp.name}: largest difference in {worst} "
              f"({diff.max():.5f}) -- wrong design file or ID_OFFSET?")
        df_params.at[gid, "param_mismatch"] = float(diff.max())
        n_mismatch += 1
        continue

    # RESUME: a file that is still being written (or whose TSLA csv is missing)
    # is skipped and simply retried in the next run instead of aborting everything
    try:
        with xr.open_dataset(fp) as ds_raw:
            ds = ds_raw.sel(time=slice("1990-01-01", None))
            mb = compute_glacier_mean(ds.sel(time=slice("2000-01-01T00:00", "2019-12-31T23:00")),
                                      "MB", "2000-01-01T01:00", None)
            albsim = compute_glacier_mean(ds, "ALBEDO", None, alb_obs_data)
            alb_vals = np.asarray(albsim.data[:n_alb])

        snowlinesim = pd.read_csv(path + csv_name, parse_dates=True, index_col="time")
        tsla_sim = snowlinesim.loc[snowlinesim.index.isin(tsla_obs.index)]
        tsla_vals = tsla_sim["Med_TSL"].values[:n_tsla]
    except Exception as e:
        print(f"ERROR reading {fp.name} ({type(e).__name__}: {e}) -- left empty, retried next run")
        n_error += 1
        continue

    df_params.loc[gid, "mb"] = mb
    df_params.loc[gid, "filename"] = name
    df_params.loc[gid, tsla_names] = tsla_vals
    df_params.loc[gid, alb_names] = alb_vals
    n_done += 1

    if n_done % 100 == 0:
        print(f"Processed {n_done} new files ({n_skipped} skipped so far).")

    # RESUME: intermediate save, so an aborted job does not lose everything
    if SAVE_EVERY and n_done % SAVE_EVERY == 0:
        df_params.to_csv(out_path)

print(f"\nnewly filled: {n_done}   taken over/skipped: {n_skipped}   no match: {n_nomatch}"
      f"   parameter mismatch: {n_mismatch}   read errors: {n_error}"
      f"   still missing: {int(df_params['mb'].isna().sum())} of {n}")

df_params.to_csv(out_path)
