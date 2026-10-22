"""
Summary of a set of COSIPY runs (wide LHS, narrow LHS, posterior or validation lists). One row per run in
OUT_CSV with the parameters, mb, tsla1..N and alb1..N. Every other COSIPY output of the run goes into one small
NetCDF per run in SUMMARY_DIR (the CSV column summary_file names it):

  * daily glacier-wide series (dim day, whole run), N_Points-weighted like the albedo:
      daily sums   MB, surfM, subM, SNOWFALL                         (m w.e. / m per day)
      daily means  ALBEDO, TS, SNOWHEIGHT, LWC_00_10, LWC_10_50, LWC_DEEP
  * per elevation band at the Sentinel-1 pass hours (dims date x pass x point, from PASS_START), for the SAR
    comparison: LWC_00_10, LWC_10_50, LWC_DEEP, SNOWHEIGHT, melt_6h, melt_24h (surfM + subM summed over the
    6 / 24 model steps up to the pass), N_Points; HGT per point.
    Pass hour = the model step (hour-ending labels) containing ~06:00 (descending) and ~18:00 (ascending) local
    solar time at the glacier longitude -- what sar_wet_compare.py picks for the actual acquisitions, up to
    +-1 h where the true pass time lies close to a full hour (printed for the first file; check against t_ms
    of the GEE tables).

Matching output files to design rows (MATCH):
  "run_id"  (default) via the _num suffix: every runner writes _num{ID_OFFSET + global_id + 1}
            (cosipy_lhs-wide_fromlist.py: ID_OFFSET 0; run_cosipy_from_list.py: its ID_OFFSET, e.g. 100000).
            The parameters in the file name are only a cross-check (mismatch -> row left empty, reported).
            Safe for narrow / posterior lists, where several rows can be near-identical.
  "params"  via the parameter values in the file name (old behaviour; for files whose numbers do not follow
            the design ids).

Resume (RESUME = True): rows of an existing OUT_CSV that are already filled (mb not NaN, summary NetCDF
present, parameters unchanged) are taken over and their files skipped. A file that cannot be read (still being
written, TSLA csv missing) is left empty and retried on the next call. SAVE_EVERY writes intermediate results.

Read all runs together:
    S = xr.open_mfdataset(f"{SUMMARY_DIR}/*.nc", combine="nested", concat_dim="run")
S.run is the global_id of the design row.
"""
import pandas as pd
import pathlib
import numpy as np
import xarray as xr
from numba import njit

# ============================================================================
# CONFIGURATION
# ============================================================================
glacier = "Dongkemadi"
TAG = "wide"                       # name of the set: output CSV / summary folder are named after it
BASE = "/data/scratch/richteny/thesis/cosipy_test_space"
path = f"{BASE}/data/output/{glacier}/"                       # folder with the .nc files (and tsla_*.csv) of THIS set
DESIGN_CSV = f"{BASE}/{glacier}_LHS-wide-prior-master.csv"    # one row per run, column global_id (else row number)
LOG_COLS = ['rrr_factor_summer', 'rrr_factor_winter', 'ws_factor']   # design columns stored as ln (wide LHS);
                                                                      # [] for linear lists (narrow, posterior)
MATCH = "run_id"                   # "run_id" (file number) or "params" (parameter values)
ID_OFFSET = 0                      # run_id = ID_OFFSET + global_id (as in the runner)
OUT_CSV = f"{BASE}/{glacier}_LHS-{TAG}_filled_params.csv"
SUMMARY_DIR = pathlib.Path(f"{BASE}/{glacier}_LHS-{TAG}_summary")
RESUME = False                     # True: take over filled rows of an existing OUT_CSV
SAVE_EVERY = 0                     # write OUT_CSV every N newly filled runs (0 = only at the end)
# ============================================================================

#alb_obs_data = xr.open_dataset("/data/scratch/richteny/Ren_21_Albedo/Halji_hrz-merged_mean-albedos.nc")
alb_obs_data = xr.open_dataset(f"/data/scratch/richteny/Ren_21_Albedo/{glacier}_hrz-merged_mean-albedos.nc")
alb_obs_data = alb_obs_data.sortby("time")
alb_obs_data = alb_obs_data.sel(time=slice("1990-01-01","2023-12-31"))

#tsla_obs = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/data/input/Halji/snowlines/Halji_TSLA_fixed-1990-2025.csv", parse_dates=True, index_col="LS_DATE")
tsla_obs = pd.read_csv(f"{BASE}/data/input/{glacier}/snowlines/{glacier}_TSLA_fixed-1990-2024.csv", parse_dates=True, index_col="LS_DATE")
tsla_obs = tsla_obs.loc["1990-01-01":"2023-12-31"]

# Position of each parameter in the output file name (after "_RRR-", Bougamont albedo layout):
# 0 RRR_factor, 1 alb_snow, 2 alb_ice, 3 alb_firn, 4 t wet, 5 t dry, 6 t K, 7 alb depth, 8 roughness fresh snow,
# 9 roughness ice, 10 roughness firn, 11 aging factor roughness, 12 bias LWin, 13 WS_factor, 14 bias T2,
# 15 center_snow_transfer, 16 min snowfall, 17 rrr summer, 18 rrr winter, 19 T2 bias summer, 20 T2 bias winter
# (19/20 only in runs made after the seasonal T2 bias was added; without a split both equal bias T2),
# 21 precip lapse rate [1/m], 22 precip peak elevation [m] (runs from 22 Oct 2026 on).
FNAME_INDEX = {'rrr_factor': 0, 'alb_snow': 1, 'alb_ice': 2, 'alb_firn': 3, 't_wet': 4, 't_dry': 5, 't_K': 6,
               'albedo_depth': 7, 'roughness_fresh_snow': 8, 'roughness_ice': 9, 'roughness_firn': 10,
               'aging_factor_roughness': 11, 'bias_LWin': 12, 'ws_factor': 13, 'bias_T2': 14, 'center_snow': 15,
               'minimum_snowfall': 16, 'rrr_factor_summer': 17, 'rrr_factor_winter': 18,
               'bias_T2_summer': 19, 'bias_T2_winter': 20, 'precip_lapse_rate': 21, 'precip_peak_elevation': 22}

# ---------------- design ----------------------------------------------------------------------------------------
df = pd.read_csv(DESIGN_CSV)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])      # old index column
if "global_id" not in df.columns:
    df["global_id"] = range(len(df))
df["global_id"] = df["global_id"].astype(int)
df = df.set_index("global_id", drop=False)
for c in LOG_COLS:
    df[c] = np.exp(df[c])

# parameter columns = design columns that appear in the file name (order of the design CSV)
param_cols = [c for c in df.columns if c in FNAME_INDEX]
other = [c for c in df.columns if c not in FNAME_INDEX and c != "global_id"]
print("parameters:", param_cols)
if other:
    print("design columns not in the file name (kept, not checked):", other)

df_params = df.copy()
n = len(df_params)

n_tsla = len(tsla_obs.index)
n_alb = len(alb_obs_data.time)
tsla_names = [f"tsla{i}" for i in range(1, n_tsla + 1)]
alb_names = [f"alb{i}" for i in range(1, n_alb + 1)]

df_params = pd.concat([df_params,
                       pd.DataFrame(np.nan, index=df_params.index, columns=tsla_names),
                       pd.DataFrame(np.nan, index=df_params.index, columns=alb_names)], axis=1)
df_params["mb"] = np.nan
df_params["filename"] = pd.Series(None, index=df_params.index, dtype=object)
df_params["summary_file"] = pd.Series(None, index=df_params.index, dtype=object)
df_params["param_mismatch"] = np.nan              # largest |design - file name| when the cross-check fails
df_params["filename_tolerance_match"] = pd.Series(None, index=df_params.index, dtype=object)   # MATCH = "params"
result_cols = tsla_names + alb_names + ["mb", "filename", "summary_file"]

SUMMARY_DIR.mkdir(exist_ok=True)

# ---------------- resume -----------------------------------------------------------------------------------------
def load_previous(fpath):
    prev = pd.read_csv(fpath, index_col=0)
    if "global_id.1" in prev.columns:                 # global_id as index and column -> read back as global_id.1
        prev = prev.rename(columns={"global_id.1": "global_id"})
    prev.index = prev.index.astype(int)
    prev.index.name = "global_id"
    return prev


already_filled = set()
if RESUME and pathlib.Path(OUT_CSV).exists():
    prev = load_previous(OUT_CSV)
    missing_cols = [c for c in result_cols + param_cols if c not in prev.columns]
    if missing_cols:
        print(f"RESUME: existing file has a different structure ({len(missing_cols)} columns missing, "
              f"e.g. {missing_cols[:3]}) -> ignoring it and starting from scratch.")
    else:
        common = prev.index[prev["mb"].notna()].intersection(df_params.index)
        same = np.isclose(prev.loc[common, param_cols].values.astype(float),
                          df_params.loc[common, param_cols].values.astype(float), atol=1e-8).all(axis=1)
        has_nc = np.array([isinstance(f, str) and (SUMMARY_DIR / f).exists()
                           for f in prev.loc[common, "summary_file"]], dtype=bool)
        if (~same).any():
            print(f"RESUME: {int((~same).sum())} filled rows differ from the design file and will be recomputed.")
        if (same & ~has_nc).any():
            print(f"RESUME: {int((same & ~has_nc).sum())} filled rows lack their summary NetCDF and will be recomputed.")
        keep = common[same & has_nc]
        df_params.update(prev.loc[keep, result_cols])
        already_filled = set(int(g) for g in keep)
        print(f"RESUME: took over {len(already_filled)} filled rows from {OUT_CSV}")
else:
    print("No previous output used -> computing everything.")


# ---------------- matching ---------------------------------------------------------------------------------------
def parse_param_vals_from_filename(fname):
    """parameter values in the order of param_cols (the file name rounds to 4 decimals)."""
    tail = fname.split("_RRR-")[1]
    tail = tail.split("_num")[0]
    vals = [float(v) for v in tail.split("_")]
    return np.array([vals[FNAME_INDEX[c]] for c in param_cols])


def parse_run_id_from_filename(fname):
    """COSIPY counts from 1, so _num{k} belongs to run_id k-1."""
    return int(fname.split("_num")[-1]) - 1


def find_row(df, cols, values, atol=5e-5):
    """MATCH = "params": the design row whose parameters equal the file name's (exact at 4 decimals, else atol)."""
    rounded_vals = np.round(values, 4)
    mask_exact = (df[cols].round(4).values == rounded_vals).all(axis=1)
    idxs = np.where(mask_exact)[0]
    if len(idxs) == 1:
        return df.index[idxs[0]], "exact"
    mask_tol = np.all(np.isclose(df[cols].values, values, atol=atol), axis=1)
    idxs = np.where(mask_tol)[0]
    if len(idxs) == 1:
        return df.index[idxs[0]], "tolerance"
    if len(idxs) > 1:
        print("  ambiguous parameter match, rows:", list(df.index[idxs]))
    return None, None


def match_file(name):
    """-> (global_id or None, status)."""
    param_vals = parse_param_vals_from_filename(name)
    if MATCH == "params":
        gid, method = find_row(df_params, param_cols, param_vals)
        return (gid, "nomatch") if gid is None else (gid, method)
    gid = parse_run_id_from_filename(name) - ID_OFFSET
    if gid not in df_params.index:
        return None, "nomatch"
    design_vals = np.round(df_params.loc[gid, param_cols].values.astype(float), 4)
    if not np.allclose(design_vals, param_vals, atol=1.5e-4):
        diff = np.abs(design_vals - param_vals)
        print(f"PARAM MISMATCH {name}: largest difference in {param_cols[int(np.argmax(diff))]} "
              f"({diff.max():.5f}) -- wrong design file or ID_OFFSET?")
        df_params.at[gid, "param_mismatch"] = float(diff.max())
        return None, "mismatch"
    return gid, "run_id"


# ---------------- glacier-wide MB and albedo -----------------------------------------------------------------------
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
        total_mass_change = (ncfile["MB"] * ref_weights).sum(dim=["lat","lon"])
        weighted_mb = total_mass_change / ref_area_total
        dfmb = weighted_mb.to_dataframe(name="weighted_mb")
        annual_mb = dfmb.resample("1YE").sum()
        return np.nanmean(annual_mb["weighted_mb"].values)
    else:
        ref_weights = ncfile["N_Points"].sum(dim=["lat","lon"])
        alb_total = (ncfile["ALBEDO"] * ncfile["N_Points"]).sum(dim=["lat","lon"])
        weighted_alb = alb_total / ref_weights
        dates, clean_day_vals, secs = prereq_res(weighted_alb)
        resampled_alb_vals = resample_by_hand(weighted_alb.data, secs, clean_day_vals).copy()
        resampled_alb = xr.DataArray(resampled_alb_vals, coords={"time": dates}, dims=["time"], name="ALBEDO_weighted")
        return resampled_alb.sortby("time").sel(time=albobs.time)


# ---------------- all other outputs: one NetCDF per run --------------------------------------------------------
DAILY_SUM = ["MB", "surfM", "subM", "SNOWFALL"]
DAILY_MEAN = ["ALBEDO", "TS", "SNOWHEIGHT", "LWC_00_10", "LWC_10_50", "LWC_DEEP"]
PASS_VARS = ["LWC_00_10", "LWC_10_50", "LWC_DEEP", "SNOWHEIGHT"]
PASS_START = "2017-01-01"                    # Sentinel-1 years used (S1_wetsnow_acquisitions.js)
PASS_LST = {"desc": 6.0, "asc": 18.0}        # h, local solar time of the S1 passes
# stored as unsigned 16-bit integers (scale, max): saves ~4x before compression; NaN -> fill value
PACK = {"LWC_00_10": 1e-5, "LWC_10_50": 1e-5, "LWC_DEEP": 1e-5,          # volume fraction, max 0.65
        "melt_6h": 1e-5, "melt_24h": 1e-5,                                # m w.e., max 0.65
        "SNOWHEIGHT": 1e-2}                                               # m, max 655


def flat(ds, v):
    """(time, lat, lon) -> (time, point)."""
    return ds[v].values.reshape(ds.sizes["time"], -1)


def point_weights(ds):
    w = ds["N_Points"]
    w = flat(ds, "N_Points") if "time" in w.dims else np.broadcast_to(w.values.ravel(), (ds.sizes["time"], w.size))
    return np.nan_to_num(w.astype(float))


def daily_glacier_wide(ds, w):
    """N_Points-weighted glacier mean per step, then daily sum (fluxes) or mean (states)."""
    days, start = np.unique(ds.time.values.astype("datetime64[D]"), return_index=True)
    wsum = w.sum(axis=1)
    wsum = np.where(wsum > 0, wsum, np.nan)
    out = {}
    for v in DAILY_SUM + DAILY_MEAN:
        if v not in ds:
            continue
        g = np.nansum(flat(ds, v) * w, axis=1) / wsum
        n_ = np.add.reduceat(np.isfinite(g).astype(float), start)
        s = np.add.reduceat(np.nan_to_num(g), start)
        s = s / n_ if v in DAILY_MEAN else s
        out[v] = (("day",), np.where(n_ > 0, s, np.nan).astype("float32"), {"units": ds[v].attrs.get("units", "")})
    return pd.DatetimeIndex(days.astype("datetime64[ns]")), out


def at_passes(ds, w):
    """per point at the model step containing each S1 pass (from PASS_START)."""
    t = pd.DatetimeIndex(ds.time.values)
    lon = float(np.mean(ds.lon.values))
    assert 60 < lon < 105, f"lon = {lon}: not an HMA longitude, cannot place the Sentinel-1 pass hours"
    dates = pd.date_range(max(t[0].normalize(), pd.Timestamp(PASS_START)), t[-1].normalize(), freq="D")
    step = np.full((len(dates), len(PASS_LST)), -1)
    for k, lst in enumerate(PASS_LST.values()):
        tp = dates + pd.to_timedelta(lst - lon / 15.0, unit="h")         # pass time, UTC
        i = t.searchsorted(tp)                                            # first step at / after the pass
        ok = (i < len(t)) & (tp >= t[0] - pd.Timedelta("1h"))
        step[ok, k] = i[ok]
    valid = step >= 0
    s = np.where(valid, step, 0)
    melt = np.nan_to_num(flat(ds, "surfM")) + np.nan_to_num(flat(ds, "subM"))
    c = np.vstack([np.zeros((1, melt.shape[1])), np.cumsum(melt, axis=0)])
    fields = {v: flat(ds, v)[s] for v in PASS_VARS}
    fields["melt_6h"] = c[s + 1] - c[np.maximum(s - 5, 0)]
    fields["melt_24h"] = c[s + 1] - c[np.maximum(s - 23, 0)]
    fields["N_Points"] = w[s]
    dims = ("date", "pass", "point")
    out = {v: (dims, np.where(valid[..., None], np.clip(x, 0, None), np.nan).astype("float32"))
           for v, x in fields.items()}
    step_time = np.where(valid, t.values[s], np.datetime64("NaT"))
    return dates, out, step_time, lon


def write_summary(ds, row, source):
    w = point_weights(ds)
    days, daily = daily_glacier_wide(ds, w)
    dates, passes, step_time, lon = at_passes(ds, w)
    S = xr.Dataset(
        {**{f"{v}_daily": (d, x, a) for v, (d, x, a) in daily.items()}, **passes},
        coords={"day": days, "date": dates, "pass": list(PASS_LST), "point": np.arange(w.shape[1]),
                "HGT": ("point", ds["HGT"].values.ravel()), "step_time": (("date", "pass"), step_time)},
        attrs={"source_file": source, "pass_local_solar_time_h": str(PASS_LST), "lon": lon,
               "daily": "N_Points-weighted glacier means per step; daily sum: " + ", ".join(DAILY_SUM)
                        + "; daily mean: " + ", ".join(DAILY_MEAN),
               "melt_6h_24h": "surfM + subM summed over the 6 / 24 model steps ending at step_time (m w.e.)"},
    ).expand_dims(run=[row])
    enc = {v: {"zlib": True, "complevel": 4} for v in S.data_vars}
    for v, sc in PACK.items():
        enc[v].update(dtype="uint16", scale_factor=sc, _FillValue=np.uint16(65535))
    out = SUMMARY_DIR / f"{source}.nc"
    S.to_netcdf(out, encoding=enc)
    return out.name, step_time


# ---------------- main loop ----------------------------------------------------------------------------------------
n_done = n_nomatch = n_mismatch = n_skipped = n_error = 0
first = True
for fp in sorted(pathlib.Path(path).glob('*.nc')):
    name = str(fp.stem)
    csv_name = "tsla_" + name.lower() + ".csv"
    try:
        gid, status = match_file(name)
    except Exception:
        print("EXCEPTION FOUND (file name not parsable):", fp.name)
        continue
    if gid is None:
        if status == "nomatch":
            print("NO MATCH:", fp.name)
            n_nomatch += 1
        else:
            n_mismatch += 1
        continue
    if gid in already_filled:
        n_skipped += 1
        continue
    if status == "tolerance":
        print("TOLERANCE:", fp.name)
        df_params.at[gid, "filename_tolerance_match"] = name

    # a file that is still being written (or whose TSLA csv is missing) is left empty and retried next call
    try:
        with xr.open_dataset(fp) as ds_raw:
            ds = ds_raw.sel(time=slice("1990-01-01", None))
            mb = compute_glacier_mean(ds.sel(time=slice("2000-01-01T00:00", "2019-12-31T23:00")),
                                      "MB", "2000-01-01T01:00", None)
            alb_vals = np.asarray(compute_glacier_mean(ds, "ALBEDO", None, alb_obs_data).data[:n_alb])
            snowlinesim = pd.read_csv(path + csv_name, parse_dates=True, index_col="time")
            tsla_vals = snowlinesim.loc[snowlinesim.index.isin(tsla_obs.index), "Med_TSL"].values[:n_tsla]
            summary_file, step_time = write_summary(ds, gid, name)
    except Exception as e:
        print(f"ERROR reading {fp.name} ({type(e).__name__}: {e}) -- left empty, retried next call")
        n_error += 1
        continue

    df_params.loc[gid, "mb"] = mb
    df_params.loc[gid, "filename"] = name
    df_params.loc[gid, "summary_file"] = summary_file
    df_params.loc[gid, tsla_names] = tsla_vals
    df_params.loc[gid, alb_names] = alb_vals
    n_done += 1
    if first:
        print("model steps used for the S1 passes (UTC, hour-ending):",
              {p: sorted(set(pd.DatetimeIndex(step_time[:, k]).dropna().strftime("%H:%M"))) for k, p in enumerate(PASS_LST)})
        first = False
    if n_done % 100 == 0:
        print(f"Processed {n_done} new files ({n_skipped} skipped so far).")
    if SAVE_EVERY and n_done % SAVE_EVERY == 0:
        df_params.to_csv(OUT_CSV)

print(f"\nnewly filled: {n_done}   taken over/skipped: {n_skipped}   no match: {n_nomatch}"
      f"   parameter mismatch: {n_mismatch}   read errors: {n_error}"
      f"   still missing: {int(df_params['mb'].isna().sum())} of {n}")
df_params.to_csv(OUT_CSV)
