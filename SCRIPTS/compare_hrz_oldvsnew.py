"""
Compare old vs new HORAYZON radiation correction for HEF.

Produces two figures:
  1. MB gradient — old/new 1D-20m (2002-2009) + old/new 30m (exemplary,
     whichever hydro-years are available) vs WGMS
  2. SWin at the two AWS stations — old/new forcing vs AWS observations
     (one hydrological year, whichever is available)
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde, circmean
from sklearn.metrics import mean_squared_error, r2_score
import pathlib, sys

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                     "axes.labelsize": 11, "xtick.labelsize": 10,
                     "ytick.labelsize": 10, "legend.fontsize": 10})


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG  — adapt all paths
# ═══════════════════════════════════════════════════════════════════════════

# 1D 20m outputs (old = original sw_dir_cor × total; new = direct+diffuse fix)
bpath = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/new_radtest_hef_review/"
OUT_1D_OLD = bpath+"HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES_PosteriorMean_19990101-20091231_RRR-0.705_0.887_0.229_0.643_14.206_1.009_0.24_3.155_4.0_0.0026_1.0_1.0_0.0_1.5553_num.nc"
OUT_1D_NEW = bpath+"HEF_COSMO_1D20m_1999_2010_HORAYZON_radfix_IntpPRES_PosteriorMean_19990101-20091231_RRR-0.705_0.887_0.229_0.643_14.206_1.009_0.24_3.155_4.0_0.0026_1.0_1.0_0.0_1.5553_num.nc"

# 30m outputs (exemplary — limited time window)
OUT_30M_OLD = bpath + "HEF_30m_HORAYZON-old_hydroyears2001to2003_PosteriorMean.nc"
OUT_30M_NEW = bpath + "HEF_30m_HORAYZON-new_hydroyears2001to2003_PosteriorMean.nc"

## Load static data for sw_dir_cor's
staticpath = "/data/scratch/richteny/thesis/cosipy_test_space/data/static/HEF/"
OUT_HRZ_1D_OLD = xr.open_dataset(staticpath+"HEF_HORAYZON-LUT_1D20m.nc")
OUT_HRZ_1D_NEW = xr.open_dataset(staticpath+"HEF_HORAYZON-LUT-new_1D20m.nc")
OUT_HRZ_2D_OLD = xr.open_dataset(staticpath+"HEF_HORAYZON-LUT_30m.nc")
OUT_HRZ_2D_NEW = xr.open_dataset(staticpath+"HEF_HORAYZON-LUT-new_30m.nc") 

## just add vals to 

# COSIPY forcing files (for SWin comparison)
bpath_cli = "/data/scratch/richteny/thesis/cosipy_test_space/data/input/HEF/"
FORCING_1D_OLD = bpath_cli + "HEF_COSMO_1D20m_HORAYZON_1999_2010_IntpPRES.nc"
FORCING_1D_NEW = bpath_cli + "HEF_COSMO_1D20m_HORAYZON_radfix_hydro_1999_2010.nc"

# AWS observations  (hourly CSV with column "SWI" for incoming SW)
AWS_PATH  = "/data/scratch/richteny/thesis/Hintereisferner/Climate/AWS_Obleitner/"
AWS_LOWER_FILE = "Fix_HEFlower_01102003_24102004.csv"
AWS_UPPER_FILE = "Fix_HEFupper_01102003_24102004.csv"
AWS_YEAR_START = "2003-10-01"
AWS_YEAR_END   = "2004-09-30"

# WGMS
WGMS_PATH    = "/data/scratch/richteny/thesis/DOI-WGMS-FoG-2022-09/data/mass_balance.csv"
MB_COL_WGMS  = "ANNUAL_BALANCE"

# AWS station elevations [m a.s.l.] — used to select closest forcing band
ELEV_LOWER = 2640.0
ELEV_UPPER = 3048.0

# Hydro-year ranges
HYDRO_YEARS_1D  = list(range(2002, 2010))   # 2002–2009 for full 1D period
HYDRO_YEARS_30M = [2001, 2002, 2003]         # 30m run: Oct 2000–Sep 2003
# Common period for fair comparison across all runs and WGMS
HYDRO_YEARS_COMMON = [2001, 2002, 2003]

MB_VAR   = "MB"
BAND_W   = 20
# Fixed elevation range aligned to WGMS bins

#ELA      = 3100
SW_MIN   = 0.0    # W m⁻² — daytime threshold for metrics


# ═══════════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════════

out_1d_old = xr.open_dataset(OUT_1D_OLD).squeeze("lon", drop=True)
out_1d_new = xr.open_dataset(OUT_1D_NEW).squeeze("lon", drop=True)
out_30m_old = xr.open_dataset(OUT_30M_OLD)
out_30m_new = xr.open_dataset(OUT_30M_NEW)

# WGMS
wgms = pd.read_csv(WGMS_PATH)
wgms = wgms.loc[
    (wgms["NAME"] == "HINTEREIS F.") &
    (wgms["YEAR"] >= 2002) & (wgms["YEAR"] <= 2009) &
    (wgms["LOWER_BOUND"] != 9999)
]
wgms.drop(["POLITICAL_UNIT", "NAME", "REMARKS"], axis=1, inplace=True)
mb_wgms = wgms.groupby("LOWER_BOUND")[MB_COL_WGMS].mean()
mb_wgms.index = mb_wgms.index + 25   # midpoint of 50 m bin
mb_wgms = mb_wgms / 1000             # kg m⁻² → m w.e.

# WGMS restricted to common comparison period (2001–2003)
wgms_c = wgms.loc[
    (wgms["YEAR"] >= min(HYDRO_YEARS_COMMON)) &
    (wgms["YEAR"] <= max(HYDRO_YEARS_COMMON)) &
    (wgms["LOWER_BOUND"] != 9999)]
mb_wgms_c = wgms_c.groupby("LOWER_BOUND")[MB_COL_WGMS].mean()
mb_wgms_c.index = mb_wgms_c.index + 25
mb_wgms_c = mb_wgms_c / 1000

# AWS
aws_lower = pd.read_csv(AWS_PATH + AWS_LOWER_FILE,
                        parse_dates=True, index_col="time")
aws_lower = aws_lower.loc[AWS_YEAR_START:AWS_YEAR_END]

aws_upper = pd.read_csv(AWS_PATH + AWS_UPPER_FILE,
                        parse_dates=True, index_col="time")
aws_upper = aws_upper.loc[AWS_YEAR_START:AWS_YEAR_END]


# Elevation range from WGMS band boundaries — all model profiles use this grid
# LOWER_BOUND is the lower edge of each 50 m bin; upper edge = max + 50 m
WGMS_BAND_START = int(wgms["LOWER_BOUND"].min())
WGMS_BAND_END   = int(wgms["LOWER_BOUND"].max()) + 50

print(f"WGMS bands: {len(mb_wgms)}, "
      f"range {mb_wgms.index.min():.0f}–{mb_wgms.index.max():.0f} m")
print(f"Lower AWS: {aws_lower['SWI'].notna().sum()} valid SWI hours")
print(f"Upper AWS: {aws_upper['SWI'].notna().sum()} valid SWI hours")


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def to_hydro_year(time_coord):
    return xr.where(time_coord.dt.month >= 10,
                    time_coord.dt.year + 1,
                    time_coord.dt.year)


def select_hydro_years(ds, years):
    start = pd.Timestamp(f"{min(years) - 1}-10-01")
    end   = pd.Timestamp(f"{max(years)}-09-30")
    return ds.sel(time=slice(start, end))


def mean_by_band(elev_1d, values_1d, band_w=BAND_W,
                 band_start=None, band_end=None, weights=None):
    """
    (Optionally weighted) mean per elevation band.
    band_start/band_end: force fixed boundaries (default: auto from data).
    weights            : per-element weights (e.g. N_Points).
    Returns (band_midpoints, means) — NaN where no data in a band.
    """
    elev_1d   = np.asarray(elev_1d,   dtype=float)
    values_1d = np.asarray(values_1d, dtype=float)
    valid = ~np.isnan(elev_1d) & ~np.isnan(values_1d)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        valid  &= ~np.isnan(weights)
        weights = weights[valid]
    elev_1d, values_1d = elev_1d[valid], values_1d[valid]
    if len(elev_1d) == 0:
        return np.array([]), np.array([])
    b0 = (band_start if band_start is not None
          else np.nanmin(elev_1d) // band_w * band_w)
    b1 = (band_end if band_end is not None
          else np.nanmax(elev_1d) // band_w * band_w + band_w)
    bands = np.arange(b0, b1 + 1e-9, band_w, dtype=float)
    result = []
    for b in bands:
        m = (elev_1d >= b) & (elev_1d < b + band_w)
        if not m.any():
            result.append(np.nan)
        elif weights is not None:
            w = weights[m]; v = values_1d[m]
            ws = np.nansum(w)
            result.append(float(np.nansum(v * w) / ws) if ws > 0 else np.nan)
        else:
            result.append(float(np.nanmean(values_1d[m])))
    return bands + band_w / 2, np.array(result)


def annual_mb_1d(ds, years, mb_var=MB_VAR):
    """Mean annual MB per elevation band for a 1D (lat-only) output."""
    ds   = select_hydro_years(ds, years)
    hy   = to_hydro_year(ds.time)
    ds   = ds.assign_coords(hydro_year=("time", hy.values))
    mb_annual = ds[mb_var].groupby("hydro_year").sum()
    mb_mean   = mb_annual.mean("hydro_year").values.ravel()
    hgt    = ds["HGT"].values.ravel()
    n_pts  = ds["N_Points"].values.ravel() if "N_Points" in ds else np.ones_like(hgt)
    sort_i = np.argsort(hgt)
    return hgt[sort_i], mb_mean[sort_i], n_pts[sort_i]

def annual_mb_2d(ds, years, mb_var=MB_VAR, band_w=BAND_W):
    """Mean annual MB per elevation band for a 2D (lat × lon) output."""
    ds   = select_hydro_years(ds, years)
    hy   = to_hydro_year(ds.time)
    ds   = ds.assign_coords(hydro_year=("time", hy.values))
    
    # Calculate mean annual MB (this removes the time dimension)
    mb_annual = ds[mb_var].groupby("hydro_year").sum()
    mb_mean   = mb_annual.mean("hydro_year").values.ravel()
    
    # Extract static variables, dropping the 'time' dimension if it exists
    hgt = ds["HGT"]
    if "time" in hgt.dims:
        hgt = hgt.isel(time=0)
        
    mask = ds["MASK"]
    if "time" in mask.dims:
        mask = mask.isel(time=0)
        
    mask_flat = mask.values.ravel().astype(bool)
    hgt_flat  = hgt.values.ravel()
    
    # Now all arrays share the exact same spatial length (34,001)
    valid     = mask_flat & ~np.isnan(hgt_flat)
    
    return mean_by_band(hgt_flat[valid], mb_mean[valid], band_w)

def extract_G_at_elevation(forcing_path, target_elev, date_start, date_end):
    """
    Return hourly G [W m⁻²] from the forcing file at the elevation band
    closest to target_elev. Works for 1D (lat × lon=1) and 2D files.
    """
    ds  = xr.open_dataset(forcing_path)
    ds  = ds.sel(time=slice(date_start, date_end))
    hgt = ds["HGT"].values
    msk = ds["MASK"].values.astype(bool) if "MASK" in ds else np.ones_like(hgt, bool)
    hgt_flat = hgt[msk].ravel()
    closest  = hgt_flat[np.argmin(np.abs(hgt_flat - target_elev))]
    print(f"  {target_elev:.0f} m → closest band {closest:.0f} m")
    where = np.where(hgt == closest)
    lat_i = where[0][0]
    lon_i = where[1][0] if hgt.ndim > 1 else 0
    g = (ds["G"].isel(lat=lat_i, lon=lon_i)
         if ds["G"].dims[-1] == "lon"
         else ds["G"].isel(lat=lat_i)).to_series().rename("G")
    ds.close()
    return g


def calc_metrics(obs, mod):
    o, m = np.asarray(obs, float), np.asarray(mod, float)
    return {"R2": r2_score(o, m),
            "MBE": float(np.mean(m - o)),
            "RMSE": float(np.sqrt(mean_squared_error(o, m))),
            "N": len(o)}


def calc_cdf(vals):
    x = np.sort(vals)
    return x, np.arange(1, len(x) + 1) / len(x)


# ═══════════════════════════════════════════════════════════════════════════
# MELT SEASON CONSTANTS (for energy → m w.e. conversion)
# ═══════════════════════════════════════════════════════════════════════════
L_F       = 334000.0    # latent heat of fusion [J kg⁻¹]
RHO_W     = 1000.0      # water density [kg m⁻³]
SIGMA     = 5.67e-8     # Stefan-Boltzmann [W m⁻² K⁻⁴]
EPS_ROCK    = 0.98      # rock emissivity (Prinz et al. 2016)
# Terrain temperature follows Prinz et al. (2016):
#   T_R = T_air + 0.01 [K W⁻¹ m²] × G
# Applied per elevation band as melt-season mean.
# The coefficient 0.01 K W⁻¹ m² is from Sicart et al. (2006, 2011).
K_SOLAR_HEAT = 0.01     # terrain solar heating coefficient [K W⁻¹ m²]
ALPHA_ICE_CAL = 0.229   # calibrated ice albedo (constant in current run)
ALPHA_ICE_MIN = 0.12    # observed minimum at the tongue (field measurements)
MELT_DAYS = 153         # approximate melt-season length May–Sep [days]
DT        = 3600.0      # hourly timestep [s]

# Unit conversions used throughout for energy→melt comparisons
# All energy panels plot MJ m⁻² a⁻¹ (bottom axis) and m w.e. a⁻¹ (top axis)
melt_hours_per_year = MELT_DAYS * 12           # 12 active h/day
W_TO_MJ   = melt_hours_per_year * DT / 1e6    # W m⁻² (mean) → MJ m⁻² a⁻¹
MJ_TO_MWE = 1e6 / (L_F * RHO_W)              # MJ m⁻² → m w.e.
W_TO_MWE  = W_TO_MJ * MJ_TO_MWE              # convenience shorthand

def add_mwe_top_axis(ax, mj_to_mwe=None):
    """Add secondary x-axis in m w.e. a⁻¹ on top of a MJ m⁻² a⁻¹ axis."""
    if mj_to_mwe is None:
        mj_to_mwe = MJ_TO_MWE
    ax2 = ax.twiny()
    xl  = ax.get_xlim()
    ax2.set_xlim(xl[0] * mj_to_mwe, xl[1] * mj_to_mwe)
    ax2.set_xlabel("m w.e. a⁻¹", fontsize=9)
    ax2.tick_params(labelsize=9)
    return ax2
DT        = 3600.0      # hourly timestep [s]

n_years   = len(HYDRO_YEARS_1D)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE MB GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════
print("Computing MB gradients...")

hgt_old_1d, mb_old_1d, npts_old = annual_mb_1d(out_1d_old, HYDRO_YEARS_1D)
hgt_new_1d, mb_new_1d, npts_new = annual_mb_1d(out_1d_new, HYDRO_YEARS_1D)

# Common period (2001-2003) for fair comparison with 30m
hgt_old_1d_c, mb_old_1d_c, npts_old_c = annual_mb_1d(out_1d_old, HYDRO_YEARS_COMMON)
hgt_new_1d_c, mb_new_1d_c, npts_new_c = annual_mb_1d(out_1d_new, HYDRO_YEARS_COMMON)

if out_30m_old["HGT"].dims == ("lat",):
    hgt_old_30m, mb_old_30m, _ = annual_mb_1d(out_30m_old, HYDRO_YEARS_30M)
    hgt_new_30m, mb_new_30m, _ = annual_mb_1d(out_30m_new, HYDRO_YEARS_30M)
else:
    hgt_old_30m, mb_old_30m = annual_mb_2d(out_30m_old, HYDRO_YEARS_30M)
    hgt_new_30m, mb_new_30m = annual_mb_2d(out_30m_new, HYDRO_YEARS_30M)


# ═══════════════════════════════════════════════════════════════════════════
# ELA FROM POSTERIOR MEAN MB  (mean over HYDRO_YEARS_1D)
# ═══════════════════════════════════════════════════════════════════════════
# Compute the ELA for each hydro year separately, then report mean ± std.
# ELA = elevation where MB interpolates to 0 for that year.
# ─────────────────────────────────────────────────────────────────────────

def compute_ela_from_ds(ds, years, mb_var=MB_VAR):
    """
    Compute ELA for each hydro year by linear interpolation of MB=0.
    Returns (mean_ela, std_ela, list_of_annual_elas).
    """
    ds_sel = select_hydro_years(ds, years)
    hy     = to_hydro_year(ds_sel.time)
    ds_sel = ds_sel.assign_coords(hydro_year=("time", hy.values))

    hgt   = ds_sel["HGT"].values.ravel()
    sort_i = np.argsort(hgt)
    hgt_s  = hgt[sort_i]

    annual_elas = []
    mb_annual = ds_sel[mb_var].groupby("hydro_year").sum()

    for yr in mb_annual.hydro_year.values:
        mb_yr = mb_annual.sel(hydro_year=yr).values.ravel()[sort_i]
        # Find first index where MB goes from negative to non-negative
        sign_change = np.where((mb_yr[:-1] < 0) & (mb_yr[1:] >= 0))[0]
        if len(sign_change) == 0:
            # All negative (very negative year) or all positive
            ela_yr = hgt_s[-1] if mb_yr[0] >= 0 else np.nan
        else:
            i = sign_change[-1]   # topmost sign change
            # Linear interpolation
            dh = hgt_s[i+1] - hgt_s[i]
            dm = mb_yr[i+1]  - mb_yr[i]
            ela_yr = hgt_s[i] + (-mb_yr[i] / dm) * dh if dm != 0 else hgt_s[i]
        annual_elas.append(ela_yr)

    valid = [e for e in annual_elas if not np.isnan(e)]
    return float(np.mean(valid)), float(np.std(valid)), annual_elas


print("Computing ELA from posterior mean MB field...")
ela_mean, ela_std, ela_annual = compute_ela_from_ds(out_1d_new, HYDRO_YEARS_1D)
print(f"  ELA (new, 1D 20m, {HYDRO_YEARS_1D[0]}-{HYDRO_YEARS_1D[-1]}): "
      f"{ela_mean:.0f} ± {ela_std:.0f} m")

# Replace the hardcoded ELA with the computed value
ELA = ela_mean


# ═══════════════════════════════════════════════════════════════════════════
# SWin GRADIENT FROM FORCING  (melt-season mean, per elevation band)
# ═══════════════════════════════════════════════════════════════════════════
print("Computing SWin profiles from forcing...")

def cumulative_G_pixels(forcing_path, months=(5, 6, 7, 8, 9)):
    """
    Mean annual cumulative melt-season G [MJ m-2 a-1] at pixel level.
    Returns raw (hgt_glacier, G_cumul_glacier) so reagg() can do a single
    clean aggregation to fixed 50 m WGMS bands without edge artefacts.
    """
    ds      = xr.open_dataset(forcing_path)
    if "lon" in ds.dims:
        ds = ds.squeeze("lon", drop=True)
    mask_t  = ds.time.dt.month.isin(months)
    G_sel   = ds["G"].sel(time=mask_t)
    # Sum per calendar year, mean across years → MJ m-2 a-1
    G_cumul = (G_sel.groupby(G_sel.time.dt.year).sum()
                    .mean("year")
                    .values.ravel() * DT / 1e6)
    hgt     = ds["HGT"].values.ravel()
    msk     = (ds["MASK"].values.ravel().astype(bool)
               if "MASK" in ds else np.ones_like(hgt, bool))
    ds.close()
    return hgt[msk], G_cumul[msk]

hgt_G_old, G_old_prof = cumulative_G_pixels(FORCING_1D_OLD)
hgt_G_new, G_new_prof = cumulative_G_pixels(FORCING_1D_NEW)


# ═══════════════════════════════════════════════════════════════════════════
# sw_dir_cor PROFILE + N_Points PER BAND  (1D 20m vs 30m, by resolution)
# ═══════════════════════════════════════════════════════════════════════════
# The old and new sw_dir_cor LUTs are identical in field — the fix was in
# how the correction was applied, not in the LUT itself.  We therefore show
# resolution dependence (1D 20m vs 30m) instead of old vs new.
# Zeros are excluded: sw_dir_cor = 0 when the sun is below the horizon
# (nighttime / polar night); they inflate the median downward.
# ─────────────────────────────────────────────────────────────────────────
print("Computing sw_dir_cor profiles (by resolution)...")

def swcor_pixel(hrz_ds, hgt_arr, msk_arr, n_pts_arr, months=(5, 6, 7, 8, 9)):
    """
    Return pixel-level May-Sep mean sw_dir_cor for glacier cells.
    Does NOT pre-aggregate — returns raw arrays so reagg() can do a single
    clean aggregation to fixed 50 m WGMS bands without edge artefacts.

    hrz_ds   : open HRZ LUT Dataset (sw_dir_cor, time axis = 2020 hourly).
    hgt_arr  : elevation per pixel (1D, length = n_pixels).
    msk_arr  : boolean glacier mask (1D, same length).
    n_pts_arr: N_Points per pixel (1D, same length).
    Returns  : (hgt_glacier, swcor_glacier, npts_glacier) — all 1D, glacier only.
    """
    mask_t   = hrz_ds["time"].dt.month.isin(months)
    swcor_da = hrz_ds["sw_dir_cor"].sel(time=mask_t)
    if "lon" in swcor_da.dims and swcor_da.sizes["lon"] == 1:
        swcor_da = swcor_da.squeeze("lon", drop=True)

    # Temporal mean over ALL May-Sep hours (zeros included — physically real)
    swcor = swcor_da.mean("time").values.ravel()

    return hgt_arr[msk_arr], swcor[msk_arr], n_pts_arr[msk_arr]


# 1D 20m — HGT/MASK/N_Points from forcing file
ds_f1d = xr.open_dataset(FORCING_1D_NEW)
if "lon" in ds_f1d.dims:
    ds_f1d = ds_f1d.squeeze("lon", drop=True)
hgt_1d_sc   = ds_f1d["HGT"].values.ravel()
msk_1d_sc   = ds_f1d["MASK"].values.ravel().astype(bool) if "MASK" in ds_f1d else np.ones_like(hgt_1d_sc, bool)
npts_1d_sc  = ds_f1d["N_Points"].values.ravel() if "N_Points" in ds_f1d else np.ones_like(hgt_1d_sc)
ds_f1d.close()

hgt_sc_1d, swcor_1d, npts_sc_1d = swcor_pixel(OUT_HRZ_1D_NEW, hgt_1d_sc, msk_1d_sc, npts_1d_sc)

# 30m — HGT from HRZ 30m file (variable may be "elevation" or "HGT")
hrz_30m = OUT_HRZ_2D_NEW  # already open
elev_key_30m = "elevation" if "elevation" in hrz_30m else "HGT"
hgt_30m_sc = hrz_30m[elev_key_30m].values.ravel()
if "MASK" in hrz_30m:
    msk_30m_sc = hrz_30m["MASK"].values.ravel().astype(bool)
else:
    msk_30m_sc = ~np.isnan(hgt_30m_sc)
npts_30m_sc = np.ones_like(hgt_30m_sc)   # pixel count uniform at native resolution

hgt_sc_30m, swcor_30m, _ = swcor_pixel(OUT_HRZ_2D_NEW, hgt_30m_sc, msk_30m_sc, npts_30m_sc)

# N_Points for the representativeness panel (from 1D 20m)
npts_sc  = npts_sc_1d
hgt_sc   = hgt_sc_1d


# ═══════════════════════════════════════════════════════════════════════════
# HYPOTHETICAL TERRAIN LWin CORRECTION (m w.e. a⁻¹)
# ═══════════════════════════════════════════════════════════════════════════
# Additional LW from surrounding rock walls, currently missing from the model.
# Uses SVF from the forcing file (if available) or a linear proxy.
# ΔLW = (1 - SVF) × ε_rock × σ × T_rock⁴  [W m⁻²]   (terrain emission)
# The current model uses only sky LW; terrain emission is excluded.
# We assume warm rock temperature only during the melt season.
# ─────────────────────────────────────────────────────────────────────────
print("Computing terrain LW hypothetical correction...")


# HGT and MASK from forcing; SVF from the new HRZ LUT (static, no time dim)
ds_forc = xr.open_dataset(FORCING_1D_NEW)
if "lon" in ds_forc.dims:
    ds_forc = ds_forc.squeeze("lon", drop=True)
hgt_f = ds_forc["HGT"].values.ravel()
msk_f = ds_forc["MASK"].values.ravel().astype(bool) if "MASK" in ds_forc else np.ones_like(hgt_f, bool)
ds_forc.close()

# SVF: static field in the new HRZ file (no time dimension).
# Old HRZ file predates SVF computation — use a linear proxy for the old run.
hrz_new_sq = OUT_HRZ_1D_NEW
if "lon" in hrz_new_sq.dims:
    hrz_new_sq = hrz_new_sq.squeeze("lon", drop=True)

if "svf" in hrz_new_sq:
    svf_vals = hrz_new_sq["svf"].values.ravel()
    print(f"  SVF loaded from HRZ file: min={svf_vals.min():.3f} mean={svf_vals.mean():.3f}")
else:
    print("  WARNING: 'svf' not found in new HRZ file — using linear proxy")
    #svf_vals = 0.97 - (3700 - hgt_f) / (3700 - 2440) * (0.97 - 0.72)

# Melt-season mean T_air and G per elevation band (from forcing)
# T_R = T_air + K_SOLAR_HEAT × G  (Prinz et al. 2016 / Sicart et al. 2006)
print("  Computing melt-season mean T_air and G per band...")
ds_melt = xr.open_dataset(FORCING_1D_NEW)
if "lon" in ds_melt.dims:
    ds_melt = ds_melt.squeeze("lon", drop=True)

melt_mask_t = ds_melt.time.dt.month.isin([5, 6, 7, 8, 9])
T_mean_band = ds_melt["T2"].sel(time=melt_mask_t).mean("time").values.ravel()
G_mean_band = ds_melt["G"].sel(time=melt_mask_t).mean("time").values.ravel()
ds_melt.close()

# Terrain temperature (Prinz et al. 2016 Eq. 6)
T_R_band = T_mean_band + K_SOLAR_HEAT * G_mean_band

# Interpolate T_R from forcing bands to SVF pixels
# (SVF is on the same elevation grid if loaded from the same 1D HRZ file)
terrain_lw_wm2 = (1.0 - svf_vals) * EPS_ROCK * SIGMA * T_R_band**4

# Keep pixel-level — reagg() does the single clean aggregation to 50 m bands
hgt_lw   = hgt_f[msk_f]
tlw_px   = terrain_lw_wm2[msk_f]            # W m⁻² per pixel
tlw_MJ   = tlw_px * W_TO_MJ                 # MJ m⁻² a⁻¹ per pixel
tlw_mwe  = tlw_px * melt_hours_per_year * DT / (L_F * RHO_W)  # m w.e. per pixel

# Scalar summary for logging only
tlw_prof = tlw_px   # alias used in reagg calls below
mean_T_R = float(np.nanmean(T_R_band))
print(f"  Mean terrain temp (Prinz): {mean_T_R - 273.15:.1f}°C")


# ═══════════════════════════════════════════════════════════════════════════
# HYPOTHETICAL ALBEDO DECREASE WITH ELEVATION  (m w.e. a⁻¹)
# ═══════════════════════════════════════════════════════════════════════════
# At the tongue, ice is dirtier / covered with algae → lower albedo.
# Current model: alpha_ice = ALPHA_ICE_CAL everywhere.
# Hypothetical: alpha decreases linearly from ALPHA_ICE_CAL at ELA
#               to ALPHA_ICE_MIN at the lowest glacier band.
# Extra absorbed SW = SWin × (alpha_cal - alpha_elev_dep)
# ─────────────────────────────────────────────────────────────────────────
print("Computing albedo effect...")

elev_min_glacier = hgt_G_new.min()
# alpha(z) = alpha_cal - (ELA - z)/(ELA - z_min) × (alpha_cal - alpha_min)
# Clipped to [alpha_min, alpha_cal] — only below ELA
f_alpha = np.clip((ELA - hgt_G_new) / (ELA - elev_min_glacier), 0.0, 1.0)
alpha_elev = ALPHA_ICE_CAL - f_alpha * (ALPHA_ICE_CAL - ALPHA_ICE_MIN)
delta_alpha = ALPHA_ICE_CAL - alpha_elev   # positive below ELA

# Extra absorbed SW = G_new × delta_alpha  (only during melt season)
extra_sw_MJ  = G_new_prof * delta_alpha   # G_new_prof already MJ m⁻² a⁻¹
extra_sw_mwe = extra_sw_MJ * MJ_TO_MWE   # m w.e. a⁻¹


# ═══════════════════════════════════════════════════════════════════════════
# SWin AT AWS STATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\nExtracting G at AWS stations...")
print("Lower AWS:")
g_old_lower = extract_G_at_elevation(FORCING_1D_OLD, ELEV_LOWER, AWS_YEAR_START, AWS_YEAR_END)
g_new_lower = extract_G_at_elevation(FORCING_1D_NEW, ELEV_LOWER, AWS_YEAR_START, AWS_YEAR_END)
print("Upper AWS:")
g_old_upper = extract_G_at_elevation(FORCING_1D_OLD, ELEV_UPPER, AWS_YEAR_START, AWS_YEAR_END)
g_new_upper = extract_G_at_elevation(FORCING_1D_NEW, ELEV_UPPER, AWS_YEAR_START, AWS_YEAR_END)

def align_daytime(obs, mod_old, mod_new):
    df = pd.DataFrame({"obs": obs, "old": mod_old, "new": mod_new}).dropna()
    day = (df["obs"] >= SW_MIN) | (df["old"] >= SW_MIN) | (df["new"] >= SW_MIN)
    return df.loc[day]

df_lower = align_daytime(aws_lower["SWI"], g_old_lower, g_new_lower)
df_upper = align_daytime(aws_upper["SWI"], g_old_upper, g_new_upper)

def calc_metrics(obs, mod):
    o, m = np.asarray(obs, float), np.asarray(mod, float)
    return {"R2": r2_score(o, m),
            "MBE": float(np.mean(m - o)),
            "RMSE": float(np.sqrt(mean_squared_error(o, m))),
            "N": len(o)}

def calc_cdf(vals):
    x = np.sort(vals)
    return x, np.arange(1, len(x) + 1) / len(x)

metrics = {}
for key, df, col in [
    ("Lower / old", df_lower, "old"), ("Lower / new", df_lower, "new"),
    ("Upper / old", df_upper, "old"), ("Upper / new", df_upper, "new"),
]:
    metrics[key] = calc_metrics(df["obs"], df[col])

print("\n── SWin validation (new vs old) ──────────────────────────────────────")
for station, key_old, key_new in [
    ("Lower AWS", "Lower / old", "Lower / new"),
    ("Upper AWS", "Upper / old", "Upper / new"),
]:
    m_o = metrics[key_old]
    m_n = metrics[key_new]
    print(f"\n  {station}  (N = {m_n['N']:,})")
    print(f"    R²:   {m_n['R2']:.3f}  ({m_o['R2']:.3f} old)")
    print(f"    MBE:  {m_n['MBE']:+.1f}  ({m_o['MBE']:+.1f} old)  W m⁻²")
    print(f"    RMSE: {m_n['RMSE']:.1f}  ({m_o['RMSE']:.1f} old)  W m⁻²")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — MB GRADIENT  (single panel, no suptitle)
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(5.5, 8), dpi=150)

ax.plot(mb_wgms.values, mb_wgms.index.values,
        color="black", lw=2.5, marker="o", ms=6,
        label=f"WGMS ({HYDRO_YEARS_1D[0]}–{HYDRO_YEARS_1D[-1]})", zorder=6)

ax.plot(mb_old_1d, hgt_old_1d, color="#d62728", lw=2, ls="--",
        label=f"1D 20 m — old")
ax.plot(mb_new_1d, hgt_new_1d, color="#d62728", lw=2, ls="-",
        label=f"1D 20 m — new")
npts_norm_old = (npts_old / npts_old.max()) * 60
npts_norm_new = (npts_new / npts_new.max()) * 60
ax.scatter(mb_old_1d, hgt_old_1d, s=npts_norm_old, color="#d62728", alpha=0.2, zorder=3)
ax.scatter(mb_new_1d, hgt_new_1d, s=npts_norm_new, color="#d62728", alpha=0.2, zorder=3)

year_label_30m = f"{HYDRO_YEARS_30M[0]}–{HYDRO_YEARS_30M[-1]}"
ax.plot(mb_old_30m, hgt_old_30m, color="#1f77b4", lw=2, ls="--",
        label=f"30 m — old ({year_label_30m})")
ax.plot(mb_new_30m, hgt_new_30m, color="#1f77b4", lw=2, ls="-",
        label=f"30 m — new ({year_label_30m})")

ax.axvline(0,   color="grey", lw=0.8)
ax.axhline(ELA, color="grey", ls=":", lw=1, label=f"ELA ~{ELA} m")
ax.set_xlabel("Mean annual MB (m w.e. a⁻¹)", fontsize=14)
ax.set_ylabel("Elevation (m a.s.l.)",         fontsize=14)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mb_gradient_old_vs_new.png", dpi=150, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — SWin AT AWS: fixed legend + no suptitle
# ═══════════════════════════════════════════════════════════════════════════
C_OBS = "black"; C_OLD = "#d62728"; C_NEW = "#1f77b4"
XLIMS_SW = [0, 1250]

fig2 = plt.figure(figsize=(16, 7), dpi=150, layout="constrained")
gs2  = gridspec.GridSpec(2, 3, figure=fig2,
                         width_ratios=[2.2, 1, 1],
                         hspace=0.45, wspace=0.30)

for row, (df, station_label, elev, m_old, m_new) in enumerate([
    (df_lower, f"Lower AWS ({ELEV_LOWER:.0f} m a.s.l.)", ELEV_LOWER,
     metrics["Lower / old"], metrics["Lower / new"]),
    (df_upper, f"Upper AWS ({ELEV_UPPER:.0f} m a.s.l.)", ELEV_UPPER,
     metrics["Upper / old"], metrics["Upper / new"]),
]):
    # Time series
    ax_ts = fig2.add_subplot(gs2[row, 0])
    ax_ts.plot(df["obs"], color=C_OBS, lw=1.0, alpha=0.85,
               label="AWS" if row == 0 else "_", zorder=5)
    ax_ts.plot(df["old"], color=C_OLD, lw=1.0, alpha=0.7, ls="--",
               label="Old" if row == 0 else "_", zorder=4)
    ax_ts.plot(df["new"], color=C_NEW, lw=1.0, alpha=0.85, ls="-",
               label="New" if row == 0 else "_", zorder=4)
    ax_ts.set_ylabel(r"$Q_{SWin}$ (W m$^{-2}$)", fontsize=12)
    ax_ts.set_ylim(-20, 1350)
    ax_ts.set_title(station_label)
    if row == 0:
        ax_ts.legend(ncol=3, loc="upper left")
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    ax_ts.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
        ax_ts.xaxis.get_major_locator()))
    ax_ts.xaxis.set_tick_params(rotation=20)
    ax_ts.grid(alpha=0.3)

    # CDF
    ax_cdf = fig2.add_subplot(gs2[row, 1])
    for label, key, col, ls in [("AWS", "obs", C_OBS, "-"),
                                  ("Old", "old", C_OLD, "--"),
                                  ("New", "new", C_NEW, "-")]:
        x, y = calc_cdf(df[key].values)
        ax_cdf.plot(x, y, color=col, lw=1.8, ls=ls,
                    label=label if row == 0 else "_")
    ax_cdf.set_xlabel(r"$Q_{SWin}$ (W m$^{-2}$)", fontsize=11)
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_xlim(XLIMS_SW); ax_cdf.set_ylim(0, 1)
    if row == 0:
        ax_cdf.legend()
    ax_cdf.grid(alpha=0.3); ax_cdf.tick_params()

    # Scatter
    ax_sc = fig2.add_subplot(gs2[row, 2])
    for label, key, col in [("Old", "old", C_OLD), ("New", "new", C_NEW)]:
        ax_sc.scatter(df["obs"].values, df[key].values,
                      s=3, alpha=0.3, color=col, rasterized=True)
    ax_sc.plot(XLIMS_SW, XLIMS_SW, "k--", lw=1.0)
    ax_sc.set_xlim(XLIMS_SW); ax_sc.set_ylim(XLIMS_SW)
    ax_sc.set_xlabel(r"Observed $Q_{SWin}$ (W m$^{-2}$)")
    ax_sc.set_ylabel(r"Modelled $Q_{SWin}$ (W m$^{-2}$)")
    ax_sc.grid(alpha=0.3)
    # Compact stats table — lower-right, below the 1:1 line (sparse area)
    stats_txt = (
        "           R²      MBE   RMSE\n"
        f"Old: {m_old['R2']:.3f}  {m_old['MBE']:+5.1f}  {m_old['RMSE']:5.1f}\n"
        f"New: {m_new['R2']:.3f}  {m_new['MBE']:+5.1f}  {m_new['RMSE']:5.1f}"
    )
    ax_sc.text(0.97, 0.03, stats_txt,
               transform=ax_sc.transAxes, fontsize=8.5,
               va="bottom", ha="right", family="monospace",
               bbox=dict(facecolor="white", alpha=0.88, edgecolor="lightgrey"))

plt.savefig("sw_comparison_old_vs_new.png", dpi=150, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# ABLATION-SEASON T2m PROFILE  (used in Figure 5)
# ═══════════════════════════════════════════════════════════════════════════
print("Computing T2m profile...")
ds_t2m = xr.open_dataset(FORCING_1D_NEW)
if "lon" in ds_t2m.dims:
    ds_t2m = ds_t2m.squeeze("lon", drop=True)
hgt_t2m = ds_t2m["HGT"].values.ravel()
msk_t2m = ds_t2m["MASK"].values.ravel().astype(bool) if "MASK" in ds_t2m           else np.ones_like(hgt_t2m, bool)
melt_t2m = ds_t2m.time.dt.month.isin([5, 6, 7, 8, 9])
T2_mean_px = ds_t2m["T2"].sel(time=melt_t2m).mean("time").values.ravel() - 273.15
# Keep pixel-level — reagg() does the single clean aggregation to 50 m bands
hgt_T  = hgt_t2m[msk_t2m]
T2_prof = T2_mean_px[msk_t2m]
ds_t2m.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — COMBINED DIAGNOSTIC  (MB + SWin + sw_dir_cor + N_Points)
#             All at 50 m bands (WGMS-aligned), common years only
# ═══════════════════════════════════════════════════════════════════════════

BAND_W_50 = 50
ELEV_LIM  = (2400, 3750)
MRK       = "o"
MRK_S     = 4
LW        = 1.6
y_grid_50 = np.arange(WGMS_BAND_START, WGMS_BAND_END + 1, BAND_W_50)
B50_KW    = dict(band_w=BAND_W_50,
                 band_start=WGMS_BAND_START,
                 band_end=WGMS_BAND_END)

def reagg(hgt_arr, val_arr, weights=None):
    """50 m fixed-band aggregation aligned to WGMS."""
    return mean_by_band(np.asarray(hgt_arr, float),
                        np.asarray(val_arr, float),
                        weights=weights, **B50_KW)

# ── Re-aggregate all profiles to fixed 50 m WGMS bands ───────────────────
# MB — N_Points-weighted (common period 2001–2003)
h_mb_o1d, mb_o1d = reagg(hgt_old_1d_c, mb_old_1d_c, weights=npts_old_c)
h_mb_n1d, mb_n1d = reagg(hgt_new_1d_c, mb_new_1d_c, weights=npts_new_c)
h_mb_o30, mb_o30 = reagg(hgt_old_30m,  mb_old_30m)
h_mb_n30, mb_n30 = reagg(hgt_new_30m,  mb_new_30m)

# SWin cumulative
h_sw_o, G_o50 = reagg(hgt_G_old, G_old_prof)
h_sw_n, G_n50 = reagg(hgt_G_new, G_new_prof)

# sw_dir_cor
h_sc_1d, sc_1d50 = reagg(hgt_sc_1d,  swcor_1d)
h_sc_30, sc_3050  = reagg(hgt_sc_30m, swcor_30m)

# N_Points
h_n50, n50 = reagg(hgt_sc_1d, npts_sc_1d)

# ── Helper for shared y-axis styling ─────────────────────────────────────
def style_ax(ax, xlabel, panel_letter):
    ax.set_yticks(np.arange(2400, 3801, 200))
    ax.set_yticks(y_grid_50, minor=True)
    ax.yaxis.grid(True, which="minor", color="lightgrey", ls=":", lw=0.6)
    ax.yaxis.grid(True, which="major", color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.xaxis.grid(True, alpha=0.3)
    ax.axhline(ELA, color="dimgrey", ls=":", lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_title(f"({panel_letter})")
    ax.set_ylim(*ELEV_LIM)

# ── Figure layout ─────────────────────────────────────────────────────────
fig3, axes = plt.subplots(1, 4, figsize=(16, 8), dpi=150,
                           sharey=True, layout="constrained",
                           gridspec_kw={"wspace": 0.06})

yr_c = f"{HYDRO_YEARS_COMMON[0]}–{HYDRO_YEARS_COMMON[-1]}"

# (a) MB gradient
ax = axes[0]
ax.plot(mb_wgms_c.values, mb_wgms_c.index.values,
        color="black", lw=LW, marker=MRK, ms=MRK_S+1,
        label=f"WGMS ({yr_c})", zorder=6)
ax.plot(mb_o1d, h_mb_o1d, color=C_OLD, lw=LW, ls="--",
        marker=MRK, ms=MRK_S, label=f"1D 20 m old ({yr_c})")
ax.plot(mb_n1d, h_mb_n1d, color=C_OLD, lw=LW, ls="-",
        marker=MRK, ms=MRK_S, label=f"1D 20 m new ({yr_c})")
ax.plot(mb_o30, h_mb_o30, color=C_NEW, lw=LW, ls="--",
        marker=MRK, ms=MRK_S, label=f"30 m old ({yr_c})")
ax.plot(mb_n30, h_mb_n30, color=C_NEW, lw=LW, ls="-",
        marker=MRK, ms=MRK_S, label=f"30 m new ({yr_c})")
ax.axvline(0, color="grey", lw=0.8)
ax.set_ylabel("Elevation (m a.s.l.)")
ax.legend(fontsize=8, loc="upper right")
style_ax(ax, "Mean annual MB (m w.e. a$^{-1}$)", "a")

# (b) Cumulative SWin
ax = axes[1]
ax.plot(G_o50, h_sw_o, color=C_OLD, lw=LW, ls="--",
        marker=MRK, ms=MRK_S, label="Old")
ax.plot(G_n50, h_sw_n, color=C_NEW, lw=LW, ls="-",
        marker=MRK, ms=MRK_S, label="New")
ax.legend(fontsize=9)
style_ax(ax, r"Cumul. $Q_{SWin}$ (MJ m$^{-2}$ a$^{-1}$)" + "\n(May\u2013Sep)", "b")

# (c) sw_dir_cor
ax = axes[2]
ax.plot(sc_1d50, h_sc_1d, color=C_OLD, lw=LW,
        marker=MRK, ms=MRK_S, label="1D 20 m")
ax.plot(sc_3050, h_sc_30, color=C_NEW, lw=LW,
        marker=MRK, ms=MRK_S, label="30 m")
# Note: gaps in a line = no glacier cells in that 50 m band
ax.legend(fontsize=9)
style_ax(ax, "sw_dir_cor (–)\n(May–Sep mean, all hours)", "c")

# (d) N_Points
ax = axes[3]
ax.barh(h_n50, n50, height=BAND_W_50 * 0.75,
        color="steelblue", alpha=0.5, edgecolor="none")
ax.plot(n50, h_n50, color="steelblue", lw=LW,
        marker=MRK, ms=MRK_S, zorder=3)
style_ax(ax, "N_Points\n(source cells per 50 m band)", "d")

plt.savefig("diagnostic_profiles_reviewer.png", dpi=150, bbox_inches="tight")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — TERRAIN LWin HYPOTHETICAL + T2m  (common years)
# ═══════════════════════════════════════════════════════════════════════════
# Re-aggregate terrain LW and T2m to fixed 50 m bands
hgt_lw_50, tlw_50 = reagg(hgt_lw, tlw_MJ)
hgt_T_50,  T2_50  = reagg(hgt_T,   T2_prof)

fig5, axes5 = plt.subplots(1, 3, figsize=(12, 8), dpi=150,
                            sharey=True, layout="constrained",
                            gridspec_kw={"wspace": 0.06})

for ax5 in axes5:
    ax5.set_yticks(np.arange(2400, 3801, 200))
    ax5.set_yticks(y_grid_50, minor=True)
    ax5.yaxis.grid(True, which="minor", color="lightgrey", ls=":", lw=0.6)
    ax5.yaxis.grid(True, which="major", color="grey", ls="--", lw=0.6, alpha=0.5)
    ax5.xaxis.grid(True, alpha=0.3)
    ax5.axhline(ELA, color="dimgrey", ls=":", lw=1.2)
    ax5.set_ylim(*ELEV_LIM)

# (a) MB gradient
ax = axes5[0]
ax.plot(mb_wgms_c.values, mb_wgms_c.index.values,
        color="black", lw=LW, marker=MRK, ms=MRK_S+1, label="WGMS", zorder=6)
ax.plot(mb_o1d, h_mb_o1d, color=C_OLD, lw=LW, ls="--",
        marker=MRK, ms=MRK_S, label="1D 20 m old")
ax.plot(mb_n1d, h_mb_n1d, color=C_OLD, lw=LW, ls="-",
        marker=MRK, ms=MRK_S, label="1D 20 m new")
ax.plot(mb_o30, h_mb_o30, color=C_NEW, lw=LW, ls="--",
        marker=MRK, ms=MRK_S, label="30 m old")
ax.plot(mb_n30, h_mb_n30, color=C_NEW, lw=LW, ls="-",
        marker=MRK, ms=MRK_S, label="30 m new")
ax.axvline(0, color="grey", lw=0.8)
ax.set_ylabel("Elevation (m a.s.l.)")
ax.set_xlabel("Mean annual MB (m w.e. a$^{-1}$)")
ax.set_title(f"(a) MB gradient ({yr_c})")
ax.legend(fontsize=8, loc="upper right")

# (b) Terrain LWin hypothetical
ax = axes5[1]
ax.plot(tlw_50, hgt_lw_50, color="darkorange", lw=LW,
        marker=MRK, ms=MRK_S)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel(r"Additional LWin (MJ m$^{-2}$ a$^{-1}$)")
ax.set_title("(b) Terrain LWin\nhypothetical\n(Prinz et al. 2016, ε=0.98)")
add_mwe_top_axis(axes5[1])

# (c) T2m lapse rate
ax = axes5[2]
ax.plot(T2_50, hgt_T_50, color="firebrick", lw=LW,
        marker=MRK, ms=MRK_S)
ax.axvline(0, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Mean T₂ₘ (°C)\n(May–Sep)")
ax.set_title("(c) Ablation-season\nair temperature")

plt.savefig("mb_terrain_lw_t2m.png", dpi=150, bbox_inches="tight")
plt.show()
