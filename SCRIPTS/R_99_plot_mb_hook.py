"""
Reviewer-response diagnostic figure (R L469: low-elevation MB flattening).

Single 4-panel figure, shared elevation axis, 50 m WGMS-aligned bands:
  (a) Mean annual MB profile: 1D 20 m run, 30 m run, WGMS
  (b) Hypothetical additional terrain LWin (Prinz et al. 2016, eps=0.98)
  (c) Mean sw_dir_cor (ALL hours; config switch for melt season only)
  (d) N_Points (COSMO source cells per band, 1D 20 m)

Built on new_compare_hrz.py (xarray-aligned SVF/T2/G terrain-LW computation).
Uses the ORIGINAL (pre-radfix) runs and LUTs, i.e. the manuscript configuration.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

plt.rcParams.update({"font.size": 22, "axes.titlesize": 22,
                     "axes.labelsize": 22, "xtick.labelsize": 18,
                     "ytick.labelsize": 18, "legend.fontsize": 18})

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
bpath = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/new_radtest_hef_review/"
OUT_1D  = bpath + "HEF_COSMO_1D20m_1999_2010_HORAYZON_IntpPRES_PosteriorMean_19990101-20091231_RRR-0.705_0.887_0.229_0.643_14.206_1.009_0.24_3.155_4.0_0.0026_1.0_1.0_0.0_1.5553_num.nc"
OUT_30M = bpath + "HEF_30m_HORAYZON-old_hydroyears2001to2003_PosteriorMean.nc"

staticpath  = "/data/scratch/richteny/thesis/cosipy_test_space/data/static/HEF/"
# sw_dir_cor from the ORIGINAL LUTs (manuscript configuration)
HRZ_1D  = staticpath + "HEF_HORAYZON-LUT_1D20m.nc"
HRZ_30M = staticpath + "HEF_HORAYZON-LUT_30m.nc"
# SVF from the -new LUTs (static terrain property; identical between LUT
# generations, but only computed/stored in the new files)
SVF_1D_FILE  = staticpath + "HEF_HORAYZON-LUT-new_1D20m.nc"
SVF_30M_FILE = staticpath + "HEF_HORAYZON-LUT-new_30m.nc"

bpath_cli  = "/data/scratch/richteny/thesis/cosipy_test_space/data/input/HEF/"
FORCING_1D = bpath_cli + "HEF_COSMO_1D20m_HORAYZON_1999_2010_IntpPRES.nc"

WGMS_PATH   = "/data/scratch/richteny/thesis/DOI-WGMS-FoG-2022-09/data/mass_balance.csv"
MB_COL_WGMS = "ANNUAL_BALANCE"

HYDRO_YEARS_1D  = [2001, 2002, 2003]        # common period with 30 m run
HYDRO_YEARS_30M = [2001, 2002, 2003]        # limited 30 m window

# sw_dir_cor averaging window: melt season (all hours incl. night zeros)
SWCOR_MONTHS = (5, 6, 7, 8, 9)

USE_CACHE  = True
CACHE_FILE = "lowelev_profiles_cache_mjjas_v6.npz"

MB_VAR   = "MB"
BAND_W50 = 50
ELEV_LIM = (2400, 3750)

# Prinz et al. (2016) terrain-LW constants
SIGMA        = 5.67e-8
EPS_ROCK     = 0.98
K_SOLAR_HEAT = 0.01            # K W-1 m2 (Sicart et al. 2006)
L_F, RHO_W   = 334000.0, 1000.0
MELT_DAYS, DT = 153, 3600.0
ALPHA_ICE_CAL = 0.229   # calibrated (constant) ice albedo
ALPHA_ICE_MIN = 0.12    # observed minimum at the tongue
melt_hours_per_year = MELT_DAYS * 12
W_TO_MJ   = melt_hours_per_year * DT / 1e6
MJ_TO_MWE = 1e6 / (L_F * RHO_W)

C_1D, C_30M = "#d62728", "#1f77b4"
MRK, MRK_S, LW = "o", 4, 1.6

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def to_hydro_year(t):
    return xr.where(t.dt.month >= 10, t.dt.year + 1, t.dt.year)

def select_hydro_years(ds, years):
    return ds.sel(time=slice(pd.Timestamp(f"{min(years)-1}-10-01"),
                             pd.Timestamp(f"{max(years)}-09-30")))

def mean_by_band(elev, vals, band_w, b0, b1, weights=None):
    elev, vals = np.asarray(elev, float), np.asarray(vals, float)
    valid = ~np.isnan(elev) & ~np.isnan(vals)
    if weights is not None:
        weights = np.asarray(weights, float)
        valid &= ~np.isnan(weights)
        weights = weights[valid]
    elev, vals = elev[valid], vals[valid]
    bands = np.arange(b0, b1 + 1e-9, band_w, dtype=float)
    out = []
    for b in bands:
        m = (elev >= b) & (elev < b + band_w)
        if not m.any():
            out.append(np.nan)
        elif weights is not None:
            w = weights[m]; ws = np.nansum(w)
            out.append(float(np.nansum(vals[m]*w)/ws) if ws > 0 else np.nan)
        else:
            out.append(float(np.nanmean(vals[m])))
    return bands + band_w/2, np.array(out)

def quantiles_by_band(elev, vals, band_w, b0, b1, qs=(0.25, 0.5, 0.75)):
    """Per-band quantiles of vals (e.g. within-band spread of 30 m cells)."""
    elev, vals = np.asarray(elev, float), np.asarray(vals, float)
    valid = ~np.isnan(elev) & ~np.isnan(vals)
    elev, vals = elev[valid], vals[valid]
    bands = np.arange(b0, b1 + 1e-9, band_w, dtype=float)
    out = np.full((len(bands), len(qs)), np.nan)
    for k, b in enumerate(bands):
        m = (elev >= b) & (elev < b + band_w)
        if m.any():
            out[k] = np.quantile(vals[m], qs)
    return bands + band_w/2, out

def annual_mb_1d(ds, years):
    ds = select_hydro_years(ds, years)
    ds = ds.assign_coords(hydro_year=("time", to_hydro_year(ds.time).values))
    mb = ds[MB_VAR].groupby("hydro_year").sum().mean("hydro_year").values.ravel()
    hgt = ds["HGT"].values.ravel()
    npts = ds["N_Points"].values.ravel() if "N_Points" in ds else np.ones_like(hgt)
    i = np.argsort(hgt)
    return hgt[i], mb[i], npts[i]

def annual_mb_2d(ds, years):
    ds = select_hydro_years(ds, years)
    ds = ds.assign_coords(hydro_year=("time", to_hydro_year(ds.time).values))
    mb = ds[MB_VAR].groupby("hydro_year").sum().mean("hydro_year").values.ravel()
    hgt = ds["HGT"];  hgt = hgt.isel(time=0) if "time" in hgt.dims else hgt
    msk = ds["MASK"]; msk = msk.isel(time=0) if "time" in msk.dims else msk
    hgt_f, msk_f = hgt.values.ravel(), msk.values.ravel().astype(bool)
    valid = msk_f & ~np.isnan(hgt_f)
    return hgt_f[valid], mb[valid]

# ═══════════════════════════════════════════════════════════════════════════
# LOAD + PROFILES
# ═══════════════════════════════════════════════════════════════════════════
import os
if USE_CACHE and os.path.exists(CACHE_FILE):
    print(f"Loading cached band profiles from {CACHE_FILE} "
          f"(delete file or set USE_CACHE=False to recompute)")
    _c = np.load(CACHE_FILE)
    (mb_wgms_idx, mb_wgms_val, h_mb1d, mb1d_50, h_mb30, mb30_50, h_lw, tlw_50MJ,
     h_asw, asw_50, h_sc1d, sc1d_50, h_sc30, sc30_50, h_np, np_50) = (
        _c[k] for k in ("wgms_i","wgms_v","h_mb1d","mb1d","h_mb30","mb30",
                        "h_lw","tlw","h_asw","asw","h_sc1d","sc1d",
                        "h_sc30","sc30","h_np","npv"))
    h_scq, sc30_q = _c["h_scq"], _c["sc30_q"]
    h_gc, gc_50 = _c["h_gc"], _c["gc"]
    h_g1d, g1d_50 = _c["h_g1d"], _c["g1d"]
    ELA = float(_c["ela"]); B0, B1 = int(_c["b0"]), int(_c["b1"])
    import pandas as _pd
    mb_wgms = _pd.Series(mb_wgms_val, index=mb_wgms_idx)
    y_grid = np.arange(B0, B1 + 1, BAND_W50)
    SKIP_COMPUTE = True
else:
    SKIP_COMPUTE = False

if not SKIP_COMPUTE:
 print("Loading model output...")
 out_1d  = xr.open_dataset(OUT_1D).squeeze("lon", drop=True)
 out_30m = xr.open_dataset(OUT_30M)

 print("Loading WGMS...")
 wgms = pd.read_csv(WGMS_PATH)
 wgms = wgms.loc[(wgms["NAME"] == "HINTEREIS F.") &
                (wgms["YEAR"] >= min(HYDRO_YEARS_1D)) &
                (wgms["YEAR"] <= max(HYDRO_YEARS_1D)) &
                (wgms["LOWER_BOUND"] != 9999)]
 mb_wgms = wgms.groupby("LOWER_BOUND")[MB_COL_WGMS].mean()
 mb_wgms.index += 25
 mb_wgms /= 1000.0
 B0 = int(wgms["LOWER_BOUND"].min())
 B1 = int(wgms["LOWER_BOUND"].max()) + 50
 def reagg(h, v, weights=None):
    return mean_by_band(h, v, BAND_W50, B0, B1, weights=weights)

 print("MB profiles...")
 hgt_1d, mb_1d, npts_1d = annual_mb_1d(out_1d, HYDRO_YEARS_1D)
 if out_30m["HGT"].dims == ("lat",):
    hgt_30m, mb_30m, _ = annual_mb_1d(out_30m, HYDRO_YEARS_30M)
 else:
    hgt_30m, mb_30m = annual_mb_2d(out_30m, HYDRO_YEARS_30M)

 h_mb1d, mb1d_50 = reagg(hgt_1d, mb_1d, weights=npts_1d)
 h_mb30, mb30_50 = reagg(hgt_30m, mb_30m)

 # ELA: topmost zero crossing of the (sorted) 1D mean-annual MB profile
 sign_change = np.where((mb_1d[:-1] < 0) & (mb_1d[1:] >= 0))[0]
 if len(sign_change):
    i0 = sign_change[-1]
    dm = mb_1d[i0+1] - mb_1d[i0]
    ELA = hgt_1d[i0] + (-mb_1d[i0]/dm)*(hgt_1d[i0+1]-hgt_1d[i0]) if dm != 0 else hgt_1d[i0]
 else:
    ELA = np.nan
 print(f"ELA (1D, {HYDRO_YEARS_1D[0]}-{HYDRO_YEARS_1D[-1]}): {ELA:.0f} m")

 # Hypothetical darker-ice absorption: alpha decreases linearly from
 # ALPHA_ICE_CAL at the ELA to ALPHA_ICE_MIN at the lowest glacier band;
 # extra absorbed SW = cumulative melt-season SWin x delta_alpha.
 print("Cumulative melt-season SWin per pixel...")
 ds_g = xr.open_dataset(FORCING_1D)
 if "lon" in ds_g.dims:
    ds_g = ds_g.squeeze("lon", drop=True)
 g_sel   = ds_g["G"].sel(time=ds_g.time.dt.month.isin([5, 6, 7, 8, 9]))
 G_cumul = (g_sel.groupby(g_sel.time.dt.year).sum().mean("year")
                .values.ravel() * DT / 1e6)          # MJ m-2 a-1
 hgt_g = ds_g["HGT"].values.ravel()
 msk_g = ds_g["MASK"].values.ravel().astype(bool) if "MASK" in ds_g else np.ones_like(hgt_g, bool)
 ds_g.close()
 hgt_G, G_prof = hgt_g[msk_g], G_cumul[msk_g]

 h_gc, gc_50 = reagg(hgt_G, G_prof)      # cumulative May-Sep SWin per band (MJ)
 f_alpha     = np.clip((ELA - hgt_G) / (ELA - hgt_G.min()), 0.0, 1.0)
 delta_alpha = f_alpha * (ALPHA_ICE_CAL - ALPHA_ICE_MIN)   # 0 at/above ELA
 extra_sw_MJ = G_prof * delta_alpha
 h_asw, asw_50 = reagg(hgt_G, extra_sw_MJ)

 # ── sw_dir_cor (ALL hours by default) ──────────────────────────────────────
 print(f"sw_dir_cor (months={'all' if SWCOR_MONTHS is None else SWCOR_MONTHS})...")
 def swcor_profile(hrz_path, hgt_arr, msk_arr):
    try:    # chunked (dask) -> time-mean streams instead of loading all hours
        hrz = xr.open_dataset(hrz_path, chunks={"time": 24 * 31})
    except (ValueError, ImportError):
        hrz = xr.open_dataset(hrz_path)
    da = hrz["sw_dir_cor"]
    if SWCOR_MONTHS is not None:
        da = da.sel(time=da["time"].dt.month.isin(SWCOR_MONTHS))
    if "lon" in da.dims and da.sizes["lon"] == 1:
        da = da.squeeze("lon", drop=True)
    sc = np.asarray(da.mean("time")).ravel()   # triggers streamed compute if dask
    hrz.close()
    return hgt_arr[msk_arr], sc[msk_arr]

 ds_f = xr.open_dataset(FORCING_1D)
 if "lon" in ds_f.dims:
    ds_f = ds_f.squeeze("lon", drop=True)
 hgt_f  = ds_f["HGT"].values.ravel()
 msk_f  = ds_f["MASK"].values.ravel().astype(bool) if "MASK" in ds_f else np.ones_like(hgt_f, bool)
 npts_f = ds_f["N_Points"].values.ravel() if "N_Points" in ds_f else np.ones_like(hgt_f)
 lat_coord = ds_f["lat"]
 melt_t = ds_f.time.dt.month.isin([5, 6, 7, 8, 9])
 T_mean_xr  = ds_f["T2"].sel(time=melt_t).mean("time")
 G_mean_xr  = ds_f["G"].sel(time=melt_t).mean("time")
 ds_f.close()

 # Sky LWin is not stored in the forcing file; take it from the COSIPY OUTPUT
 # (out_1d, already open), where LWin is by definition exactly the sky flux
 # the model applied over the full hemisphere.
 lw_var = "LWin" if "LWin" in out_1d else ("LWIN" if "LWIN" in out_1d else None)
 assert lw_var is not None, f"No LWin in output file; variables: {list(out_1d.data_vars)}"
 melt_o = out_1d.time.dt.month.isin([5, 6, 7, 8, 9])
 LW_mean_xr = (out_1d[lw_var].sel(time=melt_o).mean("time")
                            .sel(lat=lat_coord, method="nearest"))
 print(f"  sky LWin (melt-season mean, from model output): "
      f"{float(LW_mean_xr.min()):.0f}-{float(LW_mean_xr.max()):.0f} W/m2")

 h_sc1d_raw, sc1d_raw = swcor_profile(HRZ_1D, hgt_f, msk_f)

 hrz30 = xr.open_dataset(HRZ_30M)
 elev_key = "elevation" if "elevation" in hrz30 else "HGT"
 hgt30_sc = hrz30[elev_key].values.ravel()
 msk30_sc = (hrz30["MASK"].values.ravel().astype(bool)
            if "MASK" in hrz30 else ~np.isnan(hgt30_sc))
 hrz30.close()
 h_sc30_raw, sc30_raw = swcor_profile(HRZ_30M, hgt30_sc, msk30_sc)

 h_sc1d, sc1d_50 = reagg(h_sc1d_raw, sc1d_raw)
 h_sc30, sc30_50 = reagg(h_sc30_raw, sc30_raw)

 # Cumulative May-Sep SWin actually RECEIVED, from the model outputs (MJ m-2 a-1),
 # melt seasons of the common years. No reconstruction, no division.
 def cum_G(ds, years):
     g = ds["G"].sel(time=ds.time.dt.month.isin([5,6,7,8,9]))
     g = g.sel(time=g.time.dt.year.isin(years))
     return np.asarray(g.groupby(g.time.dt.year).sum().mean("year")).ravel()*DT/1e6
 G_YEARS = [2001, 2002, 2003]
 g1d_px  = cum_G(out_1d, G_YEARS)
 h_g1d, g1d_50 = reagg(hgt_f[msk_f], g1d_px[msk_f], weights=npts_f[msk_f])
 # within-band spatial quantiles of the 30 m per-cell temporal means:
 # the variability the elevation-band aggregation removes
 h_scq, sc30_q = quantiles_by_band(h_sc30_raw, sc30_raw, BAND_W50, B0, B1,
                                   qs=(0.05, 0.25, 0.5, 0.75, 0.95))

 # ── terrain LWin (Prinz 2016), lat-aligned as in new_compare_hrz.py ───────
 print("Terrain LWin (Prinz et al. 2016)...")
 # SVF from the -new LUT (static terrain field), aligned onto the forcing lat grid
 svf1d_ds = xr.open_dataset(SVF_1D_FILE)
 if "lon" in svf1d_ds.dims:
    svf1d_ds = svf1d_ds.squeeze("lon", drop=True)
 svf_xr = svf1d_ds["svf"].sel(lat=lat_coord, method="nearest")
 print(f"  SVF (1D 20m): min={float(svf_xr.min()):.3f} mean={float(svf_xr.mean()):.3f}")
 svf1d_ds.close()


 T_R_xr = T_mean_xr + K_SOLAR_HEAT * G_mean_xr
 # Prinz et al. (2016): LWI_corr = sf*LWI + (1-sf)*eps*sigma*T_R^4.
 # COSIPY applies the sky LWI over the FULL hemisphere, so the energy missing
 # from the model is the difference:
 #   dLW = LWI_corr - LWI = (1 - sf) * (eps*sigma*T_R^4 - LWI)
 tlw_xr = (1.0 - svf_xr) * (EPS_ROCK * SIGMA * T_R_xr**4 - LW_mean_xr)   # W m-2
 print(f"  terrain emission ~{float((EPS_ROCK*SIGMA*T_R_xr**4).mean()):.0f} W/m2, "
      f"sky LWin ~{float(LW_mean_xr.mean()):.0f} W/m2, "
      f"net dLW mean ~{float(tlw_xr.mean()):.1f} W/m2")
 tlw_px = tlw_xr.values[msk_f]
 h_lw, tlw_50MJ = reagg(hgt_f[msk_f], tlw_px * W_TO_MJ)

 # ── N_Points ───────────────────────────────────────────────────────────────
 h_np, np_50 = reagg(hgt_f[msk_f], npts_f[msk_f])

 np.savez(CACHE_FILE, wgms_i=mb_wgms.index.values, wgms_v=mb_wgms.values,
          h_mb1d=h_mb1d, mb1d=mb1d_50, h_mb30=h_mb30, mb30=mb30_50,
          h_lw=h_lw, tlw=tlw_50MJ, h_asw=h_asw, asw=asw_50,
          h_sc1d=h_sc1d, sc1d=sc1d_50, h_sc30=h_sc30, sc30=sc30_50,
          h_scq=h_scq, sc30_q=sc30_q, h_gc=h_gc, gc=gc_50,
          h_g1d=h_g1d, g1d=g1d_50,
          h_np=h_np, npv=np_50, ela=ELA, b0=B0, b1=B1)
 print(f"Saved band profiles to {CACHE_FILE}")

# ═══════════════════════════════════════════════════════════════════════════
# FILL-IN VALUES FOR THE RESPONSE TEXT
# ═══════════════════════════════════════════════════════════════════════════
tongue = h_lw < 2550          # band centres below 2550 m a.s.l.
def _mx(v, m):  # nan-safe max over the tongue bands
    v = np.asarray(v, float)[m]
    return np.nan if np.all(np.isnan(v)) else np.nanmax(v)

tlw_max_MJ   = _mx(tlw_50MJ, tongue)
asw_max_MJ   = _mx(asw_50,  h_asw < 2550)
tlw_max_mwe  = tlw_max_MJ * MJ_TO_MWE
asw_max_mwe  = asw_max_MJ * MJ_TO_MWE
comb_max_mwe = _mx(tlw_50MJ + np.interp(h_lw, h_asw, asw_50), tongue) * MJ_TO_MWE
i_sc  = int(np.nanargmin(sc1d_50))
np_lo = np_50[h_np < 2550]

print("\n" + "="*68)
print("FILL-IN VALUES FOR THE RESPONSE TEXT")
print("="*68)
print(f"ELA (1D, {HYDRO_YEARS_1D[0]}-{HYDRO_YEARS_1D[-1]}):            {ELA:.0f} m a.s.l.")
print(f"Terrain LW at tongue (max, <2550m):    {tlw_max_MJ:.0f} MJ m-2 a-1  =  {tlw_max_mwe:.2f} m w.e. a-1")
print(f"Darker-ice SW at tongue (max, <2550m): {asw_max_MJ:.0f} MJ m-2 a-1  =  {asw_max_mwe:.2f} m w.e. a-1")
print(f"Combined melt potential at tongue:     {comb_max_mwe:.2f} m w.e. a-1")
print(f"Ratio albedo/terrain-LW terms:         {asw_max_mwe/tlw_max_mwe:.2f}" if tlw_max_mwe else "")
print(f"Min mean SW correction factor:         {sc1d_50[i_sc]:.2f} at {h_sc1d[i_sc]:.0f} m (1D)")
g_tongue = np.nanmin(np.asarray(g1d_50,float)[h_g1d < 2550])
g_ela    = float(np.interp(ELA, h_g1d, g1d_50))
print(f"N points per band below 2550 m:        {', '.join(f'{v:.0f}' for v in np_lo)}")
print("="*68 + "\n")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════
y_grid = np.arange(B0, B1 + 1, BAND_W50)

def style_ax(ax, xlabel, letter):
    ax.set_yticks(np.arange(2400, 3801, 200))
    ax.set_yticks(y_grid, minor=True)
    ax.yaxis.grid(True, which="minor", color="lightgrey", ls=":", lw=0.6)
    ax.yaxis.grid(True, which="major", color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.xaxis.grid(True, alpha=0.3)
    if np.isfinite(ELA):
        ax.axhline(ELA, color="dimgrey", ls=":", lw=1.2)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.text(0.04, 0.975, f"{letter})", transform=ax.transAxes,
            fontsize=22, va="top", ha="left")
    ax.set_ylim(*ELEV_LIM)

fig, axes = plt.subplots(1, 5, figsize=(24, 10),
                         sharey=True, layout="constrained",
                         gridspec_kw={"width_ratios": [1.5, 1, 1, 1, 1]})
fig.get_layout_engine().set(w_pad=0.15)  # inches of padding between panels

yr1d  = f"{HYDRO_YEARS_1D[0]}\u2013{HYDRO_YEARS_1D[-1]}"
yr30m = f"{HYDRO_YEARS_30M[0]}\u2013{HYDRO_YEARS_30M[-1]}"

# (a) MB profiles
ax = axes[0]
ax.plot(mb_wgms.values, mb_wgms.index.values, color="black", lw=2.0,
        marker=MRK, ms=MRK_S+1, label=f"WGMS ({yr1d})", zorder=6)
ax.plot(mb1d_50, h_mb1d, color=C_1D, lw=LW, marker=MRK, ms=MRK_S,
        label=f"1D 20 m ({yr1d})")
ax.plot(mb30_50, h_mb30, color=C_30M, lw=LW, marker=MRK, ms=MRK_S,
        label=f"30 m ({yr30m})")
ax.axvline(0, color="grey", lw=0.8)
ax.set_ylabel("Elevation (m a.s.l.)")
ax.legend(loc="upper right")
style_ax(ax, "Mean annual MB (m w.e. a$^{-1}$)", "a")

# (b) additional terrain LWin
ax = axes[1]
ax.plot(tlw_50MJ, h_lw, color="darkorange", lw=LW, marker=MRK, ms=MRK_S)
ax.axvline(0, color="grey", lw=0.8)
style_ax(ax, "Terrain $Q_{LWin}$ (MJ m$^{-2}$ a$^{-1}$)", "b")
ax2 = ax.twiny()
xl = ax.get_xlim()
ax2.set_xlim(xl[0]*MJ_TO_MWE, xl[1]*MJ_TO_MWE)
ax2.set_xlabel("m w.e. a$^{-1}$", fontsize=16)
ax2.tick_params(labelsize=16)

# (c) hypothetical darker-ice absorption below the ELA
ax = axes[2]
ax.plot(asw_50, h_asw, color="seagreen", lw=LW, marker=MRK, ms=MRK_S)
ax.axvline(0, color="grey", lw=0.8)
style_ax(ax, "Additional $Q_{SWnet}$ (MJ m$^{-2}$ a$^{-1}$)", "c")
ax2c = ax.twiny()
xlc = ax.get_xlim()
ax2c.set_xlim(xlc[0]*MJ_TO_MWE, xlc[1]*MJ_TO_MWE)
ax2c.set_xlabel("m w.e. a$^{-1}$", fontsize=16)
ax2c.tick_params(labelsize=16)

# (d) cumulative May-Sep SWin received by the calibrated (1D) setup
ax = axes[3]
ax.plot(g1d_50, h_g1d, color=C_1D, lw=LW, marker=MRK, ms=MRK_S)
style_ax(ax, "Mean cumulative May\u2013Sep\n$Q_{SWin}$ (MJ m$^{-2}$)", "d")

# (e) N_Points
ax = axes[4]
ax.barh(h_np, np_50, height=BAND_W50*0.75, color="steelblue", alpha=0.5,
        edgecolor="none")
ax.plot(np_50, h_np, color="steelblue", lw=LW, marker=MRK, ms=MRK_S, zorder=3)
style_ax(ax, "No. of points (\u2013)", "e")

plt.savefig("fig_lowelev_diagnostic.png", dpi=300, bbox_inches="tight")
print("Saved fig_lowelev_diagnostic.png")
plt.show()
