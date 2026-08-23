#!/usr/bin/env python
"""
Profile-likelihood analysis for R2 Major Comment 2 (prior edges).

For each full-COSIPY profile run (alb_snow / alb_ice stepped beyond their bounds,
all other params at posterior mean), compute the SAME three standardized
log-likelihood scores used in the MCMC, then plot vs parameter value with the
posterior bound marked. 
"""
import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

# pkl with the LHS-derived standardization stats:
path = "/path/to/COSIPY/LHS/"

# directory holding the profile-run daily CSVs + the manifest:
profile_out = "/path/to/COSIPY/Reviews/BoundsTest/"
profile_manifest = os.path.join(profile_out, "profile_manifest.csv")

# observation files
path_snowlines   = "/path/to/snowlines/HEF-snowlines-1999-2010_manual_filtered.csv"
path_to_geodetic = "/path/to/geod_mb/Hugonnet_21_MB/dh_11_rgi60_pergla_rates.csv"
alb_obs_path     = "/path/to/albedo/HEF_processed_HRZ-30CC-filter_albedos.nc"

# posterior values of the systematic error terms
sigma_tsl_summer = 0.143
sigma_alb_summer = 0.060

# posterior bounds to mark on the plot
#        snow = pm.TruncatedNormal("albsnow", mu=0.903, sigma=0.1, lower=0.887, upper=0.928)
#        ice = pm.TruncatedNormal("albice", mu=0.17523, sigma=0.1, lower=0.1182, upper=0.2302)
bound = {"alb_snow": (0.887, 0.9328), "alb_ice": (0.1182, 0.2302)}

#
season_lookup = {12:"winter",1:"winter",2:"winter",3:"winter",4:"winter",5:"winter",
                 6:"summer",7:"summer",8:"summer",9:"summer",10:"winter",11:"winter"}

with open(path+"loglike_stats.pkl","rb") as f:
    all_stats = pickle.load(f)
median_mb,  std_mb  = all_stats["mass"]["mean"],   all_stats["mass"]["std"]
median_alb, std_alb = all_stats["albedo"]["mean"], all_stats["albedo"]["std"]
median_tsl, std_tsl = all_stats["snow"]["mean"],   all_stats["snow"]["std"]

standardize_tsla = False  # use raw TSLA (matches MCMC, no standardization)

# snowlines
tsl = pd.read_csv(path_snowlines)
time_start_dt = pd.to_datetime("2000-01-01")
time_end_dt   = pd.to_datetime("2009-12-31")
tsla_true_obs = tsl.copy()
tsla_true_obs['LS_DATE'] = pd.to_datetime(tsla_true_obs['LS_DATE'])
print("Start date:", time_start_dt); print("End date:", time_end_dt)
tsla_true_obs = tsla_true_obs.loc[(tsla_true_obs['LS_DATE'] > time_start_dt) & (tsla_true_obs['LS_DATE'] <= time_end_dt)]
tsla_true_obs.set_index('LS_DATE', inplace=True)
tsla_true_obs['SC_stdev'] = tsla_true_obs['SC_stdev'] / (tsla_true_obs['glacier_DEM_max'] - tsla_true_obs['glacier_DEM_min'])
thres_unc = 20 / (tsla_true_obs['glacier_DEM_max'].iloc[0] - tsla_true_obs['glacier_DEM_min'].iloc[0])
print(thres_unc)
sc_norm = np.where(tsla_true_obs['SC_stdev'] < thres_unc, thres_unc, tsla_true_obs['SC_stdev'])
tsla_true_obs['SC_stdev'] = sc_norm
tsla_true_obs['season']  = tsla_true_obs.index.month.map(season_lookup)
tsl_obs_vec  = tsla_true_obs['TSL_normalized'].values
tsl_obs_unc  = tsla_true_obs['SC_stdev'].values
tsl_is_summer= (tsla_true_obs['season'].values=="summer")

# albedo
alb_obs_data = xr.open_dataset(alb_obs_path).sortby("time")
months = alb_obs_data["time"].dt.month
alb_season = np.array([season_lookup[m.item()] for m in months])
alb_obs_vec = alb_obs_data['median_albedo'].values
alb_obs_unc = alb_obs_data['sigma_albedo'].values
alb_is_summer = (alb_season=="summer")

# geodetic
geod_ref = pd.read_csv(path_to_geodetic)
geod_ref = geod_ref[(geod_ref['rgiid']=="RGI60-11.00897") & (geod_ref['period']=="2000-01-01_2010-01-01")]
mb_obs   = float(geod_ref['dmdtda'].iloc[0])
mb_obs_unc = float(geod_ref['err_dmdtda'].iloc[0])

# ======================= log-likelihood (mirrors MCMC) =======================
def norm_logpdf(x, mu, sigma):
    return stats.norm.logpdf(x, loc=mu, scale=sigma)

def scores_from_model(mod_mb, mod_tsl, mod_alb):
    """mod_* are the COSIPY-derived equivalents of mu_mb, mu_tsl, mu_alb."""
    # MB (scalar), standardized
    ll_mb = norm_logpdf(mb_obs, mod_mb, mb_obs_unc)
    ll_mb_std = (ll_mb - median_mb)/std_mb

    # TSL (vector) mean, sigma = sqrt(obs^2 + sys^2), summer-only sys
    sig_tsl = np.sqrt(tsl_obs_unc**2 + np.where(tsl_is_summer, sigma_tsl_summer, 0.0)**2)
    if np.isnan(mod_tsl).any():
        ll_tsl_raw = np.nan
    else:
        ll_tsl_raw = norm_logpdf(tsl_obs_vec, mod_tsl, sig_tsl).mean()
    ll_tsl_std = (ll_tsl_raw - median_tsl) / std_tsl
    ll_tsl = ll_tsl_std if standardize_tsla else ll_tsl_raw

    # ALB (vector) mean, standardized
    sig_alb = np.sqrt(alb_obs_unc**2 + np.where(alb_is_summer, sigma_alb_summer, 0.0)**2)
    ll_alb = norm_logpdf(alb_obs_vec, mod_alb, sig_alb).mean()
    ll_alb_std = (ll_alb - median_alb)/std_alb

    # raw components
    ll_mb_raw  = ll_mb
    ll_alb_raw = ll_alb
    total_std  = np.nansum([ll_mb_std, ll_tsl_raw, ll_alb_std])   # MB & alb standardized, TSLA raw (as in MCMC)
    total_raw  = np.nansum([ll_mb_raw, ll_tsl_raw, ll_alb_raw])
    return dict(
        loglike_mb=ll_mb_std, loglike_tsl=ll_tsl, loglike_alb=ll_alb_std, total=total_std,
        loglike_mb_raw=ll_mb_raw, loglike_tsl_raw=ll_tsl_raw, loglike_alb_raw=ll_alb_raw, total_raw=total_raw,
    )

csv_mb_col   = "mean_mb"        
csv_alb_col  = "mean_albedo"    
csv_tsla_col  = "Med_TSL"        
mb_is_increment = True          # True: daily increments -> annual = sum; False: cumulative -> last-first
mb_year_start = 2000         
mb_year_end   = 2009         # inclusive, 2000-01-01_2010-01-01 geodetic period

def _load_daily(csv_path):
    d = pd.read_csv(csv_path, index_col=0)
    d.index = pd.to_datetime(d.index)
    return d.sort_index()

def extract_model_outputs(csv_path):
    """
    Reduce a COSIPY daily-CSV run to (mod_mb, mod_tsl, mod_alb), matched to the
    observation times.

      mod_mb  : scalar decadal mean specific mass change rate (m w.e. a-1)
      mod_alb : glacier-mean albedo at the albedo observation times (alb_obs_data.time).
      mod_tsl : normalized SLA at the snowline observation dates (tsla_obs.index).
    """
    d = _load_daily(csv_path)

    # --- MB climatic dmdtda ---
    cal = d.loc[str(mb_year_start):str(mb_year_end)]
    grp = cal[csv_mb_col].groupby(cal.index.year)
    if mb_is_increment:
        annual = grp.sum()                          # daily increments -> annual MB
    else:
        annual = grp.last() - grp.first()           # cumulative state -> annual change
    annual = annual[grp.count() >= 365]             # full calendar years only
    mod_mb = float(annual.mean())

    # --- albedo at exact obs times ---
    alb_times = pd.to_datetime(alb_obs_data["time"].values)
    mod_alb = d[csv_alb_col].reindex(alb_times).values

    # --- SLA at exact obs dates  ---
    sla_dates = pd.to_datetime(tsla_true_obs.index)
    mod_tsl = d[csv_tsla_col].reindex(sla_dates).values

    return mod_mb, mod_tsl, mod_alb

#
man = pd.read_csv(profile_manifest)
rows = []
for _, r in man.iterrows():
    nc = os.path.join(profile_out, f"gsa_result_sim_{r['count']}.csv")
    if not os.path.exists(nc):
        print(f"missing: {nc}"); continue
    try:
        mod_mb, mod_tsl, mod_alb = extract_model_outputs(nc)
        s = scores_from_model(mod_mb, mod_tsl, mod_alb)
        rows.append(dict(profile=r['profile'], varied=r['varied'], value=r['value'],
                         mod_mb=mod_mb,                       # modelled MB (m w.e./yr)
                         mod_alb_mean=float(np.nanmean(mod_alb)),
                         **s))
    except NotImplementedError as e:
        print("Extraction not wired yet:", e); break

res = pd.DataFrame(rows)
res.to_csv(os.path.join(profile_out, "profile_scores.csv"), index=False)
print(res)


#
# paper colours: MB #D81B1B, SLA #A5781B, albedo #1E80E5
c_mb, c_sla, c_alb = "#D81B1B", "#A5781B", "#1E80E5"
if len(res):
    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharey=True)
    for ax, var in zip(axes, ["alb_snow", "alb_ice"]):
        sub = res[res.varied == var].sort_values("value")
        if not len(sub): continue
        # MB & albedo standardized, SLA raw -- the same mix as the MCMC objective
        ax.plot(sub.value, sub.total,       'o-',  color='black', lw=2,
                label=r'$\sum \mathcal{L}$', zorder=5)
        ax.plot(sub.value, sub.loglike_mb,  's--', color=c_mb,  alpha=.8,
                label=r'$\mathcal{L}(B_{geod}|\theta)$')
        ax.plot(sub.value, sub.loglike_tsl, 'd--', color=c_sla, alpha=.8,
                label=r'$\mathcal{L}(SLA|\theta)$')
        ax.plot(sub.value, sub.loglike_alb, '^--', color=c_alb, alpha=.8,
                label=r'$\mathcal{L}(\bar{\alpha}|\theta)$')
        lo, hi = bound[var]
        ax.axvspan(lo, hi, color='green', alpha=0.10, label='posterior bounds')
        ax.axvline(lo, color='green', ls=':', lw=1.2); ax.axvline(hi, color='green', ls=':', lw=1.2)
        ax.set_xlabel({'alb_snow': r'$\alpha_{fs}$ (-)', 'alb_ice': r'$\alpha_{ice}$ (-)'}[var])
        ax.grid(True, alpha=.3, zorder=-1)
    axes[0].set_ylabel(r'log-likelihood $\mathcal{L}$')
    axes[1].legend(fontsize=13, loc='center right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig("/path/to/figures/FigS25_Profile-Likelihood-Boundstest.png", bbox_inches='tight')

# 
if len(res):
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharey=True)
    for ax, var in zip(axes2, ["alb_snow", "alb_ice"]):
        sub = res[res.varied == var].sort_values("value")
        if not len(sub): continue
        ax.plot(sub.value, sub.mod_mb, 'o-', color="#D81B1B", lw=2, label="modelled $B$")
        ax.axhline(mb_obs, color='black', ls='--', label=r'$B_{geod}$ (obs.)')
        ax.axhspan(mb_obs-mb_obs_unc, mb_obs+mb_obs_unc, color='gray', alpha=0.25,
                   label=r'$\pm\sigma_{obs}$')
        lo, hi = bound[var]
        ax.axvspan(lo, hi, color='green', alpha=0.10, label='posterior bounds')
        ax.axvline(lo, color='green', ls=':', lw=1.2); ax.axvline(hi, color='green', ls=':', lw=1.2)
        ax.set_xlabel({'alb_snow': r'$\alpha_{fs}$ (-)', 'alb_ice': r'$\alpha_{ice}$ (-)'}[var])
        ax.grid(True, alpha=.3, zorder=-1)
    axes2[0].set_ylabel(r'$B$ (m w.e. a$^{-1}$)')
    axes2[1].legend(fontsize=12, loc='best', framealpha=0.9)
    fig2.tight_layout()
    fig2.savefig("/path/to/figures/profile_mb_physical.png", bbox_inches='tight')