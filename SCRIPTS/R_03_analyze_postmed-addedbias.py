import os, json, pickle
import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

## SCRIPT created with the help of claude! 

base = "/path/to/COSIPY/"
path = base + "LHS/"                       # loglike_stats.pkl
GSA_OUT = base + "Reviews/"             # gsa_result_sim_*.csv + matrix/json
POSTERIOR_NC = base + "stage2_final_demczsyserr_posterior_combined.nc"

path_snowlines   = base + "../snowlines/HEF-snowlines-1999-2010_manual_filtered.csv"
path_to_geodetic = base + "../geod_data/Hugonnet_21_MB/dh_11_rgi60_pergla_rates.csv"  # EDIT if needed
alb_obs_path     = base + "../albedo/HEF_processed_HRZ-30CC-filter_albedos.nc"

SIGMA_TSL_SUMMER, SIGMA_ALB_SUMMER = 0.143, 0.060
CSV_MB_COL, CSV_ALB_COL, CSV_SLA_COL = "mean_mb", "mean_albedo", "Med_TSL"
MB_YEAR_START, MB_YEAR_END = 2000, 2009
N_SIGMA_MB = 3.0

# ======================= load obs + stats (as MCMC) =======================
season_lookup = {12:"winter",1:"winter",2:"winter",3:"winter",4:"winter",5:"winter",
                 6:"summer",7:"summer",8:"summer",9:"summer",10:"winter",11:"winter"}

tsl = pd.read_csv(path_snowlines)
t0, t1 = pd.to_datetime("2000-01-01"), pd.to_datetime("2009-12-31")
tobs = tsl.copy(); tobs['LS_DATE'] = pd.to_datetime(tobs['LS_DATE'])
tobs = tobs.loc[(tobs['LS_DATE']>t0)&(tobs['LS_DATE']<=t1)].set_index('LS_DATE')
tobs['SC_stdev'] = tobs['SC_stdev']/(tobs['glacier_DEM_max']-tobs['glacier_DEM_min'])
thr = 20/(tobs['glacier_DEM_max'].iloc[0]-tobs['glacier_DEM_min'].iloc[0])
tobs['SC_stdev'] = np.where(tobs['SC_stdev']<thr, thr, tobs['SC_stdev'])
tobs['season'] = tobs.index.month.map(season_lookup)
tsl_obs_vec, tsl_obs_unc = tobs['TSL_normalized'].values, tobs['SC_stdev'].values
tsl_is_summer = (tobs['season'].values=="summer")

albd = xr.open_dataset(alb_obs_path).sortby("time")
alb_season = np.array([season_lookup[m.item()] for m in albd["time"].dt.month])
alb_obs_vec, alb_obs_unc = albd['median_albedo'].values, albd['sigma_albedo'].values
alb_is_summer = (alb_season=="summer")

geod = pd.read_csv(path_to_geodetic)
geod = geod[(geod['rgiid']=="RGI60-11.00897")&(geod['period']=="2000-01-01_2010-01-01")]
mb_obs, mb_obs_unc = float(geod['dmdtda'].iloc[0]), float(geod['err_dmdtda'].iloc[0])

def reduce_run(csv):
    d = pd.read_csv(csv, index_col=0); d.index = pd.to_datetime(d.index)
    cal = d.loc[str(MB_YEAR_START):str(MB_YEAR_END)]
    grp = cal[CSV_MB_COL].groupby(cal.index.year)
    annual = grp.sum()[grp.count()>=365]
    mod_mb = float(annual.mean())
    mod_alb = d[CSV_ALB_COL].reindex(pd.to_datetime(albd["time"].values)).values
    mod_tsl = d[CSV_SLA_COL].reindex(pd.to_datetime(tobs.index)).values if CSV_SLA_COL in d.columns else np.full(len(tobs), np.nan)
    return mod_mb, mod_tsl, mod_alb

outputf = GSA_OUT + "gsa_result_sim_.csv"

from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# single output file
bmb, btsl, balb = reduce_run(outputf)

print("\n===== Run metrics =====")
print(
    f"  MB     : modelled {bmb:+.3f}  vs geodetic {mb_obs:+.3f}  "
    f"|err|={abs(bmb-mb_obs):.3f} m w.e./yr"
)

print(
    f"  albedo : R2={r2_score(alb_obs_vec, balb):.3f}  "
    f"RMSE={root_mean_squared_error(alb_obs_vec, balb):.3f}  "
    f"MAE={mean_absolute_error(alb_obs_vec, balb):.3f}"
)

if not np.isnan(btsl).any():
    print(
        f"  SLA    : R2={r2_score(tsl_obs_vec, btsl):.3f}  "
        f"RMSE={root_mean_squared_error(tsl_obs_vec, btsl):.3f}  "
        f"MAE={mean_absolute_error(tsl_obs_vec, btsl):.3f}"
    )

C_MB, C_ALB, C_SLA = "#D81B1B", "#1E80E5", "#A5781B"

plt.rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(24, 8), dpi=300)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

# Mass balance
axM = fig.add_subplot(gs[0, 0])

axM.axhspan(
    mb_obs - mb_obs_unc,
    mb_obs + mb_obs_unc,
    color="gray",
    alpha=0.25,
    label=r"$B_{geod}\pm\sigma$"
)

axM.axhline(
    mb_obs,
    color="k",
    ls="--",
    label=r"$B_{geod}$"
)

axM.scatter(
    [0],
    [bmb],
    color=C_MB,
    edgecolor="k",
    s=130,
    label="Posterior Median (bias corrected)"
)

axM.set_xticks([])
axM.set_ylabel("MB (m w.e. a$^{-1}$)")
axM.set_title("(a) Mass balance")
axM.legend(fontsize=16)
axM.grid(True, axis="y")

# Albedo
axA = fig.add_subplot(gs[0, 1])

axA.errorbar(
    balb,
    alb_obs_vec,
    yerr=alb_obs_unc,
    fmt="o",
    alpha=0.8,
    capsize=3,
    color=C_ALB,
    ecolor="gray"
)

axA.axline((0, 0), slope=1, ls="--", color="k")

r2a = r2_score(alb_obs_vec, balb)
rma = root_mean_squared_error(alb_obs_vec, balb)

axA.text(
    0.05,
    0.95,
    f"R²={r2a:.2f}\nRMSE={rma:.2f}",
    transform=axA.transAxes,
    va="top",
    fontsize=16,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

axA.set_xlim(0, 1)
axA.set_ylim(0, 1)
axA.set_xlabel(r"Modelled $\bar{\alpha}$ (-)")
axA.set_ylabel(r"Observed $\bar{\alpha}$ (-)")
axA.set_title("(b) Albedo")
axA.grid(True)

# Snowline
axS = fig.add_subplot(gs[0, 2])

if not np.isnan(btsl).any():

    axS.errorbar(
        btsl,
        tsl_obs_vec,
        yerr=tsl_obs_unc,
        fmt="o",
        alpha=0.8,
        capsize=3,
        color=C_SLA,
        ecolor="gray"
    )

    r2t = r2_score(tsl_obs_vec, btsl)
    rmt = root_mean_squared_error(tsl_obs_vec, btsl)

    axS.text(
        0.05,
        0.95,
        f"R²={r2t:.2f}\nRMSE={rmt:.2f}",
        transform=axS.transAxes,
        va="top",
        fontsize=16,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

axS.axline((-0.2, -0.2), slope=1, ls="--", color="k")
axS.set_xlim(-0.2, 1)
axS.set_ylim(-0.2, 1)
axS.set_xlabel("Modelled Norm. SLA (-)")
axS.set_ylabel("Observed Norm. SLA (-)")
axS.set_title("(c) Snowline")
axS.grid(True)

axM.set_box_aspect(1)
axA.set_box_aspect(1)
axS.set_box_aspect(1)

fig.tight_layout()

fig.savefig(
    os.path.join(GSA_OUT, "run_fit.png"),
    bbox_inches="tight"
)