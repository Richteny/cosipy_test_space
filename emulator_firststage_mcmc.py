#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pymc as pm
import tensorflow as tf
import pickle
import xarray as xr
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

# =================== Config =====================
# Paths (set these accordingly)
path = "/data/scratch/richteny/for_emulator/"
outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

# Input
params = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/simulations/LHS-narrow_1D20m_1999_2010_fullprior.csv", index_col=0)
path_snowlines = "/data/scratch/richteny/thesis/cosipy_test_space/data/input/HEF/snowlines/HEF-snowlines-1999-2010_manual_filtered.csv"
path_to_geodetic = "/data/scratch/richteny/Hugonnet_21_MB/dh_11_rgi60_pergla_rates.csv"
alb_obs_data = xr.open_dataset("/data/scratch/richteny/Ren_21_Albedo/HEF_processed_HRZ-30CC-filter_albedos.nc")
alb_obs_data = alb_obs_data.sortby("time") #ensure correct order - prob. not necessary

#Season lookup (only JJAS, rest winter)
season_lookup = {
    12: "winter", 1: "winter", 2: "winter",
    3: "winter", 4: "winter", 5: "summer",
    6: "summer", 7: "summer", 8: "summer",
    9: "summer", 10: "winter", 11: "winter"
}

months = alb_obs_data["time"].dt.month
season_str = xr.DataArray([season_lookup[m.item()] for m in months], coords={"time": alb_obs_data["time"]}, dims="time")
alb_obs_data = alb_obs_data.assign_coords(season=season_str)


# Load emulator statistics
with open(path+"loglike_stats.pkl", "rb") as f:
    all_stats = pickle.load(f)
median_mb = all_stats["mass"]["mean"]
std_mb = all_stats["mass"]["std"]
median_tsl = all_stats["snow"]["mean"]
std_tsl = all_stats["snow"]["std"]
median_alb = all_stats["albedo"]["mean"]
std_alb = all_stats["albedo"]["std"]

# Normalize and preprocess snowline
tsl = pd.read_csv(path_snowlines, parse_dates=True, index_col="LS_DATE")
thres_unc = 20 / (tsl['glacier_DEM_max'].iloc[0] - tsl['glacier_DEM_min'].iloc[0])
tsla_obs = tsl.loc["2000-01-01":"2010-01-01"]
tsla_obs['SC_norm'] = np.maximum(tsla_obs['SC_stdev'] / (tsla_obs['glacier_DEM_max'] - tsla_obs['glacier_DEM_min']), thres_unc)
tsla_obs['season'] = tsla_obs.index.month.map(season_lookup)

# Preprocess geodetic
geod_ref = pd.read_csv(path_to_geodetic)
geod_ref = geod_ref[(geod_ref['rgiid'] == "RGI60-11.00897") & (geod_ref['period'] == "2000-01-01_2010-01-01")]

# Time features
doy = tsla_obs.index.dayofyear
time_features = np.stack([np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)], axis=-1)
doy_alb = alb_obs_data.time.dt.dayofyear.data
time_features_alb = np.stack([np.sin(2 * np.pi * doy_alb / 365), np.cos(2 * np.pi * doy_alb / 365)], axis=-1)

# Subset parameters
param_data = params.drop(['center_snow_transfer', 'spread_snow_transfer','roughness_fresh_snow', 'roughness_firn','aging_factor_roughness'], axis=1)
scaler = StandardScaler()
scaler.fit(param_data.drop(['mb'] + [c for c in param_data.columns if 'sim' in c], axis=1).values)

# ============ Emulator Interface =============
def tsl_emulator_input(param_values):
    X = scaler.transform(param_values)
    X_time = np.repeat(X[:, None, :], len(doy), axis=1)
    X_alb_time = np.repeat(X[:, None, :], len(doy_alb), axis=1)

    X_train_with_time = np.concatenate([X_time, np.tile(time_features, (X.shape[0], 1, 1))], axis=-1)
    X_train_with_time_alb = np.concatenate([X_alb_time, np.tile(time_features_alb, (X.shape[0], 1, 1))], axis=-1)

    return (X, X_train_with_time, X_train_with_time_alb)

@as_op(itypes=[pt.dmatrix], otypes=[pt.dvector, pt.dvector, pt.dvector])
def run_emulators(param_stack):
    X, X_time, X_alb_time = tsl_emulator_input(param_stack)
    predictions = model_full.predict({"mass_balance_input": X, "snowlines_input": X_time, "alb_input": X_alb_time}, verbose=0)

    # Extract outputs
    mass_balance_pred = predictions[0].flatten()
    #filtered indices to exclude
    #to_exclude_filter = [4,7,9,18,21,24,29,32,39,40,45,49,52,53,55,57,60,62,69,77,78]
    snowlines_pred = predictions[1].flatten()
    albedos_pred = predictions[2].flatten()

    mod_mb = np.array(mass_balance_pred, dtype=np.float64)
    #amod_tsl = np.array(snowlines_pred, dtype=np.float64)
    #mod_tsl = np.delete(amod_tsl, to_exclude_filter) #extra 
    mod_tsl = np.array(snowlines_pred, dtype=np.float64)
    mod_alb = np.array(albedos_pred, dtype=np.float64)
    
    return (mod_mb, mod_tsl, mod_alb)

# ============ Init Value Generation =============
def generate_initvals(N):
    priors = {
        'rrrfactor': (0.5738, 1.29),
        'albsnow': (0.887, 0.93),
        'albice': (0.115, 0.233),
        'albfirn': (0.5, 0.692),
        'albaging': (2, 25),
        'albdepth': (1, 14),
        'iceroughness': (0.92, 20),
    }

    param_names = list(priors.keys())
    bounds = np.array(list(priors.values()))

    sampler = qmc.LatinHypercube(d=len(priors))
    lhs_unit = sampler.random(n=N)
    lhs_scaled = qmc.scale(lhs_unit, bounds[:,0], bounds[:,1])

    initvals = [
        {param: float(lhs_scaled[i, j]) for j, param in enumerate(param_names)}
        for i in range(N)
    ]
    return initvals

# ============ Main Chain Runner =============
if __name__ == "__main__":
    chain_id = int(sys.argv[1])
    #initvals = generate_initvals(20)
    with open(path+"initvals.pkl", "rb") as f:
    #with open (path+"albaging_initvals.pkl", "rb") as f:
        initvals = pickle.load(f)

    initval = initvals[chain_id]

    model_full = tf.keras.models.load_model(path + "first_test.keras")

    with pm.Model() as model:
        #Stage 1 Params: TSLA + ALB only
        rrr = pm.TruncatedNormal('rrrfactor', mu=0.7785, sigma=0.781, lower=0.648, upper=0.946)
        snow = pm.TruncatedNormal("albsnow", mu=0.903, sigma=0.1, lower=0.887, upper=0.928)
        ice = pm.TruncatedNormal("albice", mu=0.17523, sigma=0.1, lower=0.1182, upper=0.2302)
        firn = pm.TruncatedNormal("albfirn", mu=0.6036, sigma=0.1, lower=0.51, upper=0.6747)
        aging = pm.TruncatedNormal("albaging", mu=13.82, sigma=5.372, lower=5, upper=24.77)
        depth = pm.TruncatedNormal("albdepth", mu=1.776, sigma=0.666, lower=1.0, upper=4)
        rough = pm.TruncatedNormal("iceroughness", mu=8.612, sigma=9, lower=1.2, upper=19.65)
        
        #Stage 2 Params: Keep separate
        #rough = pm.Uniform("iceroughness", lower=1.22, upper=19.52)
        #ice = pm.Uniform("albice", lower=0.117, upper=0.232)
        
        # Define fixed values for the other parameters
        #rrr_factor = pt.constant(0.88)
        #alb_snow = pt.constant(0.927)  # Mean of original distribution
        #alb_ice = pt.constant(0.18) #mean of 3sigma ensemble
        #alb_firn = pt.constant(0.6)
        #alb_aging = pt.constant(15)  # 6+3 from your original code
        #roughness_ice = pt.constant(1.7)
        #alb_depth = pt.constant(3)

        # Priors for additional systematic errors
        #sigma_mb_sys = pm.HalfNormal("sigma_mb_sys", sigma=0.2)  # in dmdtda units
        sigma_tsl_winter = pt.constant(0)
        #sigma_tsl_winter = pm.HalfNormal("sigma_tsl_winter", sigma=0.01)  # normalized units
        sigma_tsl_summer = pt.constant(0)
        #sigma_tsl_summer = pm.HalfNormal("sigma_tsl_summer", sigma=0.03)
        sigma_alb_winter = pt.constant(0)
        sigma_alb_summer = pm.HalfNormal("sigma_alb_summer", sigma=0.02)

        #Setup observations
        geod_data = pm.Data('geod_data', np.array([geod_ref['dmdtda']]))
        tsl_data = pm.Data('tsl_data', np.array(tsla_obs['TSL_normalized']))
        alb_data = pm.Data('alb_data', np.array(alb_obs_data['median_albedo'].values))

        param_values = pm.math.stack([rrr, ice, snow, firn, aging, depth, rough], axis=0).reshape((1, 7))
        modmb, modtsl, modalb = run_emulators(param_values)

        # store output
        mu_mb = pm.Deterministic('mu_mb', modmb)
        mu_tsl = pm.Deterministic('mu_tsl', modtsl)
        mu_alb = pm.Deterministic('mu_alb', modalb)

        # --------------------------------------------------------------------------
        # Build full error terms: sqrt(obs^2 + sys^2)
        # --------------------------------------------------------------------------

        # MB (scalar)
        mb_obs_unc = np.array([geod_ref['err_dmdtda'].item()])
        #sigma_mb_total = pm.Deterministic("sigma_mb_total", pm.math.sqrt(mb_obs_unc**2 + sigma_mb_sys**2))

        # TSL (vector)
        tsl_obs_unc = np.array(tsla_obs['SC_norm'])
        is_winter_tsl = tsla_obs['season'].values == "winter"
        tsl_sys_errors = pt.where(is_winter_tsl, sigma_tsl_winter, sigma_tsl_summer)
        #print(tsl_sys_errors)
        sigma_tsl_total = pm.Deterministic("sigma_tsl_total",
             pm.math.sqrt(tsl_obs_unc**2 + tsl_sys_errors**2))

        # Albedo (vector)
        alb_obs_unc = np.array(alb_obs_data['sigma_albedo'].values)
        is_winter_alb = alb_obs_data['season'].values == "winter"
        alb_sys_errors = pt.where(is_winter_alb, sigma_alb_winter, sigma_alb_summer)
        sigma_alb_total = pm.Deterministic("sigma_alb_total",
            pm.math.sqrt(alb_obs_unc**2 + alb_sys_errors**2))

        # Likelihood definitions
        mb_obs = pm.Normal("mb_obs", mu=mu_mb, sigma=mb_obs_unc, observed=geod_data)
        tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=sigma_tsl_total, observed=tsl_data, shape=mu_tsl.shape[0])
        alb_obs = pm.Normal("alb_obs", mu=mu_alb, sigma=sigma_alb_total, observed=alb_data, shape=mu_alb.shape[0])

        # Manually compute log-likelihoods
        loglike_mb = pm.logp(mb_obs, geod_data)  # Mass balance log-likelihood
        loglike_tsl_std = pm.logp(tsl_obs, tsl_data).mean() # Snowline log-likelihood (vector)
        loglike_alb = pm.logp(alb_obs, alb_data).mean()

        # ------------------------------------------------------------------------------
        # Standardize log-likelihoods using LHS-derived stats
        # ------------------------------------------------------------------------------
        loglike_mb_std = (loglike_mb - median_mb) / std_mb
        #loglike_tsl_std = ((loglike_tsl - median_tsl) / std_tsl)
        loglike_alb_std = (loglike_alb - median_alb) / std_alb

        # ------------------------------------------------------------------------------
        # Combine standardized log-likelihoods
        # ------------------------------------------------------------------------------
        #w1 = 1.0
        #w2 = 3.0
        total_loglike = (0.1*loglike_alb_std) + (0.9*loglike_mb_std) #first stage only alb and mb

        # Store components
        pm.Deterministic("loglike_mb", loglike_mb_std)
        pm.Deterministic("loglike_tsl", loglike_tsl_std)
        pm.Deterministic("loglike_alb", loglike_alb_std)
        pm.Deterministic("total_loglike", total_loglike)

        # Use this combined, standardized score as model likelihood
        pm.Potential("balanced_loglike", total_loglike)

        step = pm.DEMetropolisZ()
        ##step = pm.DEMetropolis()
        #step = pm.Metropolis()

        trace = pm.sample(draws=100000, tune=10000, chains=1, cores=1, initvals=[initval], step=step, discard_tuned_samples=True, return_inferencedata=True, progressbar=False)
        trace.to_netcdf(f"{outpath}/stage1_chain_{chain_id}.nc")

