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
        rrr = pm.TruncatedNormal('rrrfactor', mu=0.75, sigma=0.1, lower=0.633, upper=0.897)
        snow = pm.TruncatedNormal("albsnow", mu=0.908, sigma=0.1, lower=0.887, upper=0.93)
        #test snow = pm.TruncatedNormal("albsnow", mu=0.9065, sigma=0.15, lower=0.75, upper=0.98)
        ice = pm.TruncatedNormal("albice", mu=0.1767, sigma=0.1, lower=0.117, upper=0.232)
        firn = pm.TruncatedNormal("albfirn", mu=0.60, sigma=0.1, lower=0.51, upper=0.683)
        aging = pm.TruncatedNormal("albaging", mu=16.44, sigma=5.2, lower=6, upper=24.8)
        #test aging = pm.TruncatedNormal("albaging", mu=15.33, sigma=10, lower=2, upper=25)
        depth = pm.TruncatedNormal("albdepth", mu=2.1, sigma=1.0, lower=1.0, upper=10.753)
        rough = pm.TruncatedNormal("iceroughness", mu=8.9, sigma=9, lower=1.22, upper=19.52)

        # Define fixed values for the other parameters
        #rrr_factor = pt.constant(0.88)
        #alb_snow = pt.constant(0.927)  # Mean of original distribution
        #alb_ice = pt.constant(0.18) #mean of 3sigma ensemble
        #alb_firn = pt.constant(0.6)
        #alb_aging = pt.constant(15)  # 6+3 from your original code
        #roughness_ice = pt.constant(1.7)
        #alb_depth = pt.constant(3)

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

        # Likelihood definitions
        mb_obs = pm.Normal("mb_obs", mu=mu_mb, sigma=geod_ref['err_dmdtda'], observed=geod_data)
        tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=np.array(tsla_obs['SC_norm']), observed=tsl_data, shape=mu_tsl.shape[0])
        alb_obs = pm.Normal("alb_obs", mu=mu_alb, sigma=np.array(alb_obs_data['sigma_albedo'].values), observed=alb_data, shape=mu_alb.shape[0])

        # Manually compute log-likelihoods
        loglike_mb = pm.logp(mb_obs, geod_data)  # Mass balance log-likelihood
        loglike_tsl = pm.logp(tsl_obs, tsl_data).mean() # Snowline log-likelihood (vector)
        loglike_alb = pm.logp(alb_obs, alb_data).mean()

        # ------------------------------------------------------------------------------
        # Standardize log-likelihoods using LHS-derived stats
        # ------------------------------------------------------------------------------
        loglike_mb_std = (loglike_mb - median_mb) / std_mb
        loglike_tsl_std = ((loglike_tsl - median_tsl) / std_tsl)
        loglike_alb_std = (loglike_alb - median_alb) / std_alb

        # ------------------------------------------------------------------------------
        # Combine standardized log-likelihoods
        # ------------------------------------------------------------------------------
        total_loglike = loglike_mb_std + loglike_tsl_std + loglike_alb_std

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
        trace.to_netcdf(f"{outpath}/chain_{chain_id}.nc")

