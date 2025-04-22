import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz
import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer #Input, Dense, Dropout, LSTM, TimeDistributed, Reshape, Conv1D, BatchNormalization, Add, GRU, Bidirectional, Layer, Concatenate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pytensor
pytensor.config.exception_verbosity = 'high'
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import xarray as xr
import sys

import multiprocessing as mp
mp.set_start_method("forkserver", force=True)

path = "/data/scratch/richteny/for_emulator/"
outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

## Load observations
path_snowlines = "/data/scratch/richteny/thesis/cosipy_test_space//data/input/HEF/snowlines/HEF-snowlines-1999-2010_manual.csv"

tsl = pd.read_csv(path_snowlines, parse_dates=True, index_col="LS_DATE")
tsl['SC_norm'] = (tsl['SC_stdev']) / (tsl['glacier_DEM_max'] - tsl['glacier_DEM_min'])
thres_unc = (20) / (tsl['glacier_DEM_max'].iloc[0] - tsl['glacier_DEM_min'].iloc[0])
print(thres_unc)

## Set observational uncertainty where smaller to atleast model resolution (20m) and where larger keep it
sc_norm = np.where(tsl['SC_norm'] < thres_unc, thres_unc, tsl['SC_norm'])
tsl['SC_norm'] = sc_norm
tsla_obs = tsl.loc["2000-01-01":"2010-01-01"]


## MB
rgi_id = "RGI60-11.00897"
path_to_geodetic = "/data/scratch/richteny/Hugonnet_21_MB/dh_11_rgi60_pergla_rates.csv"
geod_ref = pd.read_csv(path_to_geodetic)
geod_ref = geod_ref.loc[geod_ref['rgiid'] == rgi_id]
geod_ref = geod_ref.loc[geod_ref['period'] == "2000-01-01_2010-01-01"]
geod_ref = geod_ref[['dmdtda','err_dmdtda']]

## Albedo
alb_obs_data = xr.open_dataset("/data/scratch/richteny/Ren_21_Albedo/HEF_processed_HRZ-20CC-filter_albedos.nc")
alb_obs_data = alb_obs_data.sortby("time")

## Load weights
import pickle
with open(path+"loglike_stats.pkl", "rb") as f:
     all_stats = pickle.load(f)

median_mb   = all_stats["mass"]["median"]
std_mb      = all_stats["mass"]["std"]
median_tsl  = all_stats["snow"]["median"]
std_tsl     = all_stats["snow"]["std"]
median_alb  = all_stats["albedo"]["median"]
std_alb     = all_stats["albedo"]["std"]

## joint emulator snowline MB
model_full = tf.keras.models.load_model(path+ "first_test.keras")

# Print the summary of the new model
model_full.summary()

### -- Prepare forcing data -- ###
# params = pd.read_csv(path+"cosipy_synthetic_params_lhs.csv", index_col=0)
params = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/simulations/LHS-narrow_1D20m_1999_2010_fullprior.csv", index_col=0)
sim_list = [x for x in params.columns if 'sim' in x]

## get rid of redudant variables
param_data = params.drop(['center_snow_transfer', 'spread_snow_transfer','roughness_fresh_snow', 'roughness_firn','aging_factor_roughness'], axis=1)

doy = tsla_obs.index.dayofyear
# Define time features
time_sin = np.sin(2 * np.pi * doy / 365)
time_cos = np.cos(2 * np.pi * doy / 365)
time_features = np.stack([time_sin, time_cos], axis=-1)  # Shape: (62, 2)

### Repeat for albedo
#get time variable for snowline points - time dependency added to training
doy_alb = alb_obs_data.time.dt.dayofyear.data
# Define time features
time_sin_alb = np.sin(2 * np.pi * doy_alb / 365)
time_cos_alb = np.cos(2 * np.pi * doy_alb / 365)
time_features_alb = np.stack([time_sin_alb, time_cos_alb], axis=-1)  # Shape: (62, 2)

##
params_subset = param_data.copy()
print(params_subset)

## Create train test split
train_dataset, validation_dataset = train_test_split(params_subset.index, 
                                               train_size=0.8,
                                               test_size=0.2,
                                               random_state=77)

df_train = params_subset.loc[train_dataset]
df_validation = params_subset.loc[validation_dataset]

list_sims = [x for x in params_subset.columns if 'sim' in x]

# Fit scalers using training data
features_to_drop = ['mb'] + list_sims
df_train_X = df_train.drop(features_to_drop, axis=1)
df_train_y_mb = df_train[['mb']].values
df_train_y_tsla = df_train[list_sims].values

scaler = StandardScaler()
scaler.fit(df_train_X.values)

def tsl_emulator_input(param_values):
    """
    Prepares the forcing data for the TSL emulator.
    
    param_values: Array of sampled parameter values from PyMC.
    """
    
    X_train = scaler.transform(param_values)  # Shape: (n_samples, 6)

    # Repeat the input features for time points
    X_train_expanded = np.repeat(X_train[:, None, :], len(doy), axis=1)  # Shape: (n_samples, 62, 6)

    # Add time features without scaling
    X_train_with_time = np.concatenate([X_train_expanded, np.tile(time_features, (X_train.shape[0], 1, 1))], axis=-1)

    # Repeat for albedo
    X_train_expanded_alb = np.repeat(X_train[:, None, :], len(doy_alb), axis=1)  # Shape: (n_samples, 62, 6)

    # Add time features without scaling
    X_train_with_time_alb = np.concatenate([X_train_expanded_alb, np.tile(time_features_alb, (X_train.shape[0], 1, 1))], axis=-1)


    return (X_train, X_train_with_time, X_train_with_time_alb)

@as_op(itypes=[pt.dmatrix], otypes=[pt.dvector, pt.dvector, pt.dvector])
def run_emulators(param_stack):
    #param_values_numpy = param_stack.eval().astype(np.float32)
    X_train, X_train_with_time, X_train_with_time_alb = tsl_emulator_input(param_stack)

    #run it 
    predictions = model_full.predict({
        "mass_balance_input": X_train, 
        "snowlines_input": X_train_with_time,
        "alb_input": X_train_with_time_alb
    }, verbose=0)

    # Extract outputs
    mass_balance_pred = predictions[0].flatten()
    snowlines_pred = predictions[1].flatten()
    albedos_pred = predictions[2].flatten()

    #mod_tsl = np.array([closest_indices])
    
    mod_mb = np.array(mass_balance_pred, dtype=np.float64)
    mod_tsl = np.array(snowlines_pred, dtype=np.float64)
    mod_alb = np.array(albedos_pred, dtype=np.float64)
    
    return (mod_mb, mod_tsl, mod_alb)

## Create initvals
## Create initial values based on n_chains
from scipy.stats import qmc

def generate_initvals(N):
    """
    Generate N sets of initial values using Latin Hypercube Sampling
    that span the parameter space defined by your TruncatedNormal priors.
    """
    # Define the parameter bounds (from TruncatedNormal priors)
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

    # Create Latin Hypercube samples in [0, 1]^d
    sampler = qmc.LatinHypercube(d=len(priors))
    lhs_unit = sampler.random(n=N)

    # Scale samples to parameter bounds
    lhs_scaled = qmc.scale(lhs_unit, bounds[:, 0], bounds[:, 1])

    # Convert to list of dictionaries (one per chain)
    initvals = [
        {param: float(lhs_scaled[i, j]) for j, param in enumerate(param_names)}
        for i in range(N)
    ]

    return initvals

n_chains = 4
initvals = generate_initvals(n_chains)
print(initvals)

with pm.Model() as model:
    rrr_factor = pm.TruncatedNormal('rrrfactor', mu=0.84, sigma=0.13, lower=0.5738, upper=1.29) #sigma=0.182
    alb_snow = pm.TruncatedNormal("albsnow", mu=0.907, sigma=0.1, lower=0.887, upper=0.93)
    alb_ice = pm.TruncatedNormal("albice", mu=0.173, sigma=0.1, lower=0.115, upper=0.233)
    alb_firn = pm.TruncatedNormal("albfirn", mu=0.593, sigma=0.1, lower=0.5, upper=0.692)
    alb_aging = pm.TruncatedNormal("albaging", mu=14.37, sigma=9, lower=2, upper=25)
    alb_depth = pm.TruncatedNormal("albdepth", mu=6.07, sigma=3.53, lower=1, upper=14)
    roughness_ice = pm.TruncatedNormal("iceroughness", mu=9.875, sigma=9, lower=0.92, upper=20)
   
    # Define fixed values for the other parameters
    #rrr_factor = pt.constant(0.88)
    #alb_snow = pt.constant(0.927)  # Mean of original distribution
    #alb_ice = pt.constant(0.223)
    #alb_firn = pt.constant(0.6)
    #alb_aging = pt.constant(15)  # 6+3 from your original code
    #roughness_ice = pt.constant(1.7)
    #alb_depth = pt.constant(3)
    
    
    #Setup observations
    geod_data = pm.Data('geod_data', np.array([geod_ref['dmdtda']]))
    tsl_data = pm.Data('tsl_data', np.array(tsla_obs['TSL_normalized']))
    alb_data = pm.Data('alb_data', np.array(alb_obs_data['mean_albedo'].values))

    #created all parameters, now when we sample - need to combine all parameters to pass to emulator input function
    param_values = pm.math.stack([rrr_factor, alb_ice, alb_snow, alb_firn, alb_aging, alb_depth, roughness_ice], axis=0)
    param_values = param_values.reshape((1, 7))  # Ensure it's in the shape (1, 7)
    
    #param_values_numpy = param_values.eval().astype(np.float32)  # Convert to NumPy array

    modmb, modtsl, modalb = run_emulators(param_values)

    # store output
    mu_mb = pm.Deterministic('mu_mb', modmb)
    mu_tsl = pm.Deterministic('mu_tsl', modtsl)
    mu_alb = pm.Deterministic('mu_alb', modalb)
    
    # Likelihood definitions
    mb_obs = pm.Normal("mb_obs", mu=mu_mb, sigma=geod_ref['err_dmdtda'], observed=geod_data)
    tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=np.array(tsla_obs['SC_norm']), observed=tsl_data, shape=mu_tsl.shape[0])
    alb_obs = pm.Normal("alb_obs", mu=mu_alb, sigma=np.array(alb_obs_data['sigma_albedo'].values), observed=alb_data, shape=mu_alb.shape[0])
    #tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=np.array([0.023697]), observed=tsl_data, shape=mu_tsl.shape[0])
    # Manually compute log-likelihoods
    loglike_mb = pm.logp(mb_obs, geod_data)  # Mass balance log-likelihood
    loglike_tsl = pm.logp(tsl_obs, tsl_data).mean() # Snowline log-likelihood (vector)
    loglike_alb = pm.logp(alb_obs, alb_data).mean()

    # ------------------------------------------------------------------------------
    # Standardize log-likelihoods using LHS-derived stats
    # ------------------------------------------------------------------------------
    loglike_mb_std = (loglike_mb - median_mb) / std_mb
    loglike_tsl_std = (loglike_tsl - median_tsl) / std_tsl
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
    #step = pm.DEMetropolis()
    #step = pm.Metropolis()
    #initvals = [{'rrrfactor': 1.2, 'albaging': 3, 'albfirn': 0.52}, {'rrrfactor': 0.53, 'albaging': 22, 'albfirn': 0.66}] #'albdepth': 8} taken from script above!

    post = pm.sample(draws=10, tune=1, step=step, initvals=initvals, return_inferencedata=True,
                     chains=n_chains, cores=4, progressbar=True, discard_tuned_samples=True)
    post.to_netcdf("/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/simulations_HEF_emulator_MCMC_testing.nc")
    
