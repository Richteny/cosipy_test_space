#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pymc as pm
import tensorflow as tf
import pickle
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

# =================== Config =====================
# Paths (set these accordingly)
path = "/data/scratch/richteny/for_emulator/"
outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

# Input
df = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/pointLHS-params_fullv1.csv")
params = df.drop(['chain','like1'], axis=1)
params['parRRR_factor'] = np.exp(params['parRRR_factor'])

observed = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/data/input/HEF/cosipy_validation_upper_station.csv",
                       parse_dates=True, index_col="time")
observed = observed.resample("5D").agg({'LWout': np.nanmean, 'SR50': "last", 'albedo': np.nanmin})
unc_lwo = 15 #25 Wm2
unc_alb = 0.05 #10%
unc_sfc = 0.12 #0.4m

# Load stats
with open(path+"point_loglike_stats.pkl", "rb") as f:
    all_stats = pickle.load(f)
mean_sfc = all_stats["sfc_mean"]
std_sfc = all_stats["sfc_std"]
mean_alb = all_stats["alb_mean"]
std_alb = all_stats["alb_std"]
mean_lwo = all_stats["lwo_mean"]
std_lwo = all_stats["lwo_std"]

param_cols = [col for col in params.columns if not col.startswith('simulation')]
sim_cols = [col for col in params.columns if col.startswith('simulation')]

print(f"Param cols: {len(param_cols)}, Sim cols: {len(sim_cols)}")  # sim_cols should be 1098

# 2) Split sim_cols into 3 equal parts (for the 3 simulations)
sim_len = 366  # days per simulation
simulation1_cols = sim_cols[0:sim_len]
simulation2_cols = sim_cols[sim_len:2*sim_len]
simulation3_cols = sim_cols[2*sim_len:3*sim_len]

# 3) Create date index for daily data
dates = pd.date_range(start='2003-10-01', periods=sim_len, freq='D')

def resample_simulation(simulation_cols, sim_name, agg):
    sim_df = params[simulation_cols].copy()
    sim_df.columns = dates  # assign dates as columns
    
    # Resample weekly (7-day bins)
    if agg == "mean":
        weekly_df = sim_df.resample('5D', axis=1).mean()
    elif agg == "min":
        weekly_df = sim_df.resample('5D', axis=1).min()
    else:
        weekly_df = sim_df.resample('5D', axis=1).last()
    
    # Rename columns: simulationX_weekY
    weekly_colnames = [f"{sim_name}_week{week+1}" for week in range(len(weekly_df.columns))]
    weekly_df.columns = weekly_colnames
    
    return weekly_df

# 4) Resample each simulation separately
weekly_sim1 = resample_simulation(simulation1_cols, "simulation1", "mean") #lwo
weekly_sim2 = resample_simulation(simulation2_cols, "simulation2", "min") #alb
weekly_sim3 = resample_simulation(simulation3_cols, "simulation3", "last") #sfc

# 5) Concatenate parameter columns and all resampled simulations
final_df = pd.concat([params[param_cols], weekly_sim1, weekly_sim2, weekly_sim3], axis=1)

print(final_df.shape)
params = final_df.copy()
list_sims_lwo = [x for x in params.columns if 'simulation1_' in x]
list_sims_alb = [x for x in params.columns if 'simulation2_' in x]
list_sims_sfc = [x for x in params.columns if 'simulation3_' in x]

### Create training data! ###
doy = observed.index.dayofyear
n_steps = len(observed)
relative_time = np.arange(n_steps) / (n_steps - 1)
# Time features
time_sin = np.sin(2 * np.pi * doy / 366) #leap year
time_cos = np.cos(2 * np.pi * doy / 366) #leap year
time_features_lwo = np.stack([time_sin, time_cos, relative_time], axis=-1)  # Shape: (62, 2)

time_features_alb = np.copy(time_features_lwo)
time_features_sfc = np.copy(time_features_lwo)

#Select test/train split
params['stratify_col'] = params[list_sims_alb].min(axis=1)  # or .mean(axis=1)

# Bin the values to create stratification categories
params['stratify_bin'] = pd.qcut(params['stratify_col'], q=5, duplicates='drop')
train_dataset, validation_dataset = train_test_split(params.index, 
                                               train_size=0.7,
                                               test_size=0.3, stratify=params['stratify_bin'],
                                               random_state=42)

df_train = params.loc[train_dataset].drop(['stratify_bin','stratify_col'], axis=1)
df_validation = params.loc[validation_dataset].drop(['stratify_bin','stratify_col'], axis=1)

# Subset parameters
features_to_drop = list_sims_lwo + list_sims_alb + list_sims_sfc
df_train_X = df_train.drop(features_to_drop, axis=1)
df_train_y_lwo = (df_train[list_sims_lwo]*-1).values
df_train_y_alb = df_train[list_sims_alb].values
df_train_y_sfc = df_train[list_sims_sfc].values

scaler = StandardScaler()
scaler.fit(df_train_X.values)

# ============ Emulator Interface =============
def tsl_emulator_input(param_values):
    X_train = scaler.transform(param_values)

    # Repeat the input features for time points
    X_train_expanded_lwo = np.repeat(X_train[:, None, :], len(doy), axis=1)  # LWO
    X_train_expanded_alb = np.repeat(X_train[:, None, :], len(doy), axis=1) # ALB
    X_train_expanded_sfc = np.repeat(X_train[:, None, :], len(doy), axis=1) # SFC

    # Add time features without scaling
    X_train_with_time_lwo = np.concatenate([X_train_expanded_lwo, np.tile(time_features_lwo, (X_train.shape[0], 1, 1))], axis=-1)
    X_train_with_time_alb = np.concatenate([X_train_expanded_alb, np.tile(time_features_alb, (X_train.shape[0], 1, 1))], axis=-1)
    X_train_with_time_sfc = np.concatenate([X_train_expanded_sfc, np.tile(time_features_sfc, (X_train.shape[0], 1, 1))], axis=-1)

    return (X_train_with_time_lwo, X_train_with_time_alb, X_train_with_time_sfc)

@as_op(itypes=[pt.dmatrix], otypes=[pt.dvector, pt.dvector, pt.dvector])
def run_emulators(param_stack):
    X_lwo, X_alb, X_sfc = tsl_emulator_input(param_stack)
    predictions = model_full.predict({"lwo_input": X_lwo, "alb_input": X_alb, "sfc_input": X_sfc}, verbose=0)

    # Extract outputs
    lwo_pred = predictions[0].flatten()
    alb_pred = predictions[1].flatten()
    sfc_pred = predictions[2].flatten()

    mod_lwo = np.array(lwo_pred, dtype=np.float64)
    mod_alb = np.array(alb_pred, dtype=np.float64)
    mod_sfc = np.array(sfc_pred, dtype=np.float64)
    
    return (mod_lwo, mod_alb, mod_sfc)

# ============ Main Chain Runner =============
if __name__ == "__main__":
    chain_id = int(sys.argv[1])
    #initvals = generate_initvals(20)
    with open(path+"point_initvals.pkl", "rb") as f:
        initvals = pickle.load(f)

    initval = initvals[chain_id]

    model_full = tf.keras.models.load_model(path + "point_model_emul.keras")

    with pm.Model() as model:
        rrr = pm.TruncatedNormal('rrrfactor', mu=0.27, sigma=0.1, lower=0.1, upper=0.6)
        snow = pm.TruncatedNormal("albsnow", mu=0.90, sigma=0.1, lower=0.88, upper=0.93)
        #test snow = pm.TruncatedNormal("albsnow", mu=0.9065, sigma=0.15, lower=0.75, upper=0.98)
        ice = pm.TruncatedNormal("albice", mu=0.18, sigma=0.1, lower=0.11, upper=0.25)
        firn = pm.TruncatedNormal("albfirn", mu=0.55, sigma=0.04, lower=0.46, upper=0.65)
        aging = pm.TruncatedNormal("albaging", mu=3, sigma=0.81, lower=1, upper=8)
        #test aging = pm.TruncatedNormal("albaging", mu=15.33, sigma=10, lower=2, upper=25)
        depth = pm.TruncatedNormal("albdepth", mu=7.66, sigma=3.96, lower=0.9, upper=15.00)
        rough = pm.TruncatedNormal("iceroughness", mu=9.86, sigma=5.63, lower=0.7, upper=19.52)
        centersnow = pm.TruncatedNormal("centersnow", mu=-0.49, sigma=1.27, lower=-3, upper=1)

        # Define fixed values for the other parameters
        #rrr_factor = pt.constant(0.88)
        #alb_snow = pt.constant(0.927)  # Mean of original distribution
        #alb_ice = pt.constant(0.18) #mean of 3sigma ensemble
        #alb_firn = pt.constant(0.6)
        #alb_aging = pt.constant(15)  # 6+3 from your original code
        #roughness_ice = pt.constant(1.7)
        #alb_depth = pt.constant(3)

        #Setup observations
        lwo_data = pm.Data('lwo_data', np.array(observed['LWout']))
        alb_data = pm.Data('alb_data', np.array(observed['albedo']))
        sfc_data = pm.Data('sfc_data', np.array(observed['SR50']))

        param_values = pm.math.stack([rrr, ice, snow, firn, aging, depth, rough, centersnow], axis=0).reshape((1, 8))
        modlwo, modalb, modsfc = run_emulators(param_values)

        # store output
        mu_lwo = pm.Deterministic('mu_lwo', modlwo)
        mu_alb = pm.Deterministic('mu_alb', modalb)
        mu_sfc = pm.Deterministic('mu_sfc', modsfc)

        # Likelihood definitions
        lwo_obs = pm.Normal("lwo_obs", mu=mu_lwo, sigma=np.array([unc_lwo]), observed=lwo_data, shape=mu_lwo.shape[0])
        alb_obs = pm.Normal("alb_obs", mu=mu_alb, sigma=np.array([unc_alb]), observed=alb_data, shape=mu_alb.shape[0])
        sfc_obs = pm.Normal("sfc_obs", mu=mu_sfc, sigma=np.array([unc_sfc]), observed=sfc_data, shape=mu_sfc.shape[0])

        # Manually compute log-likelihoods
        loglike_lwo = pm.logp(lwo_obs, lwo_data).mean()  # LWO
        loglike_alb = pm.logp(alb_obs, alb_data).mean() # ALB)
        loglike_sfc = pm.logp(sfc_obs, sfc_data).mean()

        # Standardize log-likelihoods using LHS-derived stats #
        loglike_lwo_std = (loglike_lwo - mean_lwo) / std_lwo
        loglike_alb_std = (loglike_alb - mean_alb) / std_alb
        loglike_sfc_std = (loglike_sfc - mean_sfc) / std_sfc

        # ------------------------------------------------------------------------------
        # Combine standardized log-likelihoods
        # ------------------------------------------------------------------------------
        total_loglike = loglike_lwo_std + loglike_alb_std + loglike_sfc_std

        # Store components
        pm.Deterministic("loglike_lwo", loglike_lwo_std)
        pm.Deterministic("loglike_alb", loglike_alb_std)
        pm.Deterministic("loglike_sfc", loglike_sfc_std)
        pm.Deterministic("total_loglike", total_loglike)

        # Use this combined, standardized score as model likelihood
        pm.Potential("balanced_loglike", total_loglike)

        step = pm.DEMetropolisZ()
        ##step = pm.DEMetropolis()
        #step = pm.Metropolis()

        trace = pm.sample(draws=100000, tune=20000, chains=1, cores=1, initvals=[initval], step=step, discard_tuned_samples=True, return_inferencedata=True, progressbar=False)
        trace.to_netcdf(f"{outpath}/point_chain_{chain_id}.nc")

