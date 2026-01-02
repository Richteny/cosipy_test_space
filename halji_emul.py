#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pymc as pm
import tensorflow as tf
import pickle
import joblib
import xarray as xr
from scipy.stats import qmc
import pytensor.tensor as pt
from pytensor.compile.ops import as_op

# =================== Config =====================
# Paths
path = "/data/scratch/richteny/for_emulator/Halji/"
outpath = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

# Input Data
# Note: Ensure this CSV has columns for 'lwin_factor' and 'ws_factor' if using it for prior stats, 
# otherwise we set generic priors below.
path_snowlines = "/data/scratch/richteny/thesis/cosipy_test_space/data/input/Halji/snowlines/Halji_TSLA_fixed-1990-2025.csv" 
path_to_geodetic = "/data/scratch/richteny/Hugonnet_21_MB/dh_15_rgi60_pergla_rates.csv"
alb_obs_path = "/data/scratch/richteny/Ren_21_Albedo/Halji_hrz-merged_mean-albedos.nc"

# Climate & Geometry (REQUIRED for Context Generation)
# Must match what was used for training
path_climate = path+"COSIPY_HARv2_1980_2022_RGI60-15.06065.csv" 
path_geometry = path+"Halji_HARv2_1988-2022_RRR-0.8549_0.8861_0.196_0.6075_12.9837_6.8571_0.24_3.1025_4.0_0.0026_0.9725_2.3269_num2.nc"

# =================== Data Loading & Prep =====================

# 1. Load Observations
# SLA: 1990 to 2022
tsl = pd.read_csv(path_snowlines, parse_dates=True, index_col="LS_DATE")
tsla_obs = tsl.loc["1990-01-01":"2022-12-31"] 
# Simple error estimation (normalized)
tsla_obs['SC_norm'] = tsla_obs['SC_stdev'] / (tsla_obs['glacier_DEM_max'] - tsla_obs['glacier_DEM_min'])

thres_unc = (20) / (tsla_obs['glacier_DEM_max'].iloc[0] - tsla_obs['glacier_DEM_min'].iloc[0])
## Set observational uncertainty where smaller to atleast model resolution (20m) and where larger keep it
sc_norm = np.where(tsla_obs['SC_stdev'] < thres_unc, thres_unc, tsla_obs['SC_stdev'])
tsla_obs['SC_stdev'] = sc_norm

# Albedo
alb_obs_data = xr.open_dataset(alb_obs_path).sortby("time")

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
tsla_obs['season'] = tsla_obs.index.month.map(season_lookup)

# Mass Balance: 2000 to 2020
geod_ref = pd.read_csv(path_to_geodetic)
# Adjust period string to match your geodetic file format if different
geod_ref = geod_ref[(geod_ref['rgiid'] == "RGI60-15.06065") & (geod_ref['period'] == "2000-01-01_2020-01-01")]

#Climate data
forcing = xr.open_dataset(path_geometry)
forcing = forcing.sel(time=slice("1990-01-01","2022-12-31"))
rrr_factor = 0.8549
lwin_factor = 0.9725
ws_factor = 2.3269
#revert fields to normal values
forcing['SNOWFALL'] = forcing['SNOWFALL'] / rrr_factor
forcing['LWin'] = forcing['LWin'] / lwin_factor
forcing['U2'] = forcing['U2'] / ws_factor
#load file with lapse rates and add to ds
df_clim = pd.read_csv(path_climate, parse_dates=True, index_col="TIMESTAMP")
df_clim = df_clim.loc["1990-01-01":"2022-12-31"]
forcing['lr_t2'] = (('time'), df_clim['lr_t2m'].values)

# 2. Load Models & Scalers
print("Loading Emulators...")
model_mb = tf.keras.models.load_model(path + "model_mb.keras")
model_sla = tf.keras.models.load_model(path + "model_sla.keras")
model_alb = tf.keras.models.load_model(path + "model_albedo.keras")

scalers_mb = joblib.load(path + "scalers_mb.pkl")
scalers_sla = joblib.load(path + "scalers_sla.pkl")
scalers_alb = joblib.load(path + "scalers_albedo.pkl")

# 3. Context Generation Helper (From your Class)
def generate_context_matrix(target_dates, ds_cl, ds_geo, scaler_ctx):
    """
    Replicates GlacierEmulatorData._precompute_reference_climate logic.
    Calculates PDD, T30, P30, Snow Days, and Geometry.
    """
    print(f"Generating context features for {len(target_dates)} points...")
    
    # Weights & Pre-processing
    weights = ds_geo['N_Points'].fillna(0)
    weighted_cl = ds_cl.weighted(weights).mean(dim=['lat', 'lon'])
    
    t2_k = weighted_cl['T2']
    if t2_k.mean() < 200: t2_k += 273.15
    t2_c = t2_k - 273.15
    sf = weighted_cl['SNOWFALL']
    
    # --- Feature Engineering (Exact match to training) ---
    # 1. Physics
    pdd_7 = t2_c.clip(min=0).rolling(time=7, center=False).sum().fillna(0)
    t_30 = t2_c.rolling(time=30, center=False).mean().fillna(0)
    p_30 = sf.rolling(time=30, center=False).sum().fillna(0) # Restored p_30
    
    # 2. Snow Dates
    snow_thres = 1.0/1000
    is_snow = (sf > snow_thres).values
    snow_dates = pd.to_datetime(ds_cl.time.values)[is_snow]
    
    def get_days_snow(d):
        valid = snow_dates[snow_dates <= d]
        return 365.0 if len(valid)==0 else (d - valid[-1]).days
    
    # 3. Geometry (Dynamic)
    z_mean = ds_geo['HGT'].weighted(weights).mean(dim=['lat', 'lon']).fillna(ds_geo['HGT'].max())
    active_hgt = ds_geo['HGT'].where(ds_geo['N_Points'] > 0)
    z_min = active_hgt.min(dim=['lat', 'lon']).fillna(ds_geo['HGT'].max())
    z_max = active_hgt.max(dim=['lat', 'lon']).fillna(ds_geo['HGT'].max())
    area = ds_geo['N_Points'].sum(dim=['lat', 'lon'])
    
    # Create DataFrame for Lookup
    ref_data = pd.DataFrame({
        'pdd_7': pdd_7.values,
        't_30': t_30.values,
        'p_30': p_30.values,
        'z_min': z_min.values,
        'z_max': z_max.values,
        'z_mean': z_mean.values,
        'area': area.values
    }, index=pd.to_datetime(ds_cl.time.values))
    
    # Filter to Target Dates
    ctx_subset = ref_data.reindex(target_dates, method='nearest')
    days_snow = [get_days_snow(d) for d in target_dates]
    
    # Time Features
    norm_year = (target_dates.year - 1990) / 32.0
    doy_sin = np.sin(2 * np.pi * target_dates.dayofyear / 365.0)
    doy_cos = np.cos(2 * np.pi * target_dates.dayofyear / 365.0)
    
    # Stack (Order MUST match training: Year, Sin, Cos, PDD7, T30, P30, DaysSnow, Area, Min, Max, Mean)
    X_ctx_raw = np.column_stack([
        norm_year, doy_sin, doy_cos,
        ctx_subset['pdd_7'].values,
        ctx_subset['t_30'].values,
        ctx_subset['p_30'].values,
        np.array(days_snow),
        ctx_subset['area'].values,
        ctx_subset['z_min'].values,
        ctx_subset['z_max'].values,
        ctx_subset['z_mean'].values
    ])
    
    # Scale Context
    if scaler_ctx:
        return scaler_ctx.transform(X_ctx_raw)
    else:
        return X_ctx_raw

# 4. Execute Context Generation
ds_cl = forcing[['N_Points','T2','SNOWFALL']].sortby('time')
ds_geo = forcing[['N_Points','HGT']].sortby('time')

print("Building Observation Contexts...")
X_ctx_sla = generate_context_matrix(tsla_obs.index, ds_cl, ds_geo, scalers_sla['context'])
X_ctx_alb = generate_context_matrix(pd.to_datetime(alb_obs_data.time.values), ds_cl, ds_geo, scalers_alb['context'])

# Free memory
del ds_cl, ds_geo

# ============ PyTensor Op =============

@as_op(itypes=[pt.dmatrix], otypes=[pt.dvector, pt.dvector, pt.dvector])
def run_emulators(param_stack):
    """
    Wrapper to call Keras models from PyMC.
    param_stack columns: [rrr, ice, snow, firn, aging, depth, rough, lwin, ws]
    """
    # 1. Scale Parameters (Using MB scaler as generic static scaler)
    # The scaler expects 9 features now.
    params_scaled = scalers_mb['static'].transform(param_stack)
    
    # --- Mass Balance ---
    mb_pred_scaled = model_mb.predict(params_scaled, verbose=0)
    mb_pred = scalers_mb['target_mb'].inverse_transform(mb_pred_scaled).flatten()
    
    # --- SLA ---
    n_obs_sla = X_ctx_sla.shape[0]
    p_sla = np.repeat(params_scaled, n_obs_sla, axis=0) 
    X_in_sla = [p_sla, X_ctx_sla]
    sla_pred = model_sla.predict(X_in_sla, verbose=0).flatten()
    
    # --- Albedo ---
    n_obs_alb = X_ctx_alb.shape[0]
    p_alb = np.repeat(params_scaled, n_obs_alb, axis=0)
    X_in_alb = [p_alb, X_ctx_alb]
    alb_pred = model_alb.predict(X_in_alb, verbose=0).flatten()
    
    return (
        np.array(mb_pred, dtype=np.float64),
        np.array(sla_pred, dtype=np.float64),
        np.array(alb_pred, dtype=np.float64)
    )

# ============ Main Chain Runner =============
if __name__ == "__main__":
    chain_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with open(path+"halji_initvals.pkl", "rb") as f:
        initvals = pickle.load(f)
    
    # Note: Ensure initvals includes the 2 new parameters (9 total)
    initval = initvals[chain_id]

    with pm.Model() as model:
        # --- Priors (9 Parameters) ---
        rrr = pm.TruncatedNormal('rrrfactor', mu=0.6619, sigma=0.06887, lower=0.57, upper=0.86)
        snow = pm.TruncatedNormal("albsnow", mu=0.889, sigma=0.05, lower=0.83, upper=0.925)
        ice = pm.TruncatedNormal("albice", mu=0.189, sigma=0.05, lower=0.13, upper=0.27)
        firn = pm.TruncatedNormal("albfirn", mu=0.58476, sigma=0.05, lower=0.46, upper=0.68)
        aging = pm.TruncatedNormal("albaging", mu=12.384, sigma=5.372, lower=3, upper=23)
        depth = pm.TruncatedNormal("albdepth", mu=6.2, sigma=3.2, lower=1.0, upper=12)
        rough = pm.TruncatedNormal("iceroughness", mu=9.652, sigma=5.37, lower=0.7, upper=19.5)
        lwin = pm.TruncatedNormal("lwinfactor", mu=1.02053, sigma=0.02205, lower=0.95011, upper=1.05)
        ws = pm.TruncatedNormal("wsfactor", mu=1.839, sigma=0.457, lower=0.75, upper=2.5)

        # Additional Error Priors
        sigma_alb_summer = pm.HalfNormal("sigma_alb_summer", sigma=0.02)
        sigma_tsl_summer = pm.HalfNormal("sigma_tsl_summer", sigma=0.03) 
        # --- Run Emulators ---
        # Stack 9 params
        param_values = pm.math.stack([rrr, ice, snow, firn, aging, depth, rough, lwin, ws], axis=0).reshape((1, 9))
        
        modmb, modtsl, modalb = run_emulators(param_values)

        mu_mb = pm.Deterministic('mu_mb', modmb)
        mu_tsl = pm.Deterministic('mu_tsl', modtsl)
        mu_alb = pm.Deterministic('mu_alb', modalb)

        # --- Observations ---
        # MB (Scalar)
        mb_obs_val = np.array([geod_ref['dmdtda'].item()])
        mb_sigma = np.array([geod_ref['err_dmdtda'].item()])
        
        # SLA (Vector)
        tsl_obs_val = np.array(tsla_obs['TSL_normalized'])
        tsl_sigma = np.array(tsla_obs['SC_norm']) 
        # Systematic Error Logic
        is_winter_tsl = tsla_obs['season'].values == "winter"
        tsl_sys_errors = pt.where(is_winter_tsl, 0.0, sigma_tsl_summer)
        sigma_tsl_total = pm.math.sqrt(tsl_sigma**2 + tsl_sys_errors**2)

        # Albedo (Vector)
        alb_obs_val = np.array(alb_obs_data['mean_albedo'].values)
        alb_obs_unc = np.array(alb_obs_data['sigma_albedo'].values)
        is_winter = alb_obs_data['season'].values == "winter"
        sigma_alb_total = pm.math.sqrt(alb_obs_unc**2 + pt.where(is_winter, 0, sigma_alb_summer)**2)
        
        # --- Likelihoods (Student-T) ---
        # 1. Mass Balance 
        obs_mb = pm.StudentT("mb_obs", nu=4, mu=mu_mb, sigma=mb_sigma, observed=mb_obs_val)
        # 2. Snowlines
        obs_tsl = pm.StudentT("tsl_obs", nu=4, mu=mu_tsl, sigma=sigma_tsl_total, observed=tsl_obs_val)
        # 3. Albedo
        obs_alb = pm.StudentT("alb_obs", nu=4, mu=mu_alb, sigma=sigma_alb_total, observed=alb_obs_val)

        # --- Balanced Potential ---
        # Calculate Log-Likelihoods per Data Stream
        ll_mb = pm.logp(obs_mb, mb_obs_val).sum()
        ll_tsl = pm.logp(obs_tsl, tsl_obs_val).sum()
        ll_alb = pm.logp(obs_alb, alb_obs_val).sum()
        
        # Normalize by Number of Observations (Mean Log-Like per point)
        # For MB, N=1. For others, N=len(vector).
        #n_mb = 1.0
        n_tsl = tsl_obs_val.shape[0]
        n_alb = alb_obs_val.shape[0]

        ll_mb_norm = ll_mb #/ n_mb
        ll_tsl_norm = ll_tsl / n_tsl
        ll_alb_norm = ll_alb / n_alb
        
        # Combined Potential
        total_loglike = (1.0 * ll_mb_norm) + (0.0*ll_tsl_norm) + (0.0*ll_alb_norm)
        
        pm.Deterministic("loglike_mb_norm", ll_mb_norm)
        #pm.Deterministic("loglike_tsl_norm", ll_tsl_norm)
        #pm.Deterministic("loglike_alb_norm", ll_alb_norm)
        
        pm.Potential("balanced_loglike", total_loglike)

        # --- Sampling ---
        #step = pm.DEMetropolisZ()
        step = pm.Slice()
        #step = pm.Metropolis()
        trace = pm.sample(
            draws=20000, 
            tune=2000, 
            chains=1, 
            cores=1, 
            initvals=[initval], 
            step=step, 
            discard_tuned_samples=True, 
            return_inferencedata=True, 
            progressbar=True
        )
        
        trace.to_netcdf(f"{outpath}/debug2_chain_{chain_id}.nc")
"""
# ... (Imports and Config remain the same) ...

# ... (Data Loading and generate_context_matrix remain the same) ...

# ============ DEBUGGING STEP 1: PARAMETER OPTIMIZATION =============
from scipy.optimize import minimize

if __name__ == "__main__":
    print("\n===============================================")
    print("DEBUG MODE: Checking Emulator Physical Limits")
    print("===============================================")

    # 1. Define the bounds based on your PyMC Priors
    # Order: [rrr, ice, snow, firn, aging, depth, rough, lwin, ws]
    # These match the TruncatedNormal limits in your script
    bounds = [
        (0.57, 0.86),  # rrr (Lower limit 0.648 might be the bottleneck!)
        (0.13, 0.27), # albice
        (0.83, 0.925),   # albsnow
        (0.46, 0.68),   # albfirn
        (3.0, 23.0),     # albaging
        (1.0, 12.0),       # albdepth
        (0.7, 19.5),     # rough
        (0.95011, 1.05),       # lwin
        (0.75, 2.5)        # ws
    ]
    
    # Starting guess (Mean of priors)
    x0 = [0.6619, 0.189, 0.889, 0.58476, 12.384, 6.2, 9.652, 1.02053, 1.839]

    # 2. Define the objective function
    def get_min_mb(x):
"""
#        Returns the Mass Balance for a given parameter set x.
"""
        # Reshape for scaler (1 sample, 9 features)
        x_reshaped = np.array(x).reshape(1, -1)
        
        # Scale inputs (Using the MB static scaler)
        try:
            params_scaled = scalers_mb['static'].transform(x_reshaped)
        except ValueError as e:
            print(f"Scaler Error: Expected {scalers_mb['static'].n_features_in_} features, got {len(x)}")
            raise e
            
        # Predict MB (Scaled)
        mb_pred_scaled = model_mb.predict(params_scaled, verbose=0)
        
        # Inverse transform to get real units (m w.e.)
        mb_pred = scalers_mb['target_mb'].inverse_transform(mb_pred_scaled).flatten()[0]
        
        return mb_pred

    # 3. Run Optimization (Find Minimum MB)
    print("\nSearching for minimum possible Mass Balance within priors...")
    res = minimize(get_min_mb, x0, bounds=bounds, method='L-BFGS-B')

    # 4. Report Results
    min_possible_mb = res.fun
    obs_mb = geod_ref['dmdtda'].item()
    
    print(f"\n--- RESULTS ---")
    print(f"Observed Mass Balance:       {obs_mb:.4f} m w.e.")
    print(f"Minimum Emulator Output:     {min_possible_mb:.4f} m w.e.")
    print(f"Difference (Gap):            {min_possible_mb - obs_mb:.4f}")
    
    print("\n--- Optimized Parameters to achieve this Minimum ---")
    param_names = ['rrr', 'ice', 'snow', 'firn', 'aging', 'depth', 'rough', 'lwin', 'ws']
    for name, val, bound in zip(param_names, res.x, bounds):
        hit_bound = ""
        if np.isclose(val, bound[0], atol=1e-3): hit_bound = "<-- HITTING LOWER BOUND"
        if np.isclose(val, bound[1], atol=1e-3): hit_bound = "<-- HITTING UPPER BOUND"
        print(f"{name:<10}: {val:.4f}  (Bounds: {bound}) {hit_bound}")

    if min_possible_mb > obs_mb:
        print("\n[CRITICAL]: The emulator CANNOT reach the observed mass balance.")
        print("Your priors are too restrictive. The parameters marked 'HITTING BOUND' need to be widened.")
    else:
        print("\n[SUCCESS]: The emulator IS capable of reproducing the observation.")
        print("The issue lies in the Albedo/SLA likelihoods pulling the model away from this solution.")

"""
