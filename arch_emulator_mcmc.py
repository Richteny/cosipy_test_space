import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import keras
import tensorflow as tf
from tensorflow.keras.layers import Layer #Input, Dense, Dropout, LSTM, TimeDistributed, Reshape, Conv1D, BatchNormalization, Add, GRU, Bidirectional, Layer, Concatenate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pytensor
pytensor.config.exception_verbosity = 'high'
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
import xarray as xr
import sys

import multiprocessing as mp
mp.set_start_method("forkserver", force=True)

path = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/"

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

## joint emulator snowline MB
model_full = tf.keras.models.load_model(path+ "first_test.keras")

# Print model summary to verify inputs/outputs
#model.summary()
# Extract only the mass balance branch
mass_balance_model = tf.keras.Model(
    inputs=model_full.input[0],  # Only take the mass balance input
    outputs=model_full.get_layer("mass_balance_output").output  # Extract the mass balance output
)

# Print the summary of the new model
mass_balance_model.summary()
######### --- SNOWLINE EMULATOR --- #########


@keras.saving.register_keras_serializable(package="CustomLosses")
def weighted_huber_loss(y_true, y_pred, delta=0.1):
    residual = tf.abs(y_true - y_pred)
    weights = tf.where(y_true > 0, 50.0, 1.0)
    weights = tf.cast(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights, axis=-1)
    huber_loss = tf.keras.losses.huber(y_true, y_pred, delta=delta)
    return tf.reduce_mean(weights * huber_loss)


@keras.saving.register_keras_serializable(package="CustomLayers")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, time_steps, features, **kwargs):
        super().__init__(**kwargs)  # This ensures 'trainable' and 'dtype' are properly handled
        self.time_steps = time_steps
        self.features = features

    def build(self, input_shape):
        pass  # If there's any weight initialization, it should be done here

    def call(self, inputs):
        position = np.arange(self.time_steps)[:, np.newaxis]  # Shape: (time_steps, 1)

        # Create div_term array with the exact length needed
        num_pairs = self.features // 2  # Number of (sin, cos) pairs
        div_term = np.exp(np.arange(num_pairs) * -(np.log(10000.0) / self.features))  # Shape: (num_pairs,)

        # Compute positional encodings
        pos_enc = np.zeros((self.time_steps, self.features))
        pos_enc[:, :num_pairs * 2:2] = np.sin(position * div_term)  # Assign sin values
        pos_enc[:, 1:num_pairs * 2:2] = np.cos(position * div_term)  # Assign cos values

        # If the number of features is odd, add an extra column of zeros
        if self.features % 2 != 0:
            pos_enc[:, -1] = 0  # Ensures proper alignment

        # Convert to TensorFlow tensor
        pos_enc_tensor = tf.convert_to_tensor(pos_enc, dtype=tf.float32)

        # Expand and tile across the batch dimension
        pos_enc_tensor = tf.expand_dims(pos_enc_tensor, axis=0)  # Shape: (1, time_steps, features)
        pos_enc_tensor = tf.tile(pos_enc_tensor, [tf.shape(inputs)[0], 1, 1])  # Shape: (batch_size, time_steps, features)

        return inputs + pos_enc_tensor
    
    def compute_output_shape(self, input_shape):
        return input_shape  # Output shape remains unchanged

    def get_config(self):
        config = super().get_config()
        config.update({
            "time_steps": self.time_steps,
            "features": self.features
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

custom_objects = {
    "PositionalEncoding": PositionalEncoding,
    "weighted_huber_loss": weighted_huber_loss
}

model_tsl = tf.keras.models.load_model(path + "current_best_model.keras", custom_objects=custom_objects)
model_tsl.summary()

### -- Prepare forcing data -- ###
# params = pd.read_csv(path+"cosipy_synthetic_params_lhs.csv", index_col=0)
params = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/simulations/LHS_1D20m_1999_2010_fullprior.csv", index_col=0)
sim_list = [x for x in params.columns if 'sim' in x]

include_mb = False

if include_mb:
    param_data = params.drop(['center_snow_transfer', 'spread_snow_transfer',
                          'roughness_firn', 'aging_factor_roughness', 'roughness_fresh_snow']+sim_list, axis=1)
else:
    param_data = params.drop(['mb','center_snow_transfer', 'spread_snow_transfer',
                            'roughness_firn', 'aging_factor_roughness', 'roughness_fresh_snow']+sim_list, axis=1)

## Prepare forcing for MB emulator
params_subset = param_data.copy()
print(params_subset)
scaler = StandardScaler()
scaler.fit(params_subset.values)

## Prepare forcing for TSL emulator
## Prepare forcing data to parse into emulator
# Load forcing data from NetCDF (fixed across simulations)
def spatial_mean(ds):
    weights = ds['N_Points'] / np.sum(ds['N_Points'])
    result = ds.copy()
    for var in list(ds.variables):
        if var not in ["time", "lat", "lon", "HGT", "MASK", "SLOPE", "ASPECT", "N_Points"]:
            weighted_field = (ds[var] * weights).sum(dim=['lat', 'lon'])
            result[var] = weighted_field
    return result

def spatial_max(ds):
    result = ds.copy()
    for var in list(ds.variables):
        if var not in ["time", "lat", "lon", "HGT", "MASK", "SLOPE", "ASPECT", "N_Points"]:
            result[var] = ds[var].max(dim=['lat', 'lon'])  # Compute max over spatial dimensions
    return result

def spatial_min(ds):
    result = ds.copy()
    for var in list(ds.variables):
        if var not in ["time", "lat", "lon", "HGT", "MASK", "SLOPE", "ASPECT", "N_Points"]:
            result[var] = ds[var].min(dim=['lat', 'lon'])  # Compute max over spatial dimensions
    return result

def load_forcing_data(nc_file="/data/scratch/richteny/thesis/cosipy_test_space/data/input/HEF/HEF_COSMO_1D20m_HORAYZON_1999_2010_IntpPRES.nc"):
    ds = xr.open_dataset(nc_file).sel(time=slice(None,"2009-12-31T23:00:00"))
    ds = ds[['T2','SNOWFALL','G','U2','RH2','N','N_Points', 'HGT']] #'RH2','U2','G','PRES','RRR','SNOWFALL','N']]
    
    #engineere true snowfall as in COSIPY 
    minimum_snowfall = 0.001  
    ds['SNOWFALL'] = (('time','lat','lon'), np.where(ds['SNOWFALL'].values > minimum_snowfall, 0, ds['SNOWFALL'].values))
    # build avg. over area
    ds_mean = spatial_mean(ds)[['T2','SNOWFALL','G','HGT']]
    ds_max = spatial_max(ds)[['T2','SNOWFALL','G']]
    ds_max = ds_max.rename({'T2': 'spatmaxT2', 'SNOWFALL': 'spatmaxSNOWFALL', 'G': 'spatmaxG'})
    
    #merge
    ds_mean = xr.merge([ds_mean, ds_max])
    #spatial max and min as predictors?
    # snowfall only occurs if cosipy has > 0.01m snowfall
    
    #forcing_data = ds.to_array().transpose("time", "lat", "lon", "variable")
    return ds_mean #forcing_data.values  # Shape: (timesteps, lat, lon, features)


forcing = load_forcing_data()
print(forcing)
#static = forcing['HGT']
#min_glacier_elev = np.nanmin(static.data)
#max_glacier_elev = np.nanmax(static.data)
#print(min_glacier_elev, max_glacier_elev)

#create PDDs
threshold = 273.15  # you could use 1 if needed
# Calculate the positive degree days (PDD) for each time step
pdd = (forcing[['T2']] - threshold).where(forcing[['T2']] > threshold, 0) #when true, keep values else 0
pdd = pdd.rename({'T2': 'PDD'})
# Sum PDD over time
pdd_monthly = pdd.resample(time="14D").sum()

# create subfields and merge
snowfall = forcing[['SNOWFALL','spatmaxSNOWFALL']].resample(time="14D").sum() #sum for snowfall
temp = forcing[['T2','G']].resample(time="14D").mean() #sum for snowfall
#max_temp = forcing[['spatmaxT2','spatmaxG']].resample(time="1ME").mean()
max_temp = forcing[['T2','G']].resample(time="14D").max()
max_temp = max_temp.rename({'T2': 'T2max', 'G': 'Gmax'})

comb_forc = xr.merge([snowfall, temp, max_temp, pdd_monthly])
print(comb_forc)

# Create lagged features (previous 1, 2, and 3 months)
lagged_features = []
for lag in range(1, 4):  # 1 to 3 months
    ds_lagged = comb_forc[['T2','SNOWFALL','G',]].shift(time=lag)
    ds_lagged = ds_lagged.rename({var: f"{var}_lag{lag}" for var in ['T2','SNOWFALL','G']})  # Rename vars
    lagged_features.append(ds_lagged)

# Merge original dataset with lagged features
ds_with_lags = xr.merge([comb_forc] + lagged_features)
# Drop NaN values introduced by shifting
ds_with_lags = ds_with_lags.dropna(dim="time", how="any")  # Ensure all time steps are valid

print("Prepared forcing data just like in emulator creation script.")

 # Convert forcing data to NumPy array
time_series_data = ds_with_lags.to_array().transpose("time", "variable").values  # (time, features)

# Normalize forcing data
# Normalize the data using Z-score (standardization)
scaler1 = MinMaxScaler()
normalized_data = scaler1.fit_transform(time_series_data)
normalized_data_repeated = np.repeat(normalized_data[np.newaxis, :, :], 1, axis=0)
print("Repeated TS data", normalized_data_repeated.shape)
print("Normalized and prepared forcing data just like in emulator creation script.")

n_time_steps = normalized_data.shape[0]
print("N. time steps:", n_time_steps)

static_params = param_data.values
print(static_params.shape)
scaler2 = MinMaxScaler()
#norm_static_params = scaler2.fit_transform(static_params)
scaler2.fit(static_params)  # Fit on training data with 3000 samples

print("Created scaler just like in emulator creation script.")

#
idx_month = ds_with_lags.time.dt.month[0].item()-1
# Create time-based features for 120 months
first_year = np.arange(idx_month, 12)
first_year_vals = np.repeat(0,12-idx_month) 

month_raw = np.concatenate((first_year, np.arange(12, len(ds_with_lags.time)+idx_month)), axis=0) #% 12  # Month index (0-11)
years = month_raw // 12   # Year index (0-9, assuming 10 years)
months = month_raw % 12

print(months)
print(years)

# Sinusoidal encoding for seasonality
month_sin = np.sin(2 * np.pi * months / 12)
month_cos = np.cos(2 * np.pi * months / 12)

# Stack time-based features
time_features = np.column_stack((months / 12, years / 10, month_sin, month_cos))
#time_features = np.column_stack((month_sin, month_cos))
print(time_features.shape)

# Repeat for all parameter combinations
# Ensure correct tiling: repeat for each sample in X_train
num_samples = 1 # because for each parameter we just have one sample X.shape[0]
time_features = np.tile(time_features, (num_samples, 1, 1))  # Shape (num_samples, timesteps, 4)
print(time_features.shape)
# Concatenate time features with input features
#X = np.concatenate([X, time_features], axis=-1)
#print(X.shape)

# Your observational timestamps (e.g., pandas DateTimeIndex)
observational_times = tsl.loc["2000-01-01":"2010-01-01"].index

predicted_times = pd.to_datetime(ds_with_lags.indexes["time"])
predicted_times

# Find the closest indices
# `get_indexer` gives you the closest match of observational times to predicted times
closest_indices = [predicted_times.get_indexer([obs_time], method="nearest")[0] for obs_time in observational_times]
closest_indices

def tsl_emulator_input(param_values):
    """
    Prepares the forcing data for the TSL emulator.
    
    param_values: Array of sampled parameter values from PyMC.
    """
    

    # Concatenate along the last axis (column-wise)
    ## sort param values? order needs to be:
    # rrr_factor alb_ice alb_snow alb_firn albedo_aging albedo_depth roughness_ice   
    # Normalize parameter values (using same scaler from preprocessing)
    norm_params = scaler2.transform(param_values.reshape(1, -1))  # Ensure same transformation as before

    # Add constant snow height
    norm_params = np.hstack([norm_params, np.array([[0.689387]])])  #init snowheight

    # Reshape the static parameters to match the time series shape
    static_params_repeated = np.repeat(norm_params[:, np.newaxis, :], n_time_steps, axis=1)  
    #print("repeated params", static_params_repeated.shape)

    # Stack static parameters with time-dependent features
    X_1 = np.concatenate([static_params_repeated, normalized_data_repeated], axis=-1)
    
    # Concatenate time features with input features
    X = np.concatenate([X_1, time_features], axis=-1)

    return X  # (1, time, features)

## Create function to run emulators outside
@as_op(itypes=[pt.dmatrix], otypes=[pt.dvector, pt.dvector])
def run_emulators(param_stack):
    #param_values_numpy = param_stack.eval().astype(np.float32)
    X_mb = scaler.transform(param_stack)
    mod_mb = mass_balance_model.predict(X_mb, verbose=0).flatten()
    
    # Prepare TSL data
    X_tsl = tsl_emulator_input(param_stack)
    mod_tsl = model_tsl.predict(X_tsl, verbose=0).flatten()[closest_indices]
    #mod_tsl = np.array([closest_indices])
    
    mod_mb = np.array(mod_mb, dtype=np.float64)
    mod_tsl = np.array(mod_tsl, dtype=np.float64)

    
    return (mod_mb, mod_tsl)

with pm.Model() as model:
    rrr_factor = pm.TruncatedNormal('rrrfactor', mu=0.874, sigma=0.182, lower=0.52, upper=1.433)
    #alb_snow = pm.TruncatedNormal("albsnow", mu=0.907, sigma=0.1, lower=0.887, upper=0.93)
    #alb_ice = pm.TruncatedNormal("albice", mu=0.173, sigma=0.1, lower=0.115, upper=0.233)
    #alb_firn = pm.TruncatedNormal("albfirn", mu=0.593, sigma=0.1, lower=0.506, upper=0.692)
    #alb_aging = pm.TruncatedNormal("albaging", mu=11.15, sigma=9, lower=1, upper=25)
    alb_depth = pm.TruncatedNormal("albdepth", mu=3, sigma=1.31, lower=1, upper=10)
    #roughness_ice = pm.TruncatedNormal("iceroughness", mu=9.875, sigma=9, lower=0.94, upper=20)
    
    # Define fixed values for the other parameters
    #rrr_factor = pt.constant(0.88)
    alb_snow = pt.constant(0.907)  # Mean of original distribution
    alb_ice = pt.constant(0.173)
    alb_firn = pt.constant(0.6)
    alb_aging = pt.constant(8)  # 6+3 from your original code
    roughness_ice = pt.constant(1.7)
    #alb_depth = pt.constant(8)

    #Setup observations
    geod_data = pm.Data('geod_data', np.array([geod_ref['dmdtda']]))
    tsl_data = pm.Data('tsl_data', np.array(tsla_obs['TSL_normalized']))
    
    #created all parameters, now when we sample - need to combine all parameters to pass to emulator input function
    param_values = pm.math.stack([rrr_factor, alb_ice, alb_snow, alb_firn, alb_aging, alb_depth, roughness_ice], axis=0)
    param_values = param_values.reshape((1, 7))  # Ensure it's in the shape (1, 7)
    
    #param_values_numpy = param_values.eval().astype(np.float32)  # Convert to NumPy array

    modmb, modtsl = run_emulators(param_values)

    mu_mb = pm.Deterministic('mu_mb', modmb)
    mu_tsl = pm.Deterministic('mu_tsl', modtsl)
    
    # Likelihood definitions
    mb_obs = pm.Normal("mb_obs", mu=mu_mb, sigma=geod_ref['err_dmdtda'], observed=geod_data)
    tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=np.array(tsla_obs['SC_norm']), observed=tsl_data, shape=mu_tsl.shape[0])
    #tsl_obs = pm.Normal("tsl_obs", mu=mu_tsl, sigma=np.array([0.023697]), observed=tsl_data, shape=mu_tsl.shape[0])
    # Manually compute log-likelihoods
    loglike_mb = pm.logp(mb_obs, geod_data)  # Mass balance log-likelihood
    loglike_tsl = pm.logp(tsl_obs, tsl_data).mean() # Snowline log-likelihood (vector)

    # Store individual log-likelihoods
    pm.Deterministic("loglike_mb", loglike_mb)
    pm.Deterministic("loglike_tsl", loglike_tsl)  # Sum over all snowline data points

    # Compute total log-likelihood
    total_loglike = mb_obs + loglike_tsl # Averaging for balance, adjust scaling to snowlines. Without, MB was outside uncertainty 
    #total_loglike = pm.math.sum(loglike_tsl)
    pm.Deterministic("total_loglike", total_loglike)

    # Ensure PyMC uses this likelihood for inference
    pm.Potential("balanced_loglike", total_loglike)
    
    step = pm.DEMetropolisZ()
    #step = pm.DEMetropolis()
    #step = pm.Metropolis()
    initvals = [{'rrrfactor': 0.55, 'albdepth': 1.2}, {'rrrfactor': 1.3,  'albdepth': 8},
                {'rrrfactor': 0.9,  'albdepth': 6},{'rrrfactor': 0.7,  'albdepth': 5}]

    post = pm.sample(draws=10000, tune=1000, step=step, initvals=initvals, return_inferencedata=True, chains=4, cores=1, progressbar=False) #discard_tuned_samples=False)
    post.to_netcdf("/data/scratch/richteny/thesis/cosipy_test_space/simulations/emulator/simulations_HEF_emulator_MCMC_big.nc")
    
