import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, LSTM, TimeDistributed, Reshape, Conv1D, BatchNormalization, Add, GRU, Bidirectional, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xarray as xr
import pathlib

import sys

path = "/data/scratch/richteny/thesis/cosipy_test_space/simulations/"
tsla = pd.read_csv("/data/scratch/richteny/thesis/cosipy_test_space/data/input/HEF/snowlines/HEF-snowlines-1999-2010_manual.csv")
snowline_path = "/data/scratch/richteny/thesis/io/data/output/nn_data/full_snowlines/"
    
# params = pd.read_csv(path+"cosipy_synthetic_params_lhs.csv", index_col=0)
params = pd.read_csv(path+"LHS_1D20m_1999_2010_fullprior.csv", index_col=0)
sim_list = [x for x in params.columns if 'sim' in x]
param_data = params.drop(['mb','center_snow_transfer', 'spread_snow_transfer',
                          'roughness_firn', 'aging_factor_roughness', 'roughness_fresh_snow', 'roughness_ice']+sim_list, axis=1)

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

if 'win' in sys.platform:
    forcing = load_forcing_data(nc_file="E:/OneDrive - uibk.ac.at/PhD/cosipy_test_space/data/input/HEF/HEF_COSMO_1D20m_HORAYZON_IntpolPRES_1999_2010_old.nc")
else:
    forcing = load_forcing_data()
print(forcing)
static = forcing['HGT']
min_glacier_elev = np.nanmin(static.data)
max_glacier_elev = np.nanmax(static.data)
print(min_glacier_elev, max_glacier_elev)

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
ds_with_lags

## Create input features
# Extracting the time-constant parameters (these do not change over time)
# Extracting the time-constant parameters (these do not change over time)
static_params = param_data.values
print(static_params)
scaler2 = MinMaxScaler()
norm_static_params = scaler2.fit_transform(static_params)
print(norm_static_params)
print(norm_static_params.shape)

# Create a column of the same constant value
init_snowheight_col = np.full((norm_static_params.shape[0], 1), 0.689387)
# Concatenate along the last axis (column-wise)
norm_static_params = np.hstack((norm_static_params, init_snowheight_col))
print(norm_static_params.shape)

# Combine the temperature and snowfall time series into a single data array for easier handling
time_series_data = ds_with_lags.to_array().transpose("time", "variable").values #np.column_stack((temp['T2'].values, temp['G'].values, snowfall.values)) 
print("TS data", time_series_data.shape)

# Normalize the data using Z-score (standardization)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(time_series_data)

# Reshape the data for LSTM input (samples, time steps, features)
n_time_steps = normalized_data.shape[0]  # Number of timesteps (days)

# Reshape the static parameters to match the time series shape
static_params_repeated = np.repeat(norm_static_params[:, np.newaxis, :], n_time_steps, axis=1)  # Shape: (150, 3654, 6)
print("repeated params", static_params_repeated.shape)

normalized_data_repeated = np.repeat(normalized_data[np.newaxis, :, :], static_params_repeated.shape[0], axis=0)
print("Repeated TS data", normalized_data_repeated.shape )

"""
def add_lagged_climate_features(X, lags=3):
    """"""
    Adds lagged versions of climate features (temperature, snowfall) to X.

    Parameters:
    - X: Input array (samples, timesteps, features)
    - lags: Number of past timesteps to include as features

    Returns:
    - X_lagged: Updated X with additional lagged features
    """"""
    samples, timesteps, features = X.shape
    X_lagged = np.copy(X)  # Copy original features

    # Extract climate variables (assuming temp & snowfall are the first 2 features)
    climate_vars = X[:, :, :2]  # Adjust this based on your dataset

    # Create an array for lagged features
    climate_lagged = np.zeros((samples, timesteps, 2 * lags))

    for lag in range(1, lags + 1):
        climate_lagged[:, lag:, (lag - 1) * 2] = climate_vars[:, :-lag, 0]  # Temp lag
        climate_lagged[:, lag:, (lag - 1) * 2 + 1] = climate_vars[:, :-lag, 1]  # Snowfall lag

    # Concatenate lagged climate variables to X
    X_lagged = np.concatenate([X_lagged, climate_lagged], axis=-1)

    return X_lagged

# Apply function to your dataset
X = add_lagged_climate_features(normalized_data_repeated, lags=3)
print(X.shape)
"""
# Concatenate the static parameters and the time series data
X = np.concatenate([static_params_repeated, normalized_data_repeated], axis=-1)

print(X.shape) #[samples, time steps, features]

# Initialize a list to store snowline altitudes for each parameter combination
y_all = []

# Iterate over each row of the parameter dataframe and load the corresponding CSV
for idx, row in param_data.round(4).iterrows():
    # Construct the filename for the corresponding snowline CSV (based on parameter combination)
    #tsla_hef_cosmo_1d20m_1999_2010_horayzon_lhs_19990101-20091231_rrr-0.72_0.9025_0.3125_0.6448_27.1461_30.121_0.24_1.7_4.0_0.0026_num2.csv
    snowline_csv = (
        snowline_path + 
        "tsla_hef_cosmo_1d20m_1999_2010_horayzon_lhs_19990101-20091231_rrr-" +
        f"{row['rrr_factor']}_{row['alb_snow']}_{row['alb_ice']}_{row['alb_firn']}_{row['albedo_aging']}_{row['albedo_depth']}" +
        "_0.24_1.7_4.0_0.0026_num2.csv"
    )
    
    snowline_data = pd.read_csv(snowline_csv, parse_dates=True, index_col="time")[['Med_TSL']].resample("14D").max()
    snowline_data = snowline_data.loc[snowline_data.index.isin(pd.to_datetime(ds_with_lags.time))]
    snowline_vals = snowline_data["Med_TSL"].values
    y_all.append(snowline_vals)


# Stack all snowline altitudes for training (if there are multiple combinations)
y_all = np.vstack(y_all)
print(y_all.shape)
print(X.shape)

normalized_tsl = True
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
#print(time_features.shape)

doy = snowline_data.index.dayofyear
# Define time features
time_sin = np.sin(2 * np.pi * doy / 365)
time_cos = np.cos(2 * np.pi * doy / 365)
time_features = np.stack([time_sin, time_cos], axis=-1)  # Shape: (62, 2)
time_features

# Repeat for all parameter combinations
# Ensure correct tiling: repeat for each sample in X_train
num_samples = X.shape[0]
time_features = np.tile(time_features, (num_samples, 1, 1))  # Shape (num_samples, num_time, num_features)
print(time_features.shape)
# Concatenate time features with input features
X = np.concatenate([X, time_features], axis=-1)

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
if normalized_tsl:
    y_train, y_test = y_all[:train_size], y_all[train_size:]
else:
    y_train, y_test = y_all_og[:train_size], y_all_og[train_size:]
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

### Check means of e.g. prec. factor to confirm that train data selection works okay
train_rrr = param_data.values[:train_size][:,0]
test_rrr = param_data.values[train_size:][:,0]

print(np.nanmean(train_rrr)) #means look similar so it might be okay
print(np.nanmean(test_rrr))

class PositionalEncoding(Layer):
    """ Positional Encoding for time-series data (like in Transformers). """
    def __init__(self, time_steps, features):
        super(PositionalEncoding, self).__init__()
        self.time_steps = time_steps
        self.features = features

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

class CalibrationLayer(Layer):
    """ Linear transformation y = a*x + b to fine-tune predictions. """
    def __init__(self):
        super(CalibrationLayer, self).__init__()
        self.a = self.add_weight(name="scale", shape=(1,), initializer="ones", trainable=True)
        self.b = self.add_weight(name="bias", shape=(1,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return self.a * inputs + self.b

def build_model(input_shape):
    """
    Hybrid CNN-RNN Model for Snowline Prediction with:
    - Positional Encoding for temporal awareness.
    - Bidirectional LSTMs for sequence learning.
    - CNN layers for feature extraction.
    - Calibration Layer for systematic error correction.
    """
    #
    time_steps, features = input_shape

    # Input Layer
    snowlines_input = Input(shape=input_shape, name="snowlines_input")
    print(time_steps, features)
    # Apply Positional Encoding, new addition that does not seem to make it worse
    encoded_input = PositionalEncoding(time_steps, features)(snowlines_input)
    
    ### - this snippet is the current test, per suggestion from Codrut and https://arxiv.org/pdf/1505.04597 - ###    
    # LSTM Layers (Stacked)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(encoded_input)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(lstm_out)

    # Multi-Scale CNN (Inception Style)
    conv1 = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(lstm_out) #3 months, 6 months, 12 months
    conv2 = Conv1D(filters=64, kernel_size=6, padding="same", activation="relu")(lstm_out)
    conv3 = Conv1D(filters=64, kernel_size=12, padding="same", activation="relu")(lstm_out)

    # Combine Multi-Scale Features
    x = Concatenate()([conv1, conv2, conv3])
    x = BatchNormalization()(x)
    
    ### - this part was my old implementation that worked quite well already - ###
    # Build Snowline Layers
    #shared_sl = Bidirectional(LSTM(64, return_sequences=True))(encoded_input) #changing normal LSTM to Bidirectional small improvement in performance
    ##shared_sl = Bidirectional(LSTM(64, return_sequences=True))(shared_sl) #two LSTMs - check if does anything.
    
    ## GRU Layer for Further Processing -- didn't seem to improve performance
    ##shared_sl = GRU(64, return_sequences=True)(shared_sl)
    ##shared_sl = LSTM(64, return_sequences=True)(snowlines_input)
    #shared_sl = Dense(128, activation='relu')(shared_sl)
    
    # Pass into CNN Layer
    #x1 = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(shared_sl)
    #x2 = Conv1D(filters=64, kernel_size=12, padding="same", activation="relu")(shared_sl)
    #x = Add()([x1, x2])  # Combine features from different scales
    #x = BatchNormalization()(x)
    ##x = Dropout(0.2)(x) #dropout to prevent overfitting - we "want" it to overfit
    ### - old part end - ###
    # Pass back into normal layer
    sl_branch = Dense(64, activation='relu')(x)
    #sl_branch = Dropout(0.1)(sl_branch)  # Add Dropout here for snowlines branch
    
    # Output layers
    if normalized_tsl:
        print("Using sigmoid due to normalized values in range 0 to 1.") ## sigmoid does best
        snowlines_output = Dense(1, activation='sigmoid')(sl_branch)  # Predict one value per time point
    else:
        snowlines_output = Dense(1, activation='relu')(sl_branch)  # Predict one value per time point
    snowlines_output = Reshape((time_steps,), name="snowlines_output")(snowlines_output)  # Adjust shape to (batch_size, 62)
    
    # Apply Calibration Layer ## ---> calibration layer does not change quality of prediction much. Predictions seems quite insensitive.
    #calibrated_output = CalibrationLayer()(snowlines_output)


    model = Model(snowlines_input, snowlines_output, name="Snowline_CNN-RNN_Predictor")

    # Learning Rate Scheduling (Exponential Decay)
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=0.001, decay_steps=500, decay_rate=0.9, staircase=True
    #)

    def weighted_huber_loss(delta=0.1):
        def loss_fn(y_true, y_pred):
            residual = tf.abs(y_true - y_pred)
            # Instead of over-penalizing 0, we upweight nonzero values
            weights = tf.where(y_true > 0, 50.0, 1.0)  # Give more weight to nonzero values ##!! LATEST 50!! 
            weights = tf.cast(weights, dtype=tf.float32)  # Ensure correct type
            # Expand weights to match huber_loss dimensions (batch_size, time_steps)
            weights = tf.expand_dims(weights, axis=-1)
            
            huber_loss = tf.keras.losses.huber(y_true, y_pred, delta=delta)
            return tf.reduce_mean(weights * huber_loss)
        return loss_fn

    # Compile with new loss
    #model.compile(
    #    optimizer=Adam(learning_rate=0.001),
    #    loss=weighted_huber_loss(delta=0.1),
    #    metrics=[keras.metrics.RootMeanSquaredError()]
    #)

    # Compile Model with Adam Optimizer & Huber Loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=weighted_huber_loss(delta=0.1),  # Huber Loss is more robust
        metrics=[keras.metrics.RootMeanSquaredError()]  
    )

    return model

# Example: Define the input shape
timesteps = X_train.shape[1]  # Monthly data (120 timesteps)
features = X_train.shape[2]  # Climate vars, static params, and temporal encoding

model = build_model((timesteps, features))
model.summary()

# Early stopping: Stop training if validation loss doesn't improve for 10 epochs
early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
#lr_scheduler
# Reduce learning rate when validation loss plateaus
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200, batch_size=32,  # Try batch sizes 16, 32, or 64
    validation_data=(X_test, y_test),
    callbacks=[early_stopping,lr_scheduler],
    verbose=2
)

## Create evaluation figures
# Predict using the trained model
predictions = model.predict(X_test)
predictions.shape

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(path+"train_val_loss_nn.png", bbox_inches="tight")

if normalized_tsl:

    # Compute point density
    xy = np.vstack([y_test.flatten(), predictions.flatten()])
    density = gaussian_kde(xy)(xy)

    # Sort the points by density to ensure denser points appear on top
    idx = density.argsort()
    x_sorted, y_sorted, density_sorted = y_test.flatten()[idx], predictions.flatten()[idx], density[idx]

    # Create the plot
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_title("Model Evaluation", fontsize=17)
    ax.set_ylabel('Modeled Norm. TSLA', fontsize=16)
    ax.set_xlabel('Reference Norm. TSLA', fontsize=16)
    lineStart = 0.0
    lineEnd = 1.0
    ax.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    ax.set_xlim(lineStart, lineEnd)
    ax.set_ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)

    # Compute error metrics
    mae_score = mean_absolute_error(y_test.flatten(), predictions.flatten())
    r2_scores = r2_score(y_test.flatten(), predictions.flatten())
    mae_in_meter = mae_score * (max_glacier_elev - min_glacier_elev)

    textstr = '\n'.join((
        r'$MAE=%.4f$' % (mae_score, ),
        r'$MAE (m)=%.4f$' % (mae_in_meter, ),
        r'$R^2=%.4f$' % (r2_scores, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # Secondary axes
    def custom_ticks(y):
        orig_max = max_glacier_elev
        orig_min = min_glacier_elev
        return y * (orig_max - orig_min) + orig_min

    ax2y = ax.secondary_yaxis(-0.14, functions=(custom_ticks, custom_ticks))
    ax2x = ax.secondary_xaxis(-0.1, functions=(custom_ticks, custom_ticks))
    list_labels = [custom_ticks(x) for x in np.arange(0,1+0.1,0.1)]
    ax2x.set_xticks(list_labels)
    ax2y.set_yticks(list_labels)
    ax2x.set_xticklabels([round(x) for x in ax2x.get_xticks()], rotation=30)

    # Scatter plot with density coloring
    sc = ax.scatter(x_sorted, y_sorted, c=density_sorted, s=20, cmap='plasma', alpha=0.7)
    #cb = plt.colorbar(sc, ax=ax, label="Density")

    ax.set_xticks(np.arange(0,1+0.1,0.1))
    ax.set_yticks(np.arange(0,1+0.1,0.1))
    ax.grid(True)
else:
    # Plot predicted vs actual snowline altitudes
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    ax.set_title("Model Evaluation", fontsize=17)
    ax.set_ylabel('Modeled TSLA', fontsize=16)
    ax.set_xlabel('Reference TSLA', fontsize=16)

    ax.plot([0, 1], [0, 1], 'k-', transform=ax.transAxes)
    #plt.axvline(0.0, ls='-.', c='k')
    #plt.axhline(0.0, ls='-.', c='k')
    #ax.set_xlim(glacier, lineEnd)
    #ax.set_ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)

    mae_score = mean_absolute_error(y_test.flatten(), predictions.flatten())
    r2_scores = r2_score(y_test.flatten(), predictions.flatten())

    textstr = '\n'.join((
        r'$MAE=%.4f$' % (mae_score, ),
        r'$R^2=%.4f$' % (r2_scores, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # Add a secondary x-axis below the primary x-axis
    def custom_ticks(y):
        #convert back to original values
        orig_max = max_glacier_elev
        orig_min = min_glacier_elev
        return (y - orig_min) / (orig_max - orig_min)  # Define your custom transformation for tick labels

    ax2y = ax.secondary_yaxis(-0.14, functions=(custom_ticks, custom_ticks))
    ax2x = ax.secondary_xaxis(-0.1, functions=(custom_ticks, custom_ticks))
    list_labels = [custom_ticks(x) for x in np.arange(min_glacier_elev,max_glacier_elev+50,50)]
    ax2x.set_xticks(list_labels)
    ax2y.set_yticks(list_labels)
    ax2x.set_xticklabels([round(x) for x in ax2x.get_xticks()], rotation=30)
    #ax2x.set_xlabel(var)

    sc = ax.scatter(y_test, predictions, s=20, alpha=0.5)
    #ax.set_xticks(np.arange(0,1+0.1,0.1))
    #ax.set_yticks(np.arange(0,1+0.1,0.1))
    ax.grid(True)
plt.savefig(path+"scatterplot_pred_vs_test_nn.png", bbox_inches="tight")

## Calculate residuals
# Calculate residuals
residuals = y_test.reshape(y_test.shape[0],y_test.shape[1]) - predictions
print(np.nanmean(abs(residuals)))
print(np.nanmax(abs(residuals)))

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_test.reshape(y_test.shape[0],y_test.shape[1]), residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
plt.title('Residual Plot')
plt.xlabel('Actual Snowline Altitude')
plt.ylabel('Residual (Prediction Error)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig(path+"residuals_test-mod_nn.png", bbox_inches="tight")

## Find worst performing model
sample_idx = np.argmax(np.nanmean(np.abs(residuals), axis=1))
fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(ds_with_lags.time, y_test[sample_idx].flatten(), label="True Snowline")
ax.plot(ds_with_lags.time, predictions[sample_idx].flatten(), label="Predicted Snowline", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Normalized Snowline Altitude")
plt.title("Predicted vs. True Snowline (Test Sample)")
plt.legend()
plt.show()
plt.savefig(path+"timeseries_worst_mod_nn.png", bbox_inches="tight")

### Finally save the model
model.save(path+'/current_model_dailytsla.keras')
