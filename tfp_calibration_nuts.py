import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import arviz as az
import joblib
import os
import sys

# ================= 1. CONFIGURATION =================
NUM_CHAINS = 20          
PILOT_DRAWS = 2000       
PROD_DRAWS = 100000      # We save ALL draws now (unthinned)
BURNIN = 10000           
# THINNING is removed -> We handle this in ArviZ later
SAVE_PATH = "emulator_hpc_results.nc" # <--- Changed to .nc
path_to_geodetic = "/data/scratch/richteny/Hugonnet_21_MB/dh_15_rgi60_pergla_rates.csv"
# --- UPDATED PATH HERE ---
MODEL_DIR = "/data/scratch/richteny/for_emulator/Halji"
# -------------------------

#geod data
geod_ref = pd.read_csv(path_to_geodetic)
geod_ref = geod_ref[(geod_ref['rgiid'] == "RGI60-15.06065") & (geod_ref['period'] == "2000-01-01_2020-01-01")]

tfd = tfp.distributions

print(f"--- HPC CALIBRATION SCRIPT (NUTS + ArviZ) ---")
print(f"TF Version: {tf.__version__}")
print(f"Physical Devices: {tf.config.list_physical_devices()}")

# ================= 2. LOAD MODEL & DATA =================
try:
    print(f"Loading Model and Scalers from {MODEL_DIR}...")
    
    scalers_path = os.path.join(MODEL_DIR, "scalers_mb.pkl")
    scalers_mb = joblib.load(scalers_path)
    
    p_min = tf.constant(scalers_mb['static'].data_min_, dtype=tf.float32)
    p_max = tf.constant(scalers_mb['static'].data_max_, dtype=tf.float32)
    p_range = p_max - p_min

    t_mean = tf.constant(scalers_mb['target_mb'].mean_, dtype=tf.float32)
    t_scale = tf.constant(scalers_mb['target_mb'].scale_, dtype=tf.float32)

    model_path = os.path.join(MODEL_DIR, "model_mb.keras")
    model_mb = tf.keras.models.load_model(model_path)
    
    # Observation for Halji
    obs_mb = tf.constant([geod_ref['dmdtda'].item()], dtype=tf.float32) 
    sigma_mb = tf.constant([geod_ref['err_dmdtda'].item()], dtype=tf.float32)   
    
    print("Model loaded successfully.")

except Exception as e:
    print(f"\n[CRITICAL ERROR] Could not load model/scalers: {e}")
    sys.exit(1)

# ================= 3. DEFINE PRIORS =================
prior_dists = [
    tfd.TruncatedNormal(loc=0.59, scale=0.0662, low=0.44, high=0.73),   # rrr
    tfd.TruncatedNormal(loc=0.1521, scale=0.0286, low=0.11, high=0.21),       # ice
    tfd.TruncatedNormal(loc=0.866, scale=0.0245, low=0.816, high=0.915),      # snow
    tfd.TruncatedNormal(loc=0.6044, scale=0.066, low=0.46, high=0.69),     # firn
    tfd.TruncatedNormal(loc=14.95, scale=4.67, low=4.2, high=23.0),      # aging
    tfd.TruncatedNormal(loc=2.78, scale=0.86, low=1.135, high=5.75),           # depth
    tfd.TruncatedNormal(loc=9.75, scale=5.58, low=0.6, high=19.67),        # rough
    tfd.TruncatedNormal(loc=1.0273, scale=0.0142, low=0.99, high=1.05),   # lwin
    tfd.TruncatedNormal(loc=1.41, scale=0.33, low=0.75, high=2.0),       # ws
    tfd.TruncatedNormal(loc=-0.028, scale=0.5534, low=-1.0, high=1.0)           # phase
]
prior = tfd.JointDistributionSequential(prior_dists)

# ================= 4. DEFINE LOG POSTERIOR (XLA COMPILED) =================
@tf.function(jit_compile=True)
def log_posterior_batch(*params_list):
    params_vec = tf.stack(params_list, axis=-1)
    lp_prior_parts = prior.log_prob(params_list) 
    lp_prior = tf.reduce_sum(lp_prior_parts, axis=0) #sum across 10 params
    inputs_scaled = (params_vec - p_min) / p_range
    mb_scaled = model_mb(inputs_scaled)
    mb_pred = (mb_scaled * t_scale) + t_mean
    
    #likelihood_dist = tfd.StudentT(df=4, loc=mb_pred, scale=sigma_mb)
    likelihood_dist = tfd.Normal(loc=mb_pred, scale=sigma_mb)
    lp_likelihood = likelihood_dist.log_prob(obs_mb)
    
    return lp_prior + tf.squeeze(lp_likelihood)

# ================= 5. PHASE 1: PILOT RUN =================
print("\n--- PHASE 1: PILOT RUN (Learning Step Sizes) ---")

#init_state = [dist.sample(NUM_CHAINS) for dist in prior_dists]
init_state = prior.sample(NUM_CHAINS)
generic_steps = [0.01] * 10 
init_step_size = [tf.constant(s, dtype=tf.float32) for s in generic_steps]

kernel_pilot = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=log_posterior_batch,
    step_size=init_step_size,
    num_leapfrog_steps=10
)
adaptive_pilot = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel_pilot,
    num_adaptation_steps=int(PILOT_DRAWS * 0.8),
    target_accept_prob=0.75
)

# when trace_fn = None, returns only samples (list of 10 tensors)
pilot_samples  = tfp.mcmc.sample_chain(
    num_results=PILOT_DRAWS,
    num_burnin_steps=500,
    current_state=init_state,
    kernel=adaptive_pilot,
    trace_fn=None 
)

print("Calculating parameter variances from pilot run...")
learned_stds = []
for i, samples_tensor in enumerate(pilot_samples):
    std_val = tf.math.reduce_std(samples_tensor).numpy()
    std_val = max(std_val, 1e-4)
    learned_stds.append(std_val)

print("Learned Preconditioning Scales:")
param_names = ['rrr', 'ice', 'snow', 'firn', 'aging', 'depth', 'rough', 'lwin', 'ws', 'phase']
for n, s in zip(param_names, learned_stds):
    print(f"{n:<10}: {s:.5f}")

final_step_sizes = [tf.constant(s, dtype=tf.float32) for s in learned_stds]

# ================= 6. PHASE 2: PRODUCTION RUN (NUTS) =================
print(f"\n--- PHASE 2: PRODUCTION RUN ({PROD_DRAWS} Samples with NUTS) ---")

production_start_state = [s[-1] for s in pilot_samples]

kernel_prod = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=log_posterior_batch,
    step_size=final_step_sizes
)

adaptive_prod = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=kernel_prod,
    num_adaptation_steps=int(BURNIN * 0.8),
    target_accept_prob=0.80
)

def trace_fn(state, results):
    step = results.step
    if step % 1 == 0:
        avg_lp = tf.reduce_mean(results.inner_results.target_log_prob)
        #avg_depth = tf.reduce_mean(results.inner_results.depth)
        # depth doesnt exist for nuts, leapfrogs taken usually 7-31 is healthy
        avg_leapfrogs = tf.reduce_mean(results.inner_results.leapfrogs_taken)

        tf.print("Step:", step, "/", PROD_DRAWS, 
                 "| Avg LogProb:", avg_lp, 
                 "| Avg Leapfrogs:", avg_leapfrogs)
    return results

# Run Production (No Thinning)
final_samples, final_stats = tfp.mcmc.sample_chain(
    num_results=PROD_DRAWS,
    num_burnin_steps=BURNIN,
    current_state=production_start_state,
    kernel=adaptive_prod,
    trace_fn=trace_fn
)

# ================= 7. SAVE TO ARVIZ NETCDF =================
print(f"Converting to ArviZ InferenceData...")

# 1. Structure the data dictionary
# TFP returns samples as a list of tensors: [Param1(Draws, Chains), Param2...]
# ArviZ expects: {ParamName: (Chains, Draws)} <--- Note dimension swap!
posterior_dict = {}
for name, s in zip(param_names, final_samples):
    # s is shape (Draws, Chains) -> Transpose to (Chains, Draws)
    posterior_dict[name] = tf.transpose(s).numpy()

# 2. Extract Sample Stats for diagnostics
# Log Probability and Divergences are crucial for ArviZ
sample_stats_dict = {
    "lp": tf.transpose(final_stats.inner_results.accepted_results.target_log_prob).numpy(),
    "tree_depth": tf.transpose(final_stats.inner_results.depth).numpy(),
    "diverging": tf.transpose(final_stats.inner_results.has_divergence).numpy() # Important!
}

# 3. Create InferenceData object
idata = az.from_dict(
    posterior=posterior_dict,
    sample_stats=sample_stats_dict,
    coords={"chain": np.arange(NUM_CHAINS), "draw": np.arange(PROD_DRAWS)},
    dims={name: ["chain", "draw"] for name in param_names}
)

# 4. Save to NetCDF
print(f"Saving to {SAVE_PATH}...")
idata.to_netcdf(SAVE_PATH)

print("Done. File saved successfully.")
