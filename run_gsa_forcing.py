import importlib
import numpy as np
import pandas as pd
import toml
import sys
import os
import gc
import json
from SALib.sample import saltelli

from cosipy.config import Config
from cosipy.constants import Constants
from sobol_COSIPY import main as runcosipy

if len(sys.argv) < 2:
    print("Error. Chunk-ID is required. E.g., python run_gsa.py 0")
    sys.exit(1)

chunk_id = int(sys.argv[1])
NUM_CHUNKS = 6
N_BASE     = 256          # Saltelli base sample -> N*(2D+2) = 256*22 = 5632 runs
                          # (use 512 for stabler 2nd-order indices; 128 too few for D=10)

# =====================================================================
# FORCING vs. PARAMETER SENSITIVITY  (conventional, citable perturbation ranges)
#   7 model parameters : POSTERIOR support (min/max), uniform
#   3 forcing biases    : varied by the standard mass-balance-sensitivity
#                        increment around the calibrated baseline (uniform):
#                          bias_T2   baseline +- 1 K     (Hock et al.; Urumqi 2019; Chandra 2025)
#                          bias_LWIN baseline +- 20 W/m2
#                          ws_factor baseline * (1 +- 20%)
#                        rrr_factor kept as a calibrated PARAMETER (posterior range).
#   Baseline = calibrated forcing corrections (the MB=-0.87 baseline run).
# =====================================================================
# baseline calibrated forcing corrections (EDIT to match your -0.87 run)
BASE_T2, BASE_LWIN, BASE_WS = 0.230, -35.790, 0.676
DT, DLW, PCT = 1.0, 20.0, 0.20    # +-1 K, +-20 W/m2, +-20%
problem = {
    'num_vars': 10,
    'names': [
        'rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn',
        'albedo_aging', 'albedo_depth', 'rough_ice',   # 7 posterior params
        'bias_T2', 'ws_factor', 'bias_LWIN',           # 3 forcing-bias perturbations
    ],
    'bounds': [
        # ---- 7 model parameters: posterior [lower, upper] (uniform) ----
        [0.6626, 0.7572],    # rrr_factor   (calibrated precip factor, posterior range)
        [0.2098, 0.2300],    # alb_ice
        [0.8900, 0.8994],    # alb_snow
        [0.5886, 0.6746],    # alb_firn
        [10.4143, 18.5439],  # albedo_aging
        [1.0000, 1.0642],    # albedo_depth
        [0.7001, 10.2807],   # rough_ice
        # ---- 3 forcing perturbations: conventional +- increment around baseline ----
        [BASE_T2 - DT,      BASE_T2 + DT],        # bias_T2     +-1 K
        [BASE_WS*(1-PCT),   BASE_WS*(1+PCT)],     # ws_factor   +-20%  (linear)
        [BASE_LWIN - DLW,   BASE_LWIN + DLW],     # bias_LWIN   +-20 W/m2
    ],
    'dists': ['unif'] * 10,
}

PARAM_GROUP   = ['rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn', 'albedo_aging', 'albedo_depth', 'rough_ice']
FORCING_GROUP = ['bias_T2', 'ws_factor', 'bias_LWIN']

print(f"Worker {chunk_id}: generate matrix (N_base={N_BASE}, D={problem['num_vars']}).")
np.random.seed(42)
param_values = saltelli.sample(problem, N_BASE)
df = pd.DataFrame(param_values, columns=problem['names'])

df['global_id'] = range(len(df))
print(f"Total simulations: {len(df)}  (= N_base*(2*D+2) = {N_BASE}*{2*problem['num_vars']+2})")

chunk_indices = np.array_split(df.index, NUM_CHUNKS)
df_chunk = df.loc[chunk_indices[chunk_id]]
print(f"Starting chunk {chunk_id} with {len(df_chunk)} simulations.")

Config()
output_path = "./data/output/"

for index, row in df_chunk.iterrows():
    global_id = int(row['global_id'])
    cosipy_id = global_id + 1
    expected_csv = os.path.join(output_path, f"gsa_result_sim_{cosipy_id}.csv")

    if os.path.exists(expected_csv):
        print(f"-> Sim {global_id} skipped. File already exists.")
        continue

    with open('constants.toml', 'r') as f:
        constants_data = toml.load(f)

    # --- forcing-bias corrections (applied inside COSIPY core) ---
    constants_data['PRECIPITATION']['mult_factor_RRR'] = float(row['rrr_factor'])
    constants_data['PRECIPITATION']['mult_factor_WS']  = float(row['ws_factor'])
    constants_data['PRECIPITATION']['bias_LWin']       = float(row['bias_LWIN'])
    constants_data['PRECIPITATION']['bias_T2']         = float(row['bias_T2'])

    # --- model parameters ---
    constants_data['CONSTANTS']['albedo_ice']            = float(row['alb_ice'])
    constants_data['CONSTANTS']['albedo_fresh_snow']     = float(row['alb_snow'])
    constants_data['CONSTANTS']['albedo_firn']           = float(row['alb_firn'])
    constants_data['CONSTANTS']['albedo_mod_snow_aging'] = float(row['albedo_aging'])
    constants_data['CONSTANTS']['albedo_mod_snow_depth'] = float(row['albedo_depth'])
    constants_data['CONSTANTS']['roughness_ice']         = float(row['rough_ice'])

    # ice_thickness / bottom_temp / roughness_snow|firn / t_star_* held fixed:
    # this experiment isolates the 7 calibrated params vs the 3 forcing biases.

    with open('constants.toml', 'w') as f:
        toml.dump(constants_data, f)

    if 'cosipy.constants' in sys.modules:
        importlib.reload(sys.modules['cosipy.constants'])

    try:
        runcosipy(count=global_id)
    except Exception as e:
        print(f"-> Sim {global_id} failed: {e}")

    gc.collect()

print(f"\n Worker {chunk_id} finished all tasks.")

if chunk_id == 0:
    df.to_csv(os.path.join(output_path, "gsa_sample_matrix.csv"), index=False)
    with open(os.path.join(output_path, "gsa_problem.json"), "w") as f:
        json.dump({"problem": problem, "param_group": PARAM_GROUP,
                   "forcing_group": FORCING_GROUP, "n_base": N_BASE}, f, indent=2)
    print("Saved gsa_sample_matrix.csv and gsa_problem.json")
