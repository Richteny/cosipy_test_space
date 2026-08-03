import importlib
import numpy as np
import pandas as pd
import toml
import sys
import os
import gc
from SALib.sample import saltelli

from cosipy.config import Config
from cosipy.constants import Constants
from sobol_COSIPY import main as runcosipy

if len(sys.argv) < 2:
    print("Error. Chunk-ID is required. E.g., python run_gsa.py 0")
    sys.exit(1)

chunk_id = int(sys.argv[1])
NUM_CHUNKS = 6

# define global sensitivity parameters and bounds
""" FOR WIDE GSA
problem = {
    'num_vars': 16,
    'names': [
        'rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn',
        'albedo_depth', 'rough_ice', 'rough_snow', 'rough_firn',
        't_star_wet', 't_star_dry', 't_star_K',
        'bias_T2', 'ws_factor', 'bias_LWIN',
        'ice_thickness', 'bottom_temp'
    ],
    'bounds': [
        [np.log(0.33), np.log(3.0)], #rrr factor
        [0.10, 0.46], #alb ice
        [0.75, 0.98], #alb snow
        [0.46, 0.75], #alb firn
        [1.0, 15.0], #alb depth
        [0.7, 20.0], #roughness ice
        [0.02, 1.6], #roughness snow
        [1.6, 6.5],  #roughness firn
        [2.0, 20],   # t star wet
        [15, 40],    # t star dry
        [2,17],      # t star K
        [-4.0, 4.0], #bias_t2
        [np.log(0.33), np.log(3)], #ws factor
        [-50, 50], #bias lwin
        #[-2.0, 2.0], #prec frac dropped and not to be used
        [350, 650], #ice thickness
        [263.15, 273.15] #bottom temperature
    ]
}
"""
problem = {
    'num_vars': 16,
    'names': [
        'rrr_factor', 'alb_ice', 'alb_snow', 'alb_firn',
        'albedo_depth', 'rough_ice', 'rough_snow', 'rough_firn',
        't_star_wet', 't_star_dry', 't_star_K', 'bias_T2',
        'ws_factor', 'bias_LWIN', 'ice_thickness', 'bottom_temp',
    ],
    'bounds': [
        [np.log(0.3386), np.log(1.8360), -0.5913, 0.5656], #rrr factor truncnorm
        [0.1290, 0.4473], #alb ice uniform
        [0.7599, 0.9344, 0.8371, 0.0553], #alb snow truncnorm
        [0.4701, 0.7398], #alb firn uniform
        [3.6660, 14.7266], #alb depth uniform
        [0.9262, 17.9908], #roughness ice uniform
        [0.0632, 1.5815], #roughness snow uniform
        [1.7321, 6.3670], #roughness firn uniform
        [3.0336, 17.6832], #t star wet uniform
        [15.2930, 38.9355], #t star dry uniform
        [2.6416, 16.5898], #t star K uniform
        [-2.1562, 3.3516, 1.4688, 1.8069], #bias_t2 truncnorm
        [np.log(0.3952), np.log(2.6819)], #ws factor uniform
        [-44.9219, 41.0352], #bias lwin uniform
        [383.8672, 639.5117], #ice thickness uniform
        [263.6539, 272.4078, 267.7984, 2.1428], #bottom temperature truncnorm
    ],
    'dists': ['truncnorm', 'unif', 'truncnorm', 'unif', 'unif', 'unif', 'unif', 'unif', 'unif', 'unif', 'unif', 'truncnorm', 'unif', 'unif', 'unif', 'truncnorm'],
}

print(f"Worker {chunk_id}: generate matrix.")
np.random.seed(42)
param_values = saltelli.sample(problem, 128)
df = pd.DataFrame(param_values, columns=problem['names'])

df['rrr_factor'] = np.exp(df['rrr_factor'])
df['ws_factor'] = np.exp(df['ws_factor'])

df['global_id'] = range(len(df))

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
    #
    constants_data['INITIAL_CONDITIONS']['initial_glacier_height'] = float(row['ice_thickness'])
    constants_data['INITIAL_CONDITIONS']['temperature_bottom'] = float(row['bottom_temp']) 
    #
    constants_data['PRECIPITATION']['mult_factor_RRR'] = float(row['rrr_factor']) 
    constants_data['PRECIPITATION']['mult_factor_WS'] = float(row['ws_factor']) 
    constants_data['PRECIPITATION']['bias_LWin'] = float(row['bias_LWIN'])
    constants_data['PRECIPITATION']['bias_T2'] = float(row['bias_T2'])  
    #constants_data['PRECIPITATION']['center_snow_transfer_function'] = float(row['pr_frac_t2'])
    #
    constants_data['CONSTANTS']['albedo_ice'] = float(row['alb_ice']) 
    constants_data['CONSTANTS']['albedo_fresh_snow'] = float(row['alb_snow'])
    constants_data['CONSTANTS']['albedo_firn'] = float(row['alb_firn']) 
    constants_data['CONSTANTS']['albedo_mod_snow_depth'] = float(row['albedo_depth']) 
    constants_data['CONSTANTS']['roughness_ice'] = float(row['rough_ice']) 
    constants_data['CONSTANTS']['roughness_fresh_snow'] = float(row['rough_snow'])
    constants_data['CONSTANTS']['roughness_firn'] = float(row['rough_firn']) 
    constants_data['CONSTANTS']['t_star_wet'] = float(row['t_star_wet']) 
    constants_data['CONSTANTS']['t_star_dry'] = float(row['t_star_dry']) 
    constants_data['CONSTANTS']['t_star_K'] = float(row['t_star_K'])

    with open('constants.toml', 'w') as f:
        toml.dump(constants_data, f)

    if 'cosipy.constants' in sys.modules:
        importlib.reload(sys.modules['cosipy.constants'])
    #Constants()

    try:
        runcosipy(count=global_id)
    except Exception as e:
        print(f"-> Sim {global_id} failed: {e}")

    gc.collect()
print(f"\n Worker {chunk_id} finished all tasks.")

