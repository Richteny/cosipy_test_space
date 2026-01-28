import pandas as pd
import numpy as np
import sys
import gc
from scipy.stats import qmc
from cosipy.config import Config
from cosipy.constants import Constants
from COSIPY import main as runcosipy

# --- 1. CONFIGURATION ---
TOTAL_SIMULATIONS = 1800
NUM_CHUNKS = 6
SEED = 42  # CRITICAL: Ensures all 4 processes see the exact same 1000 params

# Define Parameter Ranges (Min, Max)
# Adjust these bounds to match your prior ranges
param_bounds = {
    'rrr_factor':      (np.log(0.28), np.log(0.8)),
    'alb_ice':         (0.1, 0.21),
    'alb_snow':        (0.82, 0.93),
    'alb_firn':        (0.4, 0.7),
    #'albedo_aging':    (1.0, 25.0), not needed with bougamont scheme  
    'albedo_depth':    (1.0, 14.0), 
    'center_snow':     (-1.0, 1.0), 
    'roughness_ice':   (0.7, 20.0),
    'lwin_factor':     (np.log(0.9), np.log(1.1)),
    'ws_factor':       (np.log(0.8), np.log(2)),
    'bias_T2':         (0.0, 1.5),
    't_wet':           (1.0, 23.0),
    #'t_dry':           (25.0, 35.0), keep constant at 30 in accordance with other studies and exp. nature  
    't_K':             (3.0, 16.0)
}

# --- 2. LHS GENERATOR FUNCTION ---
def generate_lhs_params(n_samples, seed):
    """Generates a reproducible LHS parameter set."""
    # Initialize LHS sampler
    sampler = qmc.LatinHypercube(d=len(param_bounds), seed=seed)
    sample = sampler.random(n=n_samples)
    
    # Scale samples to parameter bounds
    l_bounds = [b[0] for b in param_bounds.values()]
    u_bounds = [b[1] for b in param_bounds.values()]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    
    # Convert to DataFrame
    df = pd.DataFrame(sample_scaled, columns=param_bounds.keys())
    
    # Add a global ID column to ensure filenames match the master list
    df['global_id'] = range(n_samples)
    
    return df

# --- 3. MAIN EXECUTION ---

if __name__ == "__main__":
    
    # A. Parse Chunk ID from Command Line
    if len(sys.argv) < 2:
        print(f"Error: You must provide a Chunk ID (0 to {NUM_CHUNKS-1})")
        print("Usage: python run_lhs_distributed.py <chunk_id>")
        sys.exit(1)
        
    try:
        chunk_id = int(sys.argv[1])
        if chunk_id < 0 or chunk_id >= NUM_CHUNKS:
            raise ValueError
    except ValueError:
        print(f"Error: Chunk ID must be an integer between 0 and {NUM_CHUNKS-1}")
        sys.exit(1)

    print(f"--- Starting Batch Run: Chunk {chunk_id + 1}/{NUM_CHUNKS} ---")

    # B. Generate the FULL Master List (Reproducible)
    # We generate all 1000 every time to ensure consistency across chunks
    print("Generating Master LHS Parameter Set...")
    df_master = generate_lhs_params(TOTAL_SIMULATIONS, SEED)
    
    df_master.to_csv("./LHS-wide-master.csv")    
    # C. Slice the DataFrame for this Chunk
    chunk_size = TOTAL_SIMULATIONS // NUM_CHUNKS
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size
    
    # Safety catch for the last chunk (if total isn't perfectly divisible)
    if chunk_id == NUM_CHUNKS - 1:
        end_idx = TOTAL_SIMULATIONS
        
    df_chunk = df_master.iloc[start_idx:end_idx]
    
    print(f"Processing Simulations: Global ID {start_idx} to {end_idx - 1} ({len(df_chunk)} runs)")
    
    # D. Initialize COSIPY (Run once)
    Config()
    Constants()

    # E. Execution Loop
    for i, row in df_chunk.iterrows():
        global_id = int(row['global_id'])
        print(f"\n[Global Sim ID: {global_id}] Running...")

        try:
            runcosipy(
                RRR_factor        = float(np.exp(row['rrr_factor'])),
                alb_ice           = float(row['alb_ice']),
                alb_snow          = float(row['alb_snow']),
                alb_firn          = float(row['alb_firn']),
                #albedo_aging      = float(row['albedo_aging']),
                albedo_depth      = float(row['albedo_depth']),
                center_snow_transfer_function = float(row['center_snow']),
                roughness_ice     = float(row['roughness_ice']),
                LWIN_factor       = float(np.exp(row['lwin_factor'])),
                WS_factor         = float(np.exp(row['ws_factor'])),
                summer_bias_t2    = float(row['bias_T2']),
                t_wet             = float(row['t_wet']),
                t_K               = float(row['t_K']),
                count             = global_id  # CRITICAL: Use global_id for output filename
            )
            print(f" -> Sim {global_id} finished.")

        except Exception as e:
            print(f" -> Sim {global_id} FAILED: {e}")

        # Clean up memory
        gc.collect()

    print(f"\nChunk {chunk_id} Complete.")
