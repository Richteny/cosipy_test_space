import pandas as pd
import numpy as np
import sys
import gc
from scipy.stats import qmc
from distributed import Client
from cosipy.config import Config
from cosipy.constants import Constants
from LHS_COSIPY import main as runcosipy, make_cluster

# --- 1. CONFIGURATION ---
TOTAL_SIMULATIONS = 8
NUM_CHUNKS = 4
SEED = 42  # CRITICAL: Ensures all 4 processes see the exact same 1000 params

RESTART_EVERY = 50
# Define Parameter Ranges (Min, Max)
# Adjust these bounds to match your prior ranges
param_bounds = {
    'rrr_factor':      (np.log(0.333), np.log(3.0)),
    #'alb_ice':         (0.11, 0.39),
    'alb_snow':        (0.75, 0.9),
    'alb_firn':        (0.45, 0.75),
    #'albedo_aging':    (1.0, 25.0), not needed with bougamont scheme  
    'albedo_depth':    (1.0, 14.0), 
    #'center_snow':     (-1.5, 1.5), 
    #'roughness_ice':   (0.7, 20.0),
    'bias_LWin':       (-50, 50),
    'ws_factor':       (np.log(0.5), np.log(2)),
    'bias_T2':         (-2, 2),
    't_wet':           (1.0, 23.0),
    'min_snowfall':    (0.001, 0.008),
    #'t_dry':           (25.0, 35.0), keep constant at 30 in accordance with other studies and exp. nature  
    #'t_K':             (3.0, 16.0), keep constant at 
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
    
    df_master.to_csv("./Abramov_LHS-wide-master.csv")    
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


    # E. Cluster EINMAL erzeugen und ueber alle Simulationen wiederverwenden.
    #    Bisher baute main() pro Aufruf einen eigenen Cluster auf und wieder ab;
    #    dabei musste numba in jedem neuen Worker alles neu kompilieren (~40 s
    #    pro Simulation). Mit bestehenden Workern faellt das nur einmal an.
    n_ok, n_fail = 0, 0
 
    with make_cluster() as cluster:
        print(cluster)
 
        for n, (i, row) in enumerate(df_chunk.iterrows()):
            global_id = int(row['global_id'])
            print(f"\n[Global Sim ID: {global_id}]  ({n + 1}/{len(df_chunk)})")
 
            try:
                runcosipy(
                    RRR_factor        = float(np.exp(row['rrr_factor'])),
                    #alb_ice           = float(row['alb_ice']),
                    alb_snow          = float(row['alb_snow']),
                    alb_firn          = float(row['alb_firn']),
                    #albedo_aging      = float(row['albedo_aging']),
                    albedo_depth      = float(row['albedo_depth']),
                    #center_snow_transfer_function = float(row['center_snow']),
                    #roughness_ice     = float(row['roughness_ice']),
                    bias_LWIN         = float(row['bias_LWin']),
                    WS_factor         = float(np.exp(row['ws_factor'])),
                    bias_T2           = float(row['bias_T2']),
                    t_wet             = float(row['t_wet']),
                    #t_K               = float(row['t_K']),
                    minimum_snowfall  = float(row['min_snowfall']),
                    count             = global_id,   # CRITICAL: output filename
                    cluster           = cluster,     # NEU: bestehenden Cluster nutzen
                )
                n_ok += 1
                print(f" -> Sim {global_id} finished.")
 
            except Exception as e:
                n_fail += 1
                failed_ids.append(global_id)
                print(f" -> Sim {global_id} FAILED: {e}")

            # Clean up memory
            gc.collect()
 
            # Worker gelegentlich neu starten: bei mehreren hundert Laeufen
            # sammelt sich sonst Speicher an. Kostet einmal Kompilierzeit,
            # aber nur alle RESTART_EVERY Simulationen statt jedes Mal.
            if RESTART_EVERY and (n + 1) % RESTART_EVERY == 0 and (n + 1) < len(df_chunk):
                try:
                    with Client(cluster) as client:
                        client.restart()
                    print(f"   [Worker nach {n + 1} Simulationen neu gestartet]")
                except Exception as e:
                    print(f"   [Worker-Neustart fehlgeschlagen: {e}]")
 
    print(f"\nChunk {chunk_id} Complete.  {n_ok} erfolgreich, {n_fail} fehlgeschlagen.")
