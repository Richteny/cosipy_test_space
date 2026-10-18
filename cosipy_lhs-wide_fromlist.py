import pandas as pd
import numpy as np
import sys
import gc
import os
import glob
from scipy.stats import qmc, truncnorm
from distributed import Client
from cosipy.config import Config
from cosipy.constants import Constants
from contextlib import nullcontext

# make_cluster is optional: without it, COSIPY sets up its own cluster per run
try:
    from COSIPY import main as runcosipy, make_cluster
except ImportError:
    from COSIPY import main as runcosipy
    make_cluster = None

# --- 1. CONFIGURATION ---
TOTAL_SIMULATIONS = 1000
NUM_CHUNKS = 12
SEED = 11  # CRITICAL: Ensures all 4 processes see the exact same 1000 params

GLACIER = "Dongkemadi"
OUTPUT_DIR = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/"
# NOTE: the skip check (output_exists) goes by run number only. Move old wide-LHS output of this glacier out of
# OUTPUT_DIR before starting, or the new parameter sets with the same numbers are skipped as "done".
MIN_SIZE_MB = 600

RESTART_EVERY = 25

# Priors in SAMPLING space (ln for rrr_factor_* and ws_factor, which are exponentiated below), from the prior
# section of A_99_create-prior-knowledge (paste its PRIORS block here). The LHS is drawn in probability space and
# mapped through each prior's quantile function, so the wide LHS samples FROM the priors.
#   tn(mu, sd[, lo, hi]): normal truncated at mu +- TRUNC sd (and at lo / hi, physical limits)
#   ('unif', lo, hi):     no observational prior -> uniform
TRUNC = 2.5


def tn(mu, sd, lo=-np.inf, hi=np.inf):
    return ('norm', mu, sd, max(mu - TRUNC * sd, lo), min(mu + TRUNC * sd, hi))


WINTER_CENTRE = {"Abramov": 0.6780, "Bordu": 0.8459, "Dongkemadi": 0.2390, "Guliya": 0.9478, "Halji": -0.2107,
                 "InteriorGlacier": 0.4318, "Kolahoi": 0.4383, "Mera": -1.2379, "ParlungNo4": -0.5108,
                 "Urumqi": 0.5306}   # ln of the HMA_SR_D winter factor
PRIORS = {
    'rrr_factor_summer': tn(0.0, 0.8),
    'alb_snow':          tn(0.83, 0.05, 0.75, 0.90),
    'alb_firn':          ('unif', 0.45, 0.75),
    'albedo_depth':      ('unif', 0.1, 14.0),
    'bias_LWin':         tn(22.4, 25.0),
    'ws_factor':         tn(0.0, 0.7),        # <- centre from A_99 (ln of measured / forcing wind)
    'bias_T2':           tn(0.23, 1.5),
    't_wet':             ('unif', 0.1, 23.0),
    'rrr_factor_winter': tn(WINTER_CENTRE[GLACIER], 0.65),
}


def from_prior(u, spec):
    """uniform [0, 1] -> the prior (quantile function)."""
    if spec[0] == 'unif':
        return spec[1] + u * (spec[2] - spec[1])
    _, mu, sd, lo, hi = spec
    return truncnorm.ppf(u, (lo - mu) / sd, (hi - mu) / sd, loc=mu, scale=sd)


# --- 2. LHS GENERATOR FUNCTION ---
def generate_lhs_params(n_samples, seed):
    """Reproducible LHS drawn from the priors (stratified in probability space)."""
    sampler = qmc.LatinHypercube(d=len(PRIORS), optimization="random-cd", seed=seed)
    u = sampler.random(n=n_samples)
    df = pd.DataFrame({p: from_prior(u[:, j], spec) for j, (p, spec) in enumerate(PRIORS.items())})
    # Add a global ID column to ensure filenames match the master list
    df['global_id'] = range(n_samples)
    return df

def output_exists(global_id, outdir=OUTPUT_DIR, min_size_mb=MIN_SIZE_MB):
    """Gibt es schon eine brauchbare Ausgabedatei fuer diese Sim-ID?

    LHS_COSIPY zaehlt intern ab 1 (count = count + 1), der Dateiname endet
    also auf _num{global_id+1}.nc. Zu kleine Dateien stammen von
    abgebrochenen Laeufen und zaehlen nicht.
    """
    cosipy_id = global_id + 1
    hits = glob.glob(os.path.join(outdir, f"*_num{cosipy_id}.nc"))
    for h in hits:
        if os.path.getsize(h) / 1e6 >= min_size_mb:
            return h
    return None

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
    
    df_master.to_csv(f"./{GLACIER}_LHS-wide-prior-master.csv")    
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
    failed_ids = [] 
    if make_cluster is None:
        print("make_cluster not available -- COSIPY handles the cluster itself for each run")
    with (make_cluster() if make_cluster is not None else nullcontext(None)) as cluster:
        if cluster is not None:
            print(cluster)
        extra = {'cluster': cluster} if cluster is not None else {}
 
        for n, (i, row) in enumerate(df_chunk.iterrows()):
            global_id = int(row['global_id'])
            print(f"\n[Global Sim ID: {global_id}]  ({n + 1}/{len(df_chunk)})")
 
            existing = output_exists(global_id)
            if existing is not None:
                print(f"Global Sim ID: {global_id} skipped "
                      f"({os.path.basename(existing)})")
                continue

            try:
                runcosipy(
                    #RRR_factor        = float(np.exp(row['rrr_factor'])),
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
                    rrr_factor_summer = float(np.exp(row['rrr_factor_summer'])),
                    rrr_factor_winter = float(np.exp(row['rrr_factor_winter'])),
                    count             = global_id,   # CRITICAL: output filename
                    #**extra,                         # cluster=cluster, if make_cluster exists
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
            if cluster is not None and RESTART_EVERY and (n + 1) % RESTART_EVERY == 0 and (n + 1) < len(df_chunk):
                try:
                    with Client(cluster) as client:
                        client.restart()
                    print(f"   [Worker nach {n + 1} Simulationen neu gestartet]")
                except Exception as e:
                    print(f"   [Worker-Neustart fehlgeschlagen: {e}]")
 
    print(f"\nChunk {chunk_id} Complete.  {n_ok} erfolgreich, {n_fail} fehlgeschlagen.")
