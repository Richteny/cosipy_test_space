import pandas as pd
import numpy as np
import sys
import gc
import os
import glob
from distributed import Client
from cosipy.config import Config
from cosipy.constants import Constants
from contextlib import nullcontext

try:
    from COSIPY import main as runcosipy, make_cluster
except ImportError:
    from COSIPY import main as runcosipy
    make_cluster = None

# --- 1. CONFIGURATION ---
GLACIER = "Abramov"
ROUND = "narrowB"                                   # "narrowA" or "narrowB"
DESIGN_CSV = {"narrowA": f"./{GLACIER}_LHS-narrowA_design.csv",
              "narrowB": f"./{GLACIER}_LHS-narrowB_design.csv"}[ROUND]
ID_OFFSET = 0       # COSIPY count = ID_OFFSET + global_id -> no clash with the wide round (0-999);
                    # narrow-A ids are 0-999, narrow-B 1000-2499, so one offset serves both
NUM_CHUNKS = 20
OUTPUT_DIR = "./data/output/"
MIN_SIZE_MB = 600
RESTART_EVERY = 25
PARAM_COLS = ['rrr_factor_summer', 'alb_snow', 'alb_firn', 'albedo_depth', 'bias_LWin',
              'ws_factor', 'bias_T2', 't_wet', 'rrr_factor_winter']


def output_exists(count, outdir=OUTPUT_DIR, min_size_mb=MIN_SIZE_MB):
    """LHS_COSIPY counts from 1: the file ends in _num{count+1}.nc. Small files = crashed runs."""
    for h in glob.glob(os.path.join(outdir, f"*_num{count + 1}.nc")):
        if os.path.getsize(h) / 1e6 >= min_size_mb:
            return h
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python {os.path.basename(__file__)} <chunk_id 0..{NUM_CHUNKS - 1}>")
    chunk_id = int(sys.argv[1])
    if not 0 <= chunk_id < NUM_CHUNKS:
        sys.exit(f"chunk_id must be between 0 and {NUM_CHUNKS - 1}")

    design = pd.read_csv(DESIGN_CSV)
    missing = [c for c in PARAM_COLS + ['global_id'] if c not in design.columns]
    if missing:
        sys.exit(f"columns missing in {DESIGN_CSV}: {missing}")
    parts = np.array_split(np.arange(len(design)), NUM_CHUNKS)
    df_chunk = design.iloc[parts[chunk_id]]
    print(f"--- {ROUND}, chunk {chunk_id + 1}/{NUM_CHUNKS}: global_id {int(df_chunk.global_id.min())}"
          f"-{int(df_chunk.global_id.max())} ({len(df_chunk)} runs), COSIPY count = {ID_OFFSET} + global_id ---")

    Config()
    Constants()
    n_ok, n_fail = 0, 0
    failed_ids = []
    with (make_cluster() if make_cluster is not None else nullcontext(None)) as cluster:
        for n, (_, row) in enumerate(df_chunk.iterrows()):
            global_id = int(row['global_id'])
            count = ID_OFFSET + global_id
            existing = output_exists(count)
            if existing is not None:
                print(f"[{global_id}] skipped ({os.path.basename(existing)})")
                continue
            print(f"\n[global_id {global_id} -> count {count}]  ({n + 1}/{len(df_chunk)})")
            try:
                runcosipy(
                    # the design CSV holds LINEAR values -- no np.exp() here
                    alb_snow          = float(row['alb_snow']),
                    alb_firn          = float(row['alb_firn']),
                    albedo_depth      = float(row['albedo_depth']),
                    bias_LWIN         = float(row['bias_LWin']),
                    WS_factor         = float(row['ws_factor']),
                    bias_T2           = float(row['bias_T2']),
                    t_wet             = float(row['t_wet']),
                    rrr_factor_summer = float(row['rrr_factor_summer']),
                    rrr_factor_winter = float(row['rrr_factor_winter']),
                    count             = count,
                    #**({'cluster': cluster} if cluster is not None else {}),
                )
                n_ok += 1
                print(f" -> {global_id} finished.")
            except Exception as e:
                n_fail += 1
                failed_ids.append(global_id)
                print(f" -> {global_id} FAILED: {e}")
            gc.collect()
            if cluster is not None and RESTART_EVERY and (n + 1) % RESTART_EVERY == 0 and (n + 1) < len(df_chunk):
                try:
                    with Client(cluster) as client:
                        client.restart()
                except Exception as e:
                    print(f"   [worker restart failed: {e}]")

    print(f"\n{ROUND} chunk {chunk_id}: {n_ok} ok, {n_fail} failed.")
    if failed_ids:
        pd.DataFrame({'global_id': failed_ids}).to_csv(f"./{GLACIER}_{ROUND}_failed_chunk{chunk_id}.csv", index=False)
