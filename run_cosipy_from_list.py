"""
Run COSIPY for every parameter set in a CSV list.

Universal version of cosipy_lhs_round2.py: the list can come from anywhere (LHS design,
posterior draws, validation points, sensitivity tests). One row = one COSIPY run.

Usage:
    python run_cosipy_from_list.py <chunk_id> [list.csv]

    chunk_id   0 .. NUM_CHUNKS-1 (start one process per chunk, as before)
    list.csv   optional; overrides LIST_CSV below

What it does per run
    * reads the parameters of the row (columns mapped in PARAM_MAP; LOG_COLS are exp()-ed)
    * runs COSIPY via LHS_COSIPY.main with count = ID_OFFSET + id
    * moves the finished output (*_num{count+1}.nc) from COSIPY_OUTPUT_DIR into a folder of
      its own, COSIPY_OUTPUT_DIR/<RUN_TAG>/, so different lists never mix
    * appends a line to <RUN_TAG>/manifest_chunk<k>.csv: id, output file, parameters, status

Safety
    * runs already present in <RUN_TAG>/ (and large enough) are skipped -> a chunk can be restarted
    * before anything runs, the script checks that no *_num{count+1}.nc from ANOTHER list sits in
      COSIPY_OUTPUT_DIR (COSIPY would overwrite it); if so it stops and asks for another ID_OFFSET
"""

import os
import sys
import gc
import glob
import time
import shutil
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
GLACIER = "Abramov"
LIST_CSV = f"./{GLACIER}_validation_list.csv"   # default list; the 2nd argument overrides it
RUN_TAG = None              # output subfolder; None = name of the list file without .csv

NUM_CHUNKS = 1              # parallel processes over the list (e.g. 20 for an LHS of 2500 runs)
COSIPY_OUTPUT_DIR = "./data/output/"   # where LHS_COSIPY writes its files
MIN_SIZE_MB = 600           # a complete output is ~1.4 GB; smaller files are crashed runs
RESTART_EVERY = 25          # restart the dask workers after n completed runs (memory)
DRY_RUN = False             # True: only print what would be run

# Row id: column of the list (None = row number 0, 1, 2, ...). COSIPY's file number is
# ID_OFFSET + id + 1. Use a different offset for every list that shares COSIPY_OUTPUT_DIR:
# round 1 used 1-1500, round 2 0-2500 -- 100000 is far away from both.
ID_COL = "global_id"
ID_OFFSET = 100_000

# list column -> keyword of LHS_COSIPY.main
PARAM_MAP = {
    'rrr_factor_summer': 'rrr_factor_summer',
    'rrr_factor_winter': 'rrr_factor_winter',
    'ws_factor':         'WS_factor',
    'alb_snow':          'alb_snow',
    'alb_firn':          'alb_firn',
    'albedo_depth':      'albedo_depth',
    'bias_LWin':         'bias_LWIN',
    'bias_T2':           'bias_T2',
    't_wet':             't_wet',
}
LOG_COLS = []               # list columns stored in log space (exp() is applied); [] = all linear
POSITIVE = ['rrr_factor_summer', 'rrr_factor_winter', 'ws_factor', 'albedo_depth', 't_wet']
# ============================================================================


def cosipy_files(folder, cosipy_id):
    return glob.glob(os.path.join(folder, f"*_num{cosipy_id}.nc"))


def complete_file(folder, cosipy_id):
    for f in cosipy_files(folder, cosipy_id):
        if os.path.getsize(f) / 1e6 >= MIN_SIZE_MB:
            return f
    return None


def load_list(path):
    lst = pd.read_csv(path)
    missing = [c for c in PARAM_MAP if c not in lst.columns]
    if missing:
        sys.exit(f"columns missing in {path}: {missing}")
    if ID_COL is None or ID_COL not in lst.columns:
        if ID_COL is not None:
            print(f"(no column '{ID_COL}' -- using the row number as id)")
        lst['_id'] = np.arange(len(lst))
    else:
        lst['_id'] = lst[ID_COL].astype(int)
    if lst['_id'].duplicated().any():
        sys.exit(f"ids are not unique in {path}")
    for c in LOG_COLS:
        lst[c] = np.exp(lst[c])
    bad = [c for c in POSITIVE if c in lst.columns and (lst[c] <= 0).any()]
    if bad:
        sys.exit(f"non-positive values in {bad} -- are these columns in log space? (set LOG_COLS)")
    return lst


def manifest_row(run_id, row, status, out_file, minutes):
    d = {'run_id': run_id, 'list_id': int(row['_id']), 'status': status,
         'file': os.path.basename(out_file) if out_file else '', 'minutes': round(minutes, 1)}
    d.update({c: row[c] for c in PARAM_MAP})
    return d


def append_manifest(path, rec):
    pd.DataFrame([rec]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


if __name__ == "__main__":

    # ---- arguments --------------------------------------------------------------
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python run_cosipy_from_list.py <chunk_id 0..{NUM_CHUNKS - 1}> [list.csv]")
    try:
        chunk_id = int(sys.argv[1])
        assert 0 <= chunk_id < NUM_CHUNKS
    except (ValueError, AssertionError):
        sys.exit(f"chunk_id must be between 0 and {NUM_CHUNKS - 1}")
    list_csv = sys.argv[2] if len(sys.argv) > 2 else LIST_CSV
    if not os.path.exists(list_csv):
        sys.exit(f"list not found: {list_csv}")
    tag = RUN_TAG or os.path.splitext(os.path.basename(list_csv))[0]
    target_dir = os.path.join(COSIPY_OUTPUT_DIR, tag)
    os.makedirs(target_dir, exist_ok=True)
    manifest = os.path.join(target_dir, f"manifest_chunk{chunk_id}.csv")

    # ---- list and chunk ---------------------------------------------------------
    lst = load_list(list_csv)
    shutil.copy(list_csv, os.path.join(target_dir, f"list_{tag}.csv"))      # provenance
    chunk = lst.iloc[np.array_split(np.arange(len(lst)), NUM_CHUNKS)[chunk_id]]
    print(f"--- {tag}: chunk {chunk_id + 1}/{NUM_CHUNKS}, {len(chunk)} of {len(lst)} runs ---")
    print(f"list: {list_csv}\noutput: {target_dir}")
    print("\nparameter ranges in this chunk:")
    for c in PARAM_MAP:
        v = chunk[c].values
        print(f"  {c:<20}{v.min():10.4f} - {v.max():10.4f}   median {np.median(v):10.4f}")

    # ---- collision check: files with our numbers that belong to another list --------
    ids = ID_OFFSET + chunk['_id'].values
    known = set()
    for m in glob.glob(os.path.join(target_dir, "manifest_chunk*.csv")):
        known |= set(pd.read_csv(m)['run_id'].astype(int))
    clash = [int(i) for i in ids if cosipy_files(COSIPY_OUTPUT_DIR, i + 1) and int(i) not in known]
    if clash:
        sys.exit(f"{len(clash)} output files *_num<id>.nc already exist in {COSIPY_OUTPUT_DIR} and do not belong "
                 f"to this list (e.g. run {clash[0]} -> _num{clash[0] + 1}.nc). COSIPY would overwrite them. "
                 f"Choose another ID_OFFSET.")

    if DRY_RUN:
        todo = [i for i in ids if not complete_file(target_dir, i + 1)]
        print(f"\nDRY_RUN: {len(todo)} runs to do, {len(ids) - len(todo)} already complete.")
        sys.exit(0)

    # ---- runs --------------------------------------------------------------------
    from distributed import Client
    from cosipy.config import Config
    from cosipy.constants import Constants
    from LHS_COSIPY import main as runcosipy, make_cluster

    Config()
    Constants()
    n_ok = n_fail = n_skip = 0

    with make_cluster() as cluster:
        print(cluster)
        for n, (_, row) in enumerate(chunk.iterrows()):
            run_id = ID_OFFSET + int(row['_id'])
            cosipy_id = run_id + 1

            if complete_file(target_dir, cosipy_id):
                n_skip += 1
                print(f"[run {run_id}] already done -- skipped")
                continue
            # finished earlier but not moved (e.g. the script was killed right after COSIPY)
            left = complete_file(COSIPY_OUTPUT_DIR, cosipy_id)
            if left and run_id in known:
                shutil.move(left, os.path.join(target_dir, os.path.basename(left)))
                n_skip += 1
                print(f"[run {run_id}] found finished output, moved -- skipped")
                continue

            print(f"\n[run {run_id}]  ({n + 1}/{len(chunk)})  list id {int(row['_id'])}")
            append_manifest(manifest, manifest_row(run_id, row, 'started', None, 0.0))
            known.add(run_id)
            t0 = time.time()
            try:
                runcosipy(**{kw: float(row[col]) for col, kw in PARAM_MAP.items()},
                          count=run_id, cluster=cluster)
                out = complete_file(COSIPY_OUTPUT_DIR, cosipy_id)
                if out is None or os.path.getmtime(out) < t0 - 5:     # 5 s: coarse file-system timestamps
                    raise RuntimeError(f"no complete output *_num{cosipy_id}.nc after the run")
                dest = os.path.join(target_dir, os.path.basename(out))
                shutil.move(out, dest)
                append_manifest(manifest, manifest_row(run_id, row, 'ok', dest, (time.time() - t0) / 60))
                n_ok += 1
                print(f" -> run {run_id} done ({(time.time() - t0) / 60:.0f} min), {os.path.basename(dest)}")
            except Exception as e:
                append_manifest(manifest, manifest_row(run_id, row, f'failed: {e}', None, (time.time() - t0) / 60))
                n_fail += 1
                print(f" -> run {run_id} FAILED: {e}")

            gc.collect()
            if RESTART_EVERY and n_ok and n_ok % RESTART_EVERY == 0:
                try:
                    with Client(cluster) as client:
                        client.restart()
                    print(f"   [workers restarted after {n_ok} runs]")
                except Exception as e:
                    print(f"   [worker restart failed: {e}]")

    print(f"\n{tag}, chunk {chunk_id}: {n_ok} ok, {n_fail} failed, {n_skip} skipped. Manifest: {manifest}")
