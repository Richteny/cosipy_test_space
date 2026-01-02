import xarray as xr
import numpy as np
import pandas as pd
import glob
import os

# -------------------------
# CONFIG
# -------------------------
TIME_START_MB = "2000-01-01"
OUTPUT_CSV = "./LHS-narrow_MBfixed_results.csv"
PATH = "/data/scratch/richteny/thesis/cosipy_test_space/data/output/Halji/LHS-narrow/"

# -------------------------
# Helper functions
# -------------------------
def extract_id(filename):
    """
    Extract everything after 'RRR-' and before '_num2.nc'
    """
    base = os.path.basename(filename)
    return base.split("RRR-")[1].replace("_num2.nc", "")


def compute_geodetic_mb(ncfile, time_start_mb):
    dsmb = xr.open_dataset(ncfile)

    # Fixed reference weights
    ref_weights = dsmb["N_Points"].sel(
        time=time_start_mb, method="nearest"
    )

    # Reference area
    ref_area_total = ref_weights.sum()

    # Area-weighted mass change
    total_mass_change = (dsmb["MB"] * ref_weights).sum(dim=["lat", "lon"])

    # Glacier-wide MB
    weighted_mb = total_mass_change / ref_area_total

    # Annual aggregation
    dfmb = weighted_mb.to_dataframe(name="weighted_mb")
    annual_mb = dfmb.resample("1YE").sum()

    geod_mb = np.nanmean(annual_mb["weighted_mb"].values)

    dsmb.close()
    return geod_mb


# -------------------------
# Main workflow
# -------------------------

# Get files sorted by creation time (oldest first)
files = sorted(
    glob.glob(PATH+"*.nc"),
    key=os.path.getctime
)

results = []

for f in files:
    try:
        mb_value = compute_geodetic_mb(f, TIME_START_MB)
        file_id = extract_id(f)

        results.append({
            "filename": file_id,
            "MB": mb_value
        })

        print(f"Processed: {file_id}")

    except Exception as e:
        print(f"Failed on {f}: {e}")

# Write CSV
df_out = pd.DataFrame(results)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved results to {OUTPUT_CSV}")

