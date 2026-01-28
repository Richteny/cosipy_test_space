import pandas as pd
import sys
import gc
from cosipy.config import Config
from cosipy.constants import Constants
from COSIPY import main as runcosipy

# 1. Initialize Config (Run once)
Config()
Constants()

# 2. Load Batch File
if len(sys.argv) < 2:
    print("Error: Provide a csv file to run!")
    sys.exit(1)

batch_file = sys.argv[1]
print(f"Loading batch: {batch_file}")

df_batch = pd.read_csv(batch_file)

# 3. Execution Loop
for i, row in df_batch.iterrows():
    print(f"\n[Sim {i}] Running...")

    # We use 'i' as the count so every file gets a unique ID 
    # (e.g., output_0.nc, output_1.nc, etc.)
    # Make sure your COSIPY.py uses this 'count' variable in the filename!
    try:
        runcosipy(
            RRR_factor      = float(row['rrr_factor']),
            alb_ice         = float(row['alb_ice']),
            alb_snow        = float(row['alb_snow']),  # Fixed typo here
            alb_firn        = float(row['alb_firn']),
            #albedo_aging    = float(row['paralbedo_aging']),
            albedo_depth    = float(row['albedo_depth']),
            center_snow_transfer_function = float(row['center_snow']),
            roughness_ice   = float(row['roughness_ice']),
            LWIN_factor     = float(row['lwin_factor']),
            WS_factor       = float(row['ws_factor']),
            summer_bias_t2  = float(row['summer_bias_t2']),
            t_wet           = float(row['t_wet']),
            t_K             = float(row['t_K']),
            count           = i
            #RRR_factor      = float(row['parRRR_factor']),
            #alb_ice         = float(row['paralb_ice']),
            #alb_snow        = float(row['paralb_snow']),  # Fixed typo here
            #alb_firn        = float(row['paralb_firn']),
            #albedo_aging    = float(row['paralbedo_aging']),
            #albedo_depth    = float(row['paralbedo_depth']),
            #center_snow_transfer_function = float(row['parcenter_snow']),
            #roughness_ice   = float(row['parroughness_ice']),
            #LWIN_factor     = float(row['parLWIN_factor']),
            #WS_factor       = float(row['parWS_factor']),
            #count           = 2  # Critical Fix: Use loop index
        )
        print(f" -> Sim {i} finished.")

    except Exception as e:
        print(f" -> Sim {i} FAILED: {e}")

    # 4. Clean up memory
    gc.collect()

print("\nBatch Run Complete.")
