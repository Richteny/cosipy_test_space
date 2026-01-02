import argparse
import xarray as xr
import pandas as pd
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple static files, keeping time dim only for changing variables.")
    
    parser.add_argument(
        "-i", "--inputs",
        nargs='+',
        required=True,
        help="List of input NetCDF files (e.g., file_2000.nc file_2010.nc)"
    )
    
    parser.add_argument(
        "-t", "--timestamps",
        nargs='+',
        required=True,
        help="List of timestamps corresponding to the input files (e.g., 2000 2010)"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output merged NetCDF file."
    )
    
    return parser.parse_args()

def merge_and_compress(files, timestamps, output_path):
    # 1. Load and Stack Files
    datasets = []
    
    if len(files) != len(timestamps):
        print("Error: Number of files must match number of timestamps.")
        return

    print(f"Loading {len(files)} files...")
    
    for f, t in zip(files, timestamps):
        ds = xr.open_dataset(f)
        
        # Convert timestamp to pandas datetime for Xarray
        # Handles just years ("2000") or full dates ("2000-01-01")
        try:
            dt_obj = pd.to_datetime(str(t))
        except:
            dt_obj = pd.to_datetime(f"{t}-01-01")

        # Add time dimension to this file
        ds = ds.expand_dims(time=[dt_obj])
        datasets.append(ds)
    
    # Combine into one big 3D cube
    combined = xr.concat(datasets, dim='time')
    combined = combined.sortby('time')
    
    # 2. Check variables for variation
    print("\nChecking for temporal changes...")
    
    final_vars = {}
    
    # Keep coordinates separate
    coords = dict(combined.coords)
    
    forced_static = ["ASPECT", "SLOPE"]
    
    for var_name in combined.data_vars:
        da = combined[var_name]
        
        # --- NEW LOGIC: Enforce specific variables to be static ---
        if var_name in forced_static:
            print(f"  [FORCED STATIC] {var_name:<12} -> Enforcing values from first file")
            flat_var = da.isel(time=0, drop=True)
            flat_var.attrs = da.attrs 
            final_vars[var_name] = flat_var
            continue # Skip the rest of the loop for this variable
        # ----------------------------------------------------------

        # Calculate Min and Max across time (ignores NaNs by default)
        min_val = da.min(dim='time', skipna=True)
        max_val = da.max(dim='time', skipna=True)
        
        # Calculate difference (fill NaNs with 0 to allow comparison)
        diff = (max_val - min_val).fillna(0)
        
        # Check if difference is effectively zero everywhere
        is_constant = np.isclose(diff, 0).all()
        
        if is_constant:
            print(f"  [STATIC]  {var_name:<12} -> Collapsing to 2D (Values constant or NaN)")
            # Take median to flatten (handles NaNs gracefully)
            flat_var = da.median(dim='time', skipna=True)
            flat_var.attrs = da.attrs 
            final_vars[var_name] = flat_var
        else:
            print(f"  [DYNAMIC] {var_name:<12} -> Keeping time dimension (Values change)")
            final_vars[var_name] = da

    # 3. Create and Save Final Dataset
    new_ds = xr.Dataset(final_vars, coords=coords)
    
    print(f"\nSaving to {output_path}...")
    # Add compression for efficiency
    #encoding = {var: {'zlib': True, 'complevel': 5} for var in new_ds.data_vars}
    new_ds.to_netcdf(output_path) #, encoding=encoding)
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    merge_and_compress(args.inputs, args.timestamps, args.output)
