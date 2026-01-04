import xarray as xr
import numpy as np
import argparse
import sys
from numba import njit, prange

# -----------------------------------------------------------------------------
# 1. Physics: OpenAmundsen Logic (Numba Optimized)
# -----------------------------------------------------------------------------

@njit(cache=True)
def _shift_arr_retain(M, dir, n):
    """Shift array helper."""
    S = M.copy()
    if dir == 0: S[: -n - 1, :] = M[1 + n :, :]      # N
    elif dir == 1: S[: -n - 1, 1 + n :] = M[1 + n :, : -n - 1] # NE
    elif dir == 2: S[:, 1 + n :] = M[:, : -n - 1]    # E
    elif dir == 3: S[1 + n :, 1 + n :] = M[: -n - 1, : -n - 1] # SE
    elif dir == 4: S[1 + n :, :] = M[: -n - 1, :]    # S
    elif dir == 5: S[1 + n :, : -n - 1] = M[: -n - 1, 1 + n :] # SW
    elif dir == 6: S[:, : -n - 1] = M[:, 1 + n :]    # W
    elif dir == 7: S[: -n - 1, : -n - 1] = M[1 + n :, 1 + n :] # NW
    return S

@njit(cache=True, parallel=True)
def _openness_dir(dem, res, L, dir):
    """Calculate openness for one direction."""
    opn_dir = np.full(dem.shape, np.inf)
    
    for i in prange(int(np.ceil(L / res))):
        dist = res * (i + 1) * np.array([1, np.sqrt(2)])[dir % 2]
        Z_shift = _shift_arr_retain(dem, dir, i)
        
        # Angle from Zenith (Look down)
        angle = np.pi / 2 - np.arctan2(Z_shift - dem, dist)
        
        # Keep minimum angle
        idxs = np.flatnonzero(angle < opn_dir)
        opn_dir.ravel()[idxs] = angle.ravel()[idxs]

    return opn_dir

def openness(dem, res, L, negative=False):
    """Main openness function."""
    dirs = np.arange(8)
    opn = np.full((len(dirs), dem.shape[0], dem.shape[1]), np.inf)
    dem_in = -dem if negative else dem

    for dir in dirs:
        opn[dir, :, :] = _openness_dir(dem_in, res, L, dir)

    return opn.mean(axis=0)

# -----------------------------------------------------------------------------
# 2. Helpers
# -----------------------------------------------------------------------------

def get_resolution_meters(ds, x_dim, y_dim):
    """Robustly determines grid resolution in METERS."""
    if x_dim in ds.coords:
        dx = abs(ds[x_dim].values[1] - ds[x_dim].values[0])
        if dx < 0.1: # Degrees
            mid_lat = float(ds[y_dim].mean())
            res_y = dx * 111132.0
            res_x = dx * 111132.0 * np.cos(np.deg2rad(mid_lat))
            return (res_x + res_y) / 2.0
        return dx
    return 30.0

def crop_and_save_2d(ds_full, srf_data, output_path, y_dim, x_dim, buffer=1):
    """
    Crops dataset using explicit dimension names to prevent offsets.
    """
    # Create output dataset
    ds_out = ds_full.copy(deep=False)
    
    # Add SRF variable matching HGT dimensions
    ds_out['SRF'] = (ds_full['HGT'].dims, srf_data)
    ds_out['SRF'].attrs = {'long_name': 'Snow Redistribution Factor', 'units': '-'}
    
    if 'MASK' not in ds_out:
        print("Warning: No MASK found. Saving full domain.")
        ds_out.to_netcdf(output_path)
        return

    # Extract mask values matching the HGT shape
    # We use ds_full['MASK'] directly to ensure alignment
    mask = ds_out['MASK'].values
    
    if np.nansum(mask) == 0:
        print("Warning: MASK is empty. Saving full domain.")
        ds_out.to_netcdf(output_path)
        return

    # Find indices where mask is 1
    # np.where returns indices in order of dimensions (dim0, dim1)
    # Since HGT is (lat, lon), rows=lat, cols=lon
    rows, cols = np.where(mask == 1)
    
    # Calculate Crop Bounds
    y_min = max(0, np.min(rows) - buffer)
    y_max = min(ds_out.dims[y_dim], np.max(rows) + buffer)
    
    x_min = max(0, np.min(cols) - buffer)
    x_max = min(ds_out.dims[x_dim], np.max(cols) + buffer)
    
    print(f"Cropping 2D Map: {y_dim}={y_min}:{y_max}, {x_dim}={x_min}:{x_max}")
    
    # Slice using explicit dimension names
    slice_dict = {
        y_dim: slice(y_min, y_max),
        x_dim: slice(x_min, x_max)
    }
    
    # .isel() handles the coordinate slicing automatically
    ds_cropped = ds_out.isel(**slice_dict)
    ds_cropped.to_netcdf(output_path)

# -----------------------------------------------------------------------------
# 3. Main Processing
# -----------------------------------------------------------------------------

def process(static_2d_path, target_1d_path, output_path, output_2d_path, band_width):
    
    print(f"Loading 2D Static: {static_2d_path}")
    ds_2d = xr.open_dataset(static_2d_path)
    print(f"Loading 1D Target: {target_1d_path}")
    ds_1d = xr.open_dataset(target_1d_path)
    
    # 1. Identify Dimensions from HGT variable (The Truth)
    # Your file has HGT(lat, lon). dims[0]=lat, dims[1]=lon.
    hgt_dims = ds_2d['HGT'].dims
    y_dim = hgt_dims[0] # lat
    x_dim = hgt_dims[1] # lon
    
    print(f"Detected Dimensions: Y='{y_dim}', X='{x_dim}'")
    
    # Get values in correct order (lat, lon)
    dem_2d = ds_2d['HGT'].values
    
    if 'MASK' in ds_2d:
        mask_2d = ds_2d['MASK'].values
        mask_2d[np.isnan(mask_2d)] = 0
    else:
        mask_2d = np.ones_like(dem_2d)

    # 2. Prepare Data (Fill NaNs)
    dem_calc = dem_2d.copy()
    valid_pixels = ~np.isnan(dem_calc)
    if np.any(valid_pixels):
        dem_calc[~valid_pixels] = np.nanmedian(dem_calc[valid_pixels])
    
    res_meters = get_resolution_meters(ds_2d, x_dim, y_dim)

    # 3. Calculate Physics
    print("Generating 2D SRF Map...")
    dem_arr = dem_calc.astype(np.float64)
    
    print("  > Computing Negative Openness (L=50m)...")
    neg_50 = openness(dem_arr, res_meters, L=50.0, negative=True)
    
    print("  > Computing Negative Openness (L=5000m)...")
    neg_5000 = openness(dem_arr, res_meters, L=5000.0, negative=True)
    
    psi_eff_50 = 3.0 * (neg_50 - 1.2)
    psi_eff_5000 = 3.0 * (neg_5000 - 1.0)
    srf_50 = np.clip(psi_eff_50, 0.1, 1.6)
    srf_5000 = np.clip(psi_eff_5000, 0.1, 1.6)
    
    srf_2d = 0.5 * (srf_50 + srf_5000)

    # 4. Save 2D Map (Cropped)
    if output_2d_path:
        # Pass explicit dimension names to ensure correct slicing
        crop_and_save_2d(ds_2d, srf_2d, output_2d_path, y_dim, x_dim)
    
    # 5. Aggregate to 1D
    print("Aggregating to 1D Bands...")
    target_levels = ds_1d['HGT'].values
    flat_levels = target_levels.flatten()
    flat_srf = np.zeros_like(flat_levels)
    half_width = band_width / 2.0
    
    # Mask SRF for stats
    srf_glacier = srf_2d.copy()
    srf_glacier[mask_2d != 1] = np.nan
    
    for i, center_elev in enumerate(flat_levels):
        if np.isnan(center_elev):
            flat_srf[i] = np.nan
            continue
        z_min = center_elev - half_width
        z_max = center_elev + half_width
        in_band = (dem_2d >= z_min) & (dem_2d < z_max) & (mask_2d == 1)
        
        if np.any(in_band):
            flat_srf[i] = np.nanmean(srf_glacier[in_band])
        else:
            flat_srf[i] = 1.0
            
    srf_final = flat_srf.reshape(target_levels.shape)

    # 6. Save 1D Output
    ds_out = ds_1d.copy(deep=True)
    ds_out['SRF'] = (ds_1d['HGT'].dims, srf_final)
    ds_out['SRF'].attrs = {'long_name': 'Snow Redistribution Factor (Hanzer 2016)', 'units': '-'}
    
    print(f"Saving 1D profile to {output_path}")
    ds_out.to_netcdf(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--static2d", required=True)
    parser.add_argument("-t", "--target1d", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-o2", "--output2d", required=False)
    parser.add_argument("-b", "--bandwidth", type=float, default=20.0)
    args = parser.parse_args()
    
    process(args.static2d, args.target1d, args.output, args.output2d, args.bandwidth)