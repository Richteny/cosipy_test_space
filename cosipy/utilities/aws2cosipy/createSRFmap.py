import xarray as xr
import numpy as np
import pandas as pd
import argparse
import sys
from numba import njit, prange

# -----------------------------------------------------------------------------
# 1. Physics: OpenAmundsen Logic (Numba Optimized)  -- unchanged
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
        if dx < 0.1:  # Degrees
            mid_lat = float(ds[y_dim].mean())
            res_y = dx * 111132.0
            res_x = dx * 111132.0 * np.cos(np.deg2rad(mid_lat))
            return (res_x + res_y) / 2.0
        return dx
    return 30.0


def build_topographic_srf(dem_arr, res_meters):
    """Compute the *unnormalised* topographic SRF field from the DEM only.

    This is the expensive part (the L=5000 m openness scan) and depends ONLY
    on HGT.  As long as the DEM grid is identical across outlines, it is the
    same for every mask, so compute it once and reuse it.
    """
    print("  > Computing Negative Openness (L=50m)...")
    neg_50 = openness(dem_arr, res_meters, L=50.0, negative=True)

    print("  > Computing Negative Openness (L=5000m)...")
    neg_5000 = openness(dem_arr, res_meters, L=5000.0, negative=True)

    # Derived over the Alps; no other reference, so we keep it.
    psi_eff_50 = 3.0 * (neg_50 - 1.2)
    psi_eff_5000 = 3.0 * (neg_5000 - 1.0)

    # Wide pre-clip only to kill numerical blow-ups; final range is enforced
    # later inside the mass-conserving normalisation.
    srf_50 = np.clip(psi_eff_50, 0.0, 5.0)
    srf_5000 = np.clip(psi_eff_5000, 0.0, 5.0)

    return 0.5 * (srf_50 + srf_5000)


def normalize_mass_conserving(srf_raw, mask, lo=0.2, hi=1.6,
                              max_iter=100, tol=1e-5, verbose=True):
    """Normalise SRF so the glacier mean is 1.0 *and* values stay in [lo, hi].

    A plain clip(srf/mean, lo, hi) breaks conservation because the tails are
    truncated.  Here we alternate:
        (a) clip to [lo, hi]
        (b) restore the glacier mean to 1.0 by distributing the residual over
            the interior (un-pinned) glacier pixels only.
    Pinned pixels stay at their bound; interior pixels absorb the deficit.
    Converges whenever lo < 1 < hi.

    Returns (srf_normalised, final_glacier_mean).
    """
    glac = (mask == 1)
    n_glac = int(np.count_nonzero(glac))
    if n_glac == 0:
        return srf_raw, np.nan

    out = srf_raw.astype(np.float64).copy()

    # Initial scale to mean 1 over the *current* glacier extent.
    m0 = np.nanmean(out[glac])
    if m0 > 0:
        out = out / m0

    for it in range(max_iter):
        out = np.clip(out, lo, hi)
        m = np.nanmean(out[glac])
        err = 1.0 - m
        if abs(err) < tol:
            break
        interior = glac & (out > lo) & (out < hi)
        n_int = int(np.count_nonzero(interior))
        if n_int == 0:
            # everything pinned at a bound; cannot conserve within range
            break
        # shift interior pixels so the *glacier* mean moves by `err`
        out[interior] += err * n_glac / n_int

    out = np.clip(out, lo, hi)
    final_mean = float(np.nanmean(out[glac]))
    if verbose:
        status = "OK" if abs(final_mean - 1.0) < 1e-3 else "RESIDUAL"
        print(f"    normalisation [{status}] glacier-mean SRF = {final_mean:.5f} "
              f"after {it + 1} iters (range [{lo}, {hi}])")
    return out, final_mean


def aggregate_to_bands(srf_norm_2d, dem_2d, mask_2d, target_levels,
                       band_width, empty_fill=1.0):
    """Aggregate the 2D normalised SRF onto the 1D elevation-band grid.

    For each band centre, average the on-glacier 30 m pixels whose elevation
    falls within +/- band_width/2.  Bands with no on-glacier pixels (e.g. a
    band that has fully de-glaciated, N_Points -> 0) get `empty_fill` (1.0,
    i.e. neutral) since they will be inactive anyway.

    Returns SRF on the same shape as `target_levels`.
    """
    flat_levels = target_levels.flatten()
    flat_srf = np.zeros_like(flat_levels, dtype=np.float64)
    half_width = band_width / 2.0

    srf_glacier = srf_norm_2d.copy()
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
            flat_srf[i] = empty_fill

    return flat_srf.reshape(target_levels.shape)


def count_points_per_band(dem_2d, mask_2d, target_levels, band_width):
    """Fallback N_Points: count on-glacier 30 m pixels per band."""
    flat_levels = target_levels.flatten()
    counts = np.zeros_like(flat_levels, dtype=np.float64)
    half_width = band_width / 2.0
    for i, center_elev in enumerate(flat_levels):
        if np.isnan(center_elev):
            counts[i] = np.nan
            continue
        z_min = center_elev - half_width
        z_max = center_elev + half_width
        in_band = (dem_2d >= z_min) & (dem_2d < z_max) & (mask_2d == 1)
        counts[i] = float(np.count_nonzero(in_band))
    return counts.reshape(target_levels.shape)


def load_2d_static(static_2d_path):
    """Open a 30 m static and return (ds, dem_2d, mask_2d, y_dim, x_dim, res)."""
    ds_2d = xr.open_dataset(static_2d_path)
    hgt_dims = ds_2d['HGT'].dims
    y_dim, x_dim = hgt_dims[0], hgt_dims[1]
    dem_2d = ds_2d['HGT'].values
    if 'MASK' in ds_2d:
        mask_2d = ds_2d['MASK'].values.copy()
        mask_2d[np.isnan(mask_2d)] = 0
    else:
        mask_2d = np.ones_like(dem_2d)
    res = get_resolution_meters(ds_2d, x_dim, y_dim)
    return ds_2d, dem_2d, mask_2d, y_dim, x_dim, res


def filled_dem(dem_2d):
    """Fill NaNs in the DEM with the median so openness is well-defined."""
    dem_calc = dem_2d.copy()
    valid = ~np.isnan(dem_calc)
    if np.any(valid):
        dem_calc[~valid] = np.nanmedian(dem_calc[valid])
    return dem_calc.astype(np.float64)


def crop_and_save_2d(ds_full, srf_data, output_path, y_dim, x_dim, buffer=1):
    """Crops dataset using explicit dimension names to prevent offsets."""
    ds_out = ds_full.copy(deep=False)
    ds_out['SRF'] = (ds_full['HGT'].dims, srf_data)
    ds_out['SRF'].attrs = {'long_name': 'Snow Redistribution Factor', 'units': '-'}

    if 'MASK' not in ds_out:
        print("Warning: No MASK found. Saving full domain.")
        ds_out.to_netcdf(output_path)
        return

    mask = ds_out['MASK'].values
    if np.nansum(mask) == 0:
        print("Warning: MASK is empty. Saving full domain.")
        ds_out.to_netcdf(output_path)
        return

    rows, cols = np.where(mask == 1)
    y_min = max(0, np.min(rows) - buffer)
    y_max = min(ds_out.dims[y_dim], np.max(rows) + buffer)
    x_min = max(0, np.min(cols) - buffer)
    x_max = min(ds_out.dims[x_dim], np.max(cols) + buffer)

    print(f"Cropping 2D Map: {y_dim}={y_min}:{y_max}, {x_dim}={x_min}:{x_max}")
    ds_cropped = ds_out.isel(**{y_dim: slice(y_min, y_max),
                                x_dim: slice(x_min, x_max)})
    ds_cropped.to_netcdf(output_path)


# -----------------------------------------------------------------------------
# 3a. Single-outline processing (original behaviour, new normalisation)
# -----------------------------------------------------------------------------

def process_single(static_2d_path, target_1d_path, output_path, output_2d_path,
                   band_width, lo, hi):

    print(f"Loading 2D Static: {static_2d_path}")
    ds_2d, dem_2d, mask_2d, y_dim, x_dim, res = load_2d_static(static_2d_path)
    print(f"Detected Dimensions: Y='{y_dim}', X='{x_dim}'  (res ~{res:.1f} m)")

    print(f"Loading 1D Target: {target_1d_path}")
    ds_1d = xr.open_dataset(target_1d_path)

    if abs(res - 30.0) > 5.0:
        print(f"Warning: DEM resolution is {res:.1f} m; L=50 m scans only "
              f"{int(np.ceil(50 / res))} pixels. Results may be unreliable.")

    print("Generating 2D SRF Map...")
    srf_2d = build_topographic_srf(filled_dem(dem_2d), res)

    srf_norm, _ = normalize_mass_conserving(srf_2d, mask_2d, lo=lo, hi=hi)

    if output_2d_path:
        crop_and_save_2d(ds_2d, srf_norm, output_2d_path, y_dim, x_dim)

    print("Aggregating to 1D Bands...")
    target_levels = ds_1d['HGT'].values
    srf_final = aggregate_to_bands(srf_norm, dem_2d, mask_2d,
                                   target_levels, band_width)

    ds_out = ds_1d.copy(deep=True)
    ds_out['SRF'] = (ds_1d['HGT'].dims, srf_final)
    ds_out['SRF'].attrs = {'long_name': 'Norm. Snow Redistribution Factor (Hanzer 2016)',
                           'units': '-'}
    print(f"Saving 1D profile to {output_path}")
    ds_out.to_netcdf(output_path)
    print("Done.")


# -----------------------------------------------------------------------------
# 3b. Multi-outline merge -> single time-stamped static
# -----------------------------------------------------------------------------

def process_merge(static_list, target_list, dates, merge_output,
                  band_width, lo, hi):
    """Run all outlines and stack SRF (+ N_Points) along a sparse time axis.

    The openness/topographic field is computed ONCE from the first DEM and
    reused for every outline (DEM grid assumed identical; checked).  Only the
    mask-dependent steps (normalisation + band aggregation) are redone per
    outline, which is fast.

    Output is a single static with:
        SRF(time, y, x)       -- mass-conserving, in [lo, hi], per outline
        N_Points(time, y, x)  -- per outline (from target files or recomputed)
    plus the static geometry (HGT, MASK, SLOPE, ASPECT, SVF ...) from the
    first target as a template.  `time` holds the sporadic outline dates;
    cosmo2cosipy forward-fills these onto the simulation timeline.
    """
    n = len(static_list)
    if not (len(target_list) == n == len(dates)):
        raise ValueError(
            f"--static2d ({len(static_list)}), --target1d ({len(target_list)}) "
            f"and --dates ({len(dates)}) must have the same length.")

    times = pd.to_datetime(list(dates))
    order = np.argsort(times.values)
    static_list = [static_list[i] for i in order]
    target_list = [target_list[i] for i in order]
    times = times[order]
    print(f"Outline order (by date): "
          + ", ".join(f"{t.date()}<-{s}" for t, s in zip(times, static_list)))

    # --- topographic field, computed ONCE from the first DEM ---
    ds0, dem0, mask0, y_dim, x_dim, res0 = load_2d_static(static_list[0])
    print(f"Reference DEM '{static_list[0]}': Y='{y_dim}', X='{x_dim}', "
          f"res ~{res0:.1f} m")
    print("Generating 2D SRF Map (once, shared across outlines)...")
    srf_2d = build_topographic_srf(filled_dem(dem0), res0)

    tmpl = xr.open_dataset(target_list[0])
    ty_dim, tx_dim = tmpl['HGT'].dims
    target_shape = tmpl['HGT'].shape

    srf_stack, np_stack = [], []

    for k, (spath, tpath, t) in enumerate(zip(static_list, target_list, times)):
        print(f"\n[{k + 1}/{n}] outline {t.date()}  static={spath}")
        ds_s, dem_s, mask_s, ys, xs, res_s = load_2d_static(spath)

        # DEM-identity check: openness reuse is only valid if HGT matches.
        if dem_s.shape != dem0.shape or not np.allclose(
                np.nan_to_num(dem_s), np.nan_to_num(dem0), atol=1e-3):
            print("    WARNING: this DEM differs from the reference DEM. "
                  "Re-computing openness for this outline.")
            srf_use = build_topographic_srf(filled_dem(dem_s), res_s)
        else:
            srf_use = srf_2d

        # mask-dependent steps (fast)
        srf_norm, _ = normalize_mass_conserving(srf_use, mask_s, lo=lo, hi=hi)

        ds_t = xr.open_dataset(tpath)
        levels = ds_t['HGT'].values
        srf_1d = aggregate_to_bands(srf_norm, dem_s, mask_s, levels, band_width)

        if 'N_Points' in ds_t.variables:
            npts = ds_t['N_Points'].values.astype(np.float64)
        else:
            print("    N_Points not in target; recomputing by pixel count.")
            npts = count_points_per_band(dem_s, mask_s, levels, band_width)

        # weighted-mean sanity check over the active glacier
        flat_srf, flat_n = srf_1d.flatten(), npts.flatten()
        good = np.isfinite(flat_srf) & np.isfinite(flat_n) & (flat_n > 0)
        if good.any():
            wmean = np.sum(flat_srf[good] * flat_n[good]) / np.sum(flat_n[good])
            print(f"    N_Points-weighted mean SRF = {wmean:.5f} "
                  f"(active bands: {int(good.sum())}, "
                  f"total points: {int(np.nansum(flat_n)):d})")

        srf_stack.append(srf_1d)
        np_stack.append(npts)
        ds_s.close(); ds_t.close()

    srf_stack = np.stack(srf_stack, axis=0)   # (time, y, x)
    np_stack = np.stack(np_stack, axis=0)

    # --- assemble merged static from the template geometry ---
    ds_out = tmpl.copy(deep=True)
    for v in ('SRF', 'N_Points'):
        if v in ds_out.variables:
            ds_out = ds_out.drop_vars(v)
    ds_out = ds_out.assign_coords(time=("time", times.values))

    ds_out['SRF'] = (("time", ty_dim, tx_dim), srf_stack)
    ds_out['SRF'].attrs = {
        'long_name': 'Norm. Snow Redistribution Factor (Hanzer 2016)',
        'units': '-',
        'note': f'Per-outline, mass-conserving on glacier, clipped to [{lo},{hi}]'}

    ds_out['N_Points'] = (("time", ty_dim, tx_dim), np_stack)
    ds_out['N_Points'].attrs = {'long_name': 'Number of Points in each bin',
                                'units': 'count'}

    print(f"\nSaving merged static (time={n}) to {merge_output}")
    ds_out.to_netcdf(merge_output)
    print("Done.")


# -----------------------------------------------------------------------------
# 4. CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute SRF for one or many glacier outlines.")
    parser.add_argument("-s", "--static2d", required=True, nargs="+",
                        help="30 m static file(s) with HGT + MASK.")
    parser.add_argument("-t", "--target1d", required=True, nargs="+",
                        help="1D band target file(s) (HGT, ideally N_Points).")
    parser.add_argument("-o", "--output",
                        help="[single mode] output 1D SRF file.")
    parser.add_argument("-o2", "--output2d",
                        help="[single mode] optional cropped 2D SRF map.")
    parser.add_argument("--merge-output", dest="merge_output",
                        help="[merge mode] single time-stamped static output.")
    parser.add_argument("--dates", nargs="+",
                        help="[merge mode] one date per outline, e.g. "
                             "--dates 2000-01-01 2013-01-01 2017-01-01.")
    parser.add_argument("-b", "--bandwidth", type=float, default=20.0)
    parser.add_argument("--clip-min", dest="clip_min", type=float, default=0.2)
    parser.add_argument("--clip-max", dest="clip_max", type=float, default=1.6)
    args = parser.parse_args()

    merge_mode = bool(args.merge_output) or len(args.static2d) > 1

    if merge_mode:
        if not args.merge_output:
            sys.exit("Merge mode needs --merge-output.")
        if not args.dates:
            sys.exit("Merge mode needs --dates (one per outline).")
        process_merge(args.static2d, args.target1d, args.dates,
                      args.merge_output, args.bandwidth,
                      args.clip_min, args.clip_max)
    else:
        if not args.output:
            sys.exit("Single mode needs -o/--output.")
        process_single(args.static2d[0], args.target1d[0], args.output,
                       args.output2d, args.bandwidth,
                       args.clip_min, args.clip_max)
