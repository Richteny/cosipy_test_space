import xarray as xr
import numpy as np
import pandas as pd
import argparse
import os
import sys
from numba import njit

# -----------------------------------------------------------------------------
# 1. Physics: OpenAmundsen Logic (Numba Optimized)  -- maths unchanged, race fixed
# -----------------------------------------------------------------------------

@njit
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

@njit(cache=True)
def _openness_dir(dem, res, L, dir):
    """Calculate openness for one direction.

    Serial on purpose: the earlier prange loop updated the running minimum
    from several threads at once (a race), so a larger angle could overwrite
    a smaller one and results changed from run to run (up to ~0.1 rad)."""
    opn_dir = np.full(dem.shape, np.inf)

    for i in range(int(np.ceil(L / res))):
        dist = res * (i + 1) * np.array([1, np.sqrt(2)])[dir % 2]
        Z_shift = _shift_arr_retain(dem, dir, i)

        # Angle from Zenith (Look down)
        angle = np.pi / 2 - np.arctan2(Z_shift - dem, dist)

        # Keep minimum angle
        opn_dir = np.minimum(opn_dir, angle)

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
    """Compute the *unnormalised* topographic SRF fields from the DEM only.

    Returns (combined, srf_50): the Hanzer combination 0.5*(50 m + 5 km term)
    and the 50 m (local curvature) term alone.

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

    return 0.5 * (srf_50 + srf_5000), srf_50


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


# -----------------------------------------------------------------------------
# 2b. Avalanche factor k_ava (Kneib et al. 2025, SnowSlide on the OGGM grids)
# -----------------------------------------------------------------------------
# One GeoTIFF per RGI6 glacier (<kneib_dir>/RGI13|14|15/**/*.tif), each on its
# own glacier-centred Transverse Mercator grid. SnowSlide runs on the whole
# DEM domain, so the tifs carry k_ava for all terrain around the glacier, not
# only on it. No RGI shapefiles are needed: the glacier is cut out by each
# outline's MASK (only mask pixels enter the band means), and the tifs that
# overlap the static grid are found from their own bounds. Where several tifs
# cover a pixel (same terrain, different domains), the one in which the pixel
# lies farthest from the domain edge wins. Not normalised: the glacier mean is
# > 1 where avalanches bring snow from the surrounding slopes onto the glacier.


def grid_transform(ds, y_dim, x_dim):
    """North-up affine transform of the lat/lon static grid, and whether the
    file's rows run south->north (then the 2D arrays are flipped)."""
    from rasterio.transform import from_origin
    lat, lon = ds[y_dim].values, ds[x_dim].values
    dx, dy = abs(lon[1] - lon[0]), abs(lat[1] - lat[0])
    return from_origin(lon.min() - dx / 2, lat.max() + dy / 2, dx, dy), lat[0] < lat[-1]


def kneib_index(kneib_dir):
    """lon/lat bounds of every Kneib tif. Scanning ~10^4-10^5 headers takes a
    few minutes, so the table is cached as kneib_index.csv in kneib_dir."""
    import glob, os
    import rasterio
    from rasterio.warp import transform_bounds
    cache = os.path.join(kneib_dir, "kneib_index.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    files = [f for f in glob.glob(os.path.join(kneib_dir, "RGI*", "**", "*.tif*"), recursive=True)
             if f.lower().endswith((".tif", ".tiff"))]      # skip .tif.aux.xml side files
    for f in sorted(files):
        with rasterio.open(f) as src:
            w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        rows.append((os.path.relpath(f, kneib_dir), w, s, e, n))
    if not rows:
        sys.exit(f"no k_ava tifs under {kneib_dir}/RGI*/")
    idx = pd.DataFrame(rows, columns=["file", "w", "s", "e", "n"])
    try:
        idx.to_csv(cache, index=False)
    except OSError:
        pass                                     # read-only folder: just rescan next time
    return idx


def kava_on_static(ds, y_dim, x_dim, glac, kneib_dir):
    """2D k_ava on the 30 m static grid (file orientation). `glac` (bool) is
    only used for the printout; pixels without any k_ava get 1."""
    import os
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject, Resampling

    transform, flip = grid_transform(ds, y_dim, x_dim)
    mask = glac[::-1] if flip else glac
    shape = mask.shape
    w, s, e, n = array_bounds(*shape, transform)
    lon = transform.c + transform.a * (np.arange(shape[1]) + 0.5)
    lat = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
    LON, LAT = np.meshgrid(lon, lat)
    coslat = np.cos(np.deg2rad(lat.mean()))

    idx = kneib_index(kneib_dir)
    cand = idx[(idx.w < e) & (idx.e > w) & (idx.s < n) & (idx.n > s)]
    kava = np.full(shape, np.nan)
    best = np.full(shape, -np.inf)               # distance to the edge of the tif in use
    owner = np.full(shape, -1)
    for i, row in enumerate(cand.itertuples()):
        with rasterio.open(os.path.join(kneib_dir, row.file)) as src:
            a = src.read(1).astype("float32")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            a[~np.isfinite(a) | (a < 0)] = np.nan
            dst = np.full(shape, np.nan, "float32")
            reproject(a, dst, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
        # SnowSlide misses snow from beyond its domain, so near a domain edge
        # k_ava is least reliable: take the tif in which the pixel lies deepest
        d = np.minimum(np.minimum(LON - row.w, row.e - LON) * coslat,
                       np.minimum(LAT - row.s, row.n - LAT))
        take = np.isfinite(dst) & (d > best)
        kava[take], best[take], owner[take] = dst[take], d[take], i
    print(f"    {len(cand)} Kneib tifs overlap the grid; used on the glacier:")
    for i, f in enumerate(cand.file):
        on = mask & (owner == i)
        if on.any():
            print(f"      {os.path.basename(f)}: {on.sum() / mask.sum():.0%} of the glacier pixels, "
                  f"mean {kava[on].mean():.3f}, max {kava[on].max():.2f}")
    print(f"    k_ava found on {np.mean(np.isfinite(kava[mask])):.0%} of the glacier pixels "
          f"(rest -> 1), glacier mean {np.nanmean(np.where(np.isfinite(kava), kava, 1.0)[mask]):.3f}")
    kava = np.where(np.isfinite(kava), kava, 1.0)
    return kava[::-1] if flip else kava


def to_ref_grid(spath, ds0, y_dim, x_dim):
    """HGT and MASK of a static on the reference grid of ds0. Identical grids
    are passed through; otherwise nearest neighbour within one pixel of the
    static's own grid (both are regular lat/lon grids). Prints how much of the
    outline's area lands on the reference grid."""
    ds = xr.open_dataset(spath)
    ys, xs = ds['HGT'].dims
    if (ys, xs) != (y_dim, x_dim):
        ds = ds.rename({ys: y_dim, xs: x_dim})
    m_own = (np.nan_to_num(ds['MASK'].values) == 1) if 'MASK' in ds else np.isfinite(ds['HGT'].values)
    same = (ds.sizes[y_dim] == ds0.sizes[y_dim] and ds.sizes[x_dim] == ds0.sizes[x_dim]
            and np.allclose(ds[y_dim], ds0[y_dim]) and np.allclose(ds[x_dim], ds0[x_dim]))
    if same:
        dem, mask = ds['HGT'].values, m_own
    else:
        tol = max(abs(float(ds[y_dim][1] - ds[y_dim][0])), abs(float(ds[x_dim][1] - ds[x_dim][0])))
        tgt = {y_dim: ds0[y_dim], x_dim: ds0[x_dim]}
        dem = ds['HGT'].reindex(tgt, method="nearest", tolerance=tol).values
        mask = xr.DataArray(m_own.astype(float), dims=(y_dim, x_dim),
                            coords={y_dim: ds[y_dim], x_dim: ds[x_dim]}
                            ).reindex(tgt, method="nearest", tolerance=tol).fillna(0).values == 1
        mask &= np.isfinite(dem)
        cell = lambda d: abs(float(d[y_dim][1] - d[y_dim][0]) * float(d[x_dim][1] - d[x_dim][0]))
        share = mask.sum() * cell(ds0) / (m_own.sum() * cell(ds))
        print(f"    {os.path.basename(spath)}: other grid {ds.sizes[y_dim]}x{ds.sizes[x_dim]} -> "
              f"matched onto the reference grid, {share:.1%} of its glacier area"
              + ("   ! outline reaches beyond the reference grid" if share < 0.98 else ""))
    ds.close()
    return dem, mask


def weighted_mean(field_1d, npts):
    """N_Points-weighted glacier mean of a band field (bands with N_Points > 0)."""
    f, w = field_1d.flatten(), npts.flatten()
    good = np.isfinite(f) & np.isfinite(w) & (w > 0)
    return np.sum(f[good] * w[good]) / np.sum(w[good]) if good.any() else np.nan


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
    srf_2d, srf50_2d = build_topographic_srf(filled_dem(dem_2d), res)

    srf_norm, _ = normalize_mass_conserving(srf_2d, mask_2d, lo=lo, hi=hi)
    srf50_norm, _ = normalize_mass_conserving(srf50_2d, mask_2d, lo=lo, hi=hi)

    if output_2d_path:
        crop_and_save_2d(ds_2d, srf_norm, output_2d_path, y_dim, x_dim)

    print("Aggregating to 1D Bands...")
    target_levels = ds_1d['HGT'].values
    srf_final = aggregate_to_bands(srf_norm, dem_2d, mask_2d,
                                   target_levels, band_width)
    srf50_final = aggregate_to_bands(srf50_norm, dem_2d, mask_2d,
                                     target_levels, band_width)

    ds_out = ds_1d.copy(deep=True)
    ds_out['SRF'] = (ds_1d['HGT'].dims, srf_final)
    ds_out['SRF'].attrs = {'long_name': 'Norm. Snow Redistribution Factor (Hanzer 2016)',
                           'units': '-'}
    ds_out['SRF_50'] = (ds_1d['HGT'].dims, srf50_final)
    ds_out['SRF_50'].attrs = {'long_name': 'Norm. SRF, 50 m term only (local curvature)',
                              'units': '-'}
    print(f"Saving 1D profile to {output_path}")
    ds_out.to_netcdf(output_path)
    print("Done.")


# -----------------------------------------------------------------------------
# 3b. Multi-outline merge -> single time-stamped static
# -----------------------------------------------------------------------------

def process_merge(static_list, target_list, dates, merge_output,
                  band_width, lo, hi, kneib_dir=None, output_2d=None):
    """Run all outlines and stack SRF (+ N_Points) along a sparse time axis.

    The openness/topographic field is computed ONCE from the first DEM, and
    normalised ONCE on the union of all outlines (largest extent); k_ava is
    reprojected once on the same extent. Per outline only the band aggregation
    over that outline's mask is redone; statics on other grids are matched
    onto the reference grid (nearest neighbour).

    Output is a single static with:
        SRF(time, y, x)       -- in [lo, hi], mean 1 over the union of outlines,
                                 band mean per outline
        K_AVA(time, y, x)     -- Kneib et al. 2025 avalanche factor, not
                                 normalised; only with kneib_dir. Formed in 2D
                                 so that band SRF x band K_AVA = band mean of
                                 the pixel product SRF*k_ava (see below)
        SRF_50(time, y, x)    -- the 50 m term alone, same normalisation as SRF
        K_AVA_50, K_AVA_PLAIN -- k_ava weighted by SRF_50 / plain band mean, for
                                 runs with the 50 m SRF / without SRF
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
    srf_2d, srf50_2d = build_topographic_srf(filled_dem(dem0), res0)

    # --- every outline's MASK and HGT on the reference grid (the earliest
    # static); statics on other grids are matched by nearest neighbour.
    # Reference extent = union of all outlines (largest extent).
    print("Putting all outlines on the reference grid...")
    on_ref = [to_ref_grid(spath, ds0, y_dim, x_dim) for spath in static_list]
    masks = [m for _, m in on_ref]
    ref_mask = np.any(masks, axis=0).astype(float)

    # --- k_ava, also ONCE, as a 2D field on the static grid
    if kneib_dir:
        print("Reprojecting k_ava (Kneib et al. 2025) onto the static grid...")
        kava_2d = kava_on_static(ds0, y_dim, x_dim, ref_mask == 1, kneib_dir)
    else:
        print("No --kneib-dir: K_AVA is not written (COSIPY then uses SRF alone).")
        kava_2d = None

    # --- SRF normalised ONCE, on the same reference extent: a pixel keeps its
    # factor when ice elsewhere disappears. Later (smaller) outlines therefore
    # have glacier means slightly off 1 -- the remaining ice is more or less
    # sheltered than the glacier as a whole was.
    print("Normalising SRF once on the union of outlines...")
    srf_norm_ref, _ = normalize_mass_conserving(srf_2d, ref_mask, lo=lo, hi=hi)
    print("Normalising SRF_50 (50 m term only) once on the union of outlines...")
    srf50_norm_ref, _ = normalize_mass_conserving(srf50_2d, ref_mask, lo=lo, hi=hi)

    if output_2d:                                # the 2D maps behind the bands, for checking
        ds2 = ds0[['HGT']].copy()
        ds2['MASK_UNION'] = (ds0['HGT'].dims, ref_mask)
        for k, m in enumerate(masks):
            ds2[f'MASK_{times[k].year}'] = (ds0['HGT'].dims, m.astype(float))
        ds2['SRF'] = (ds0['HGT'].dims, srf_norm_ref)
        ds2['SRF_50'] = (ds0['HGT'].dims, srf50_norm_ref)
        if kava_2d is not None:
            ds2['K_AVA'] = (ds0['HGT'].dims, kava_2d)
        print(f"Saving 2D maps to {output_2d}")
        ds2.to_netcdf(output_2d)

    tmpl = xr.open_dataset(target_list[0])
    ty_dim, tx_dim = tmpl['HGT'].dims
    target_shape = tmpl['HGT'].shape

    srf_stack, srf50_stack, kava_stack, kava50_stack, kavapl_stack, np_stack = [], [], [], [], [], []

    for k, (spath, tpath, t) in enumerate(zip(static_list, target_list, times)):
        print(f"\n[{k + 1}/{n}] outline {t.date()}  static={spath}")
        # this outline's own DEM decides the bands; SRF and k_ava are the
        # shared fields on the reference grid (openness from the reference DEM)
        dem_s, mask_b = on_ref[k]
        mask_s = mask_b.astype(float)
        srf_norm = srf_norm_ref

        ds_t = xr.open_dataset(tpath)
        levels = ds_t['HGT'].values
        srf_1d = aggregate_to_bands(srf_norm, dem_s, mask_s, levels, band_width)
        srf50_1d = aggregate_to_bands(srf50_norm_ref, dem_s, mask_s, levels, band_width)

        if 'N_Points' in ds_t.variables:
            npts = ds_t['N_Points'].values.astype(np.float64)
        else:
            print("    N_Points not in target; recomputing by pixel count.")
            npts = count_points_per_band(dem_s, mask_s, levels, band_width)

        # weighted-mean sanity check over the active glacier
        print(f"    N_Points-weighted mean SRF {weighted_mean(srf_1d, npts):.5f}, "
              f"SRF_50 {weighted_mean(srf50_1d, npts):.5f} "
              f"(active bands: {int(np.sum(npts > 0))}, "
              f"total points: {int(np.nansum(npts)):d})")

        if kava_2d is not None:
            # COSIPY multiplies the band values, but snow falls per pixel: the
            # band input must be mean(SRF*k_ava), not mean(SRF)*mean(k_ava)
            # (sheltered pixels can also be avalanche-fed). So the product is
            # formed in 2D, and K_AVA is stored as the SRF-weighted band mean
            # of k_ava; then band SRF x band K_AVA = band mean of the product.
            prod_1d = aggregate_to_bands(srf_norm * kava_2d, dem_s, mask_s, levels, band_width)
            kava_1d = prod_1d / srf_1d
            plain_1d = aggregate_to_bands(kava_2d, dem_s, mask_s, levels, band_width)
            print(f"    N_Points-weighted mean K_AVA {weighted_mean(kava_1d, npts):.4f} "
                  f"(plain band mean {weighted_mean(plain_1d, npts):.4f}), "
                  f"SRF x K_AVA {weighted_mean(prod_1d, npts):.4f}, "
                  f"band K_AVA max {np.nanmax(kava_1d):.3f}")
            kava_stack.append(kava_1d)
            # the same for the 50 m-only SRF, and the plain band mean for runs without SRF
            prod50_1d = aggregate_to_bands(srf50_norm_ref * kava_2d, dem_s, mask_s, levels, band_width)
            kava50_stack.append(prod50_1d / srf50_1d)
            kavapl_stack.append(plain_1d)
            print(f"    SRF_50 x K_AVA_50 {weighted_mean(prod50_1d, npts):.4f}")

        srf_stack.append(srf_1d)
        srf50_stack.append(srf50_1d)
        np_stack.append(npts)
        ds_t.close()

    srf_stack = np.stack(srf_stack, axis=0)   # (time, y, x)
    np_stack = np.stack(np_stack, axis=0)

    # --- assemble merged static from the template geometry ---
    ds_out = tmpl.copy(deep=True)
    for v in ('SRF', 'SRF_50', 'K_AVA', 'K_AVA_50', 'K_AVA_PLAIN', 'N_Points'):
        if v in ds_out.variables:
            ds_out = ds_out.drop_vars(v)
    ds_out = ds_out.assign_coords(time=("time", times.values))

    ds_out['SRF'] = (("time", ty_dim, tx_dim), srf_stack)
    ds_out['SRF'].attrs = {
        'long_name': 'Norm. Snow Redistribution Factor (Hanzer 2016)',
        'units': '-',
        'note': f'Mean 1 over the union of all outlines, clipped to [{lo},{hi}]; '
                'band mean over each outline'}

    ds_out['SRF_50'] = (("time", ty_dim, tx_dim), np.stack(srf50_stack, axis=0))
    ds_out['SRF_50'].attrs = {
        'long_name': 'Norm. SRF, 50 m term only (local curvature, Hanzer 2016 small scale)',
        'units': '-',
        'note': f'Same normalisation as SRF: mean 1 over the union of outlines, clipped to [{lo},{hi}]'}

    if kava_2d is not None:
        ds_out['K_AVA_50'] = (("time", ty_dim, tx_dim), np.stack(kava50_stack, axis=0))
        ds_out['K_AVA_50'].attrs = {
            'long_name': 'Avalanche factor k_ava (Kneib et al. 2025), SRF_50-weighted band mean',
            'units': '-', 'note': 'SRF_50 x K_AVA_50 = band mean of the pixel product; use with SRF_50'}
        ds_out['K_AVA_PLAIN'] = (("time", ty_dim, tx_dim), np.stack(kavapl_stack, axis=0))
        ds_out['K_AVA_PLAIN'].attrs = {
            'long_name': 'Avalanche factor k_ava (Kneib et al. 2025), plain band mean',
            'units': '-', 'note': 'use when no SRF is applied'}
        ds_out['K_AVA'] = (("time", ty_dim, tx_dim), np.stack(kava_stack, axis=0))
        ds_out['K_AVA'].attrs = {
            'long_name': 'Avalanche redistribution factor k_ava (Kneib et al. 2025)',
            'units': '-',
            'note': 'SRF-weighted band mean of k_ava over each outline, so that '
                    'SRF x K_AVA = band mean of the pixel product; not normalised '
                    '(glacier mean > 1 = net avalanche input). COSIPY applies SRF x K_AVA.'}

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
                        help="optional 2D maps (single: cropped SRF; merge: SRF, "
                             "K_AVA and the outline masks on the static grid).")
    parser.add_argument("--merge-output", dest="merge_output",
                        help="[merge mode] single time-stamped static output.")
    parser.add_argument("--dates", nargs="+",
                        help="[merge mode] one date per outline, e.g. "
                             "--dates 2000-01-01 2013-01-01 2017-01-01.")
    parser.add_argument("--kneib-dir", dest="kneib_dir",
                        help="[merge mode] Kneib et al. 2025 k_ava tifs (folder with "
                             "RGI13/, RGI14/, RGI15/); without it no K_AVA is written.")
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
                      args.clip_min, args.clip_max,
                      args.kneib_dir, args.output2d)
    else:
        if not args.output:
            sys.exit("Single mode needs -o/--output.")
        process_single(args.static2d[0], args.target1d[0], args.output,
                       args.output2d, args.bandwidth,
                       args.clip_min, args.clip_max)
