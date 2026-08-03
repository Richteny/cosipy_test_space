"""
Creates the ray-tracing based correction factors for incoming shortwave radiation
and the sky view factor (SVF) for diffuse shortwave correction.

Processing routine is based on the HORAYZON package:
    https://github.com/ChristianSteger/HORAYZON
Reference:
    Steger et al. (2022): https://gmd.copernicus.org/articles/15/6817/2022/

This script requires HORAYZON and xesmf:
    https://xesmf.readthedocs.io/en/stable/

Prerequisite: a static file with surrounding terrain, created via COSIPY utilities.
If regridding is desired, compute correction factors at high resolution first and
then regrid; do not compute directly on low-resolution static files.

Two outputs per run
-------------------
sw_dir_cor : time-varying (hourly, 2020) correction factor for DIRECT-BEAM
             shortwave radiation.  Produced by hray.shadow.Terrain.sw_dir_cor().
svf        : static sky view factor for DIFFUSE shortwave correction.
             Computed from hray.horizon.horizon_gridded() +
             hray.topo_param.sky_view_factor(), using slope_plane_meth
             (following the official HORAYZON examples for gridded curved DEMs).
             Only stored for glacier cells (MASK == 1); NaN elsewhere.

Usage:
    python -m cosipy.utilities.aws2cosipy.createHORAYZONfields \\
        -s <static_file> -o <output_file> \\
        [-c <coarse_static>] [-r <regrid>] [-e <elevation_profile>] \\
        [-es <band_size_m>] [-d <elev_static_out>] [-eb <elev_bins_file>]

Required arguments:
    -s / --static       Path to high-resolution static .nc file.
    -o / --output       Path for the resulting netCDF output file.

Optional arguments (defaults shown):
    -c / --coarse-static    Path to coarse static file for regridding [None].
    -r / --regridding       Regrid output to coarse grid [False].
    -e / --elevation_prof   Compute 1-D elevation band output [False].
    -es / --elevation_size  Elevation band width in metres [20].
    -d / --elev_data        Path for static elevation-band file (needs -e) [None].
    -eb / --elev_bins       Path to reference elevation-bin file (needs -e) [None].
"""

import argparse
import datetime as dt
import time

import horayzon as hray
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from skyfield.api import load, wgs84

from cosipy.utilities.config_utils import UtilitiesConfig

# ── Module-level globals (set by main()) ──────────────────────────────────────
_args = None
_cfg  = None

ELLPS = "WGS84"   # Earth-surface approximation

# Horizon search distance for SVF computation [km].
# 50 km is appropriate for Alpine terrain; increase for larger domains.
# azim_num is intentionally left at the HORAYZON default (360 directions).
SVF_DIST_SEARCH = 20.0  # horizon search distance [km]


# ═══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_variable_along_timelatlon(ds, var, name, units, long_name):
    ds[name] = (("time", "lat", "lon"), var)
    ds[name].attrs["units"]     = units
    ds[name].attrs["long_name"] = long_name
    return ds


def add_variable_along_latlon(ds, var, name, units, long_name):
    ds[name] = (("lat", "lon"), var)
    ds[name].attrs["units"]     = units
    ds[name].attrs["long_name"] = long_name
    ds[name].encoding["_FillValue"] = -9999
    return ds


def assign_attrs(ds, name, units, long_name):
    ds[name].attrs["units"]      = units
    ds[name].attrs["long_name"]  = long_name
    ds[name].attrs["_FillValue"] = -9999


def aspect_means(x):
    """Circular mean of aspect values [degrees]."""
    mean_sin = np.nanmean(np.sin(np.radians(x)))
    mean_cos = np.nanmean(np.cos(np.radians(x)))
    r        = np.sqrt(mean_cos**2 + mean_sin**2)
    return np.degrees(np.arctan2(mean_sin / r, mean_cos / r))


# ═══════════════════════════════════════════════════════════════════════════════
# ELEVATION-BIN HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_elev_bins(elev_bins_file):
    """Load reference elevation bins from an existing 1-D static file."""
    print(f"Loading reference bins from: {elev_bins_file}")
    holder        = xr.open_dataset(elev_bins_file)
    mask          = holder.MASK.values.flatten()
    raw_labels    = holder["HGT"].values.flatten()[mask == 1]
    sort_idx      = np.argsort(raw_labels)
    sorted_labels = raw_labels[sort_idx]
    band_size     = np.mean(np.diff(sorted_labels))
    bin_starts    = sorted_labels - band_size / 2.0
    bins          = np.append(bin_starts, bin_starts[-1] + band_size)
    ref_lats      = holder["lat"].values
    ref_lons      = holder["lon"].values
    holder.close()
    return band_size, bins, raw_labels, sorted_labels, ref_lats, ref_lons


def calculate_1d_elevationband(xds, elevation_var, mask_var, var_of_interest,
                                elev_bandsize, bins=None, sorted_labels=None,
                                raw_labels=None):
    """
    Aggregate a 2-D spatial field to 1-D elevation bands.

    Method by variable type:
        lat / lon   : arithmetic mean
        ASPECT      : circular mean (avoids 350 + 10 = 180 artefact)
        mask_var    : sum  (counts cells per band, i.e. N_Points)
        SLOPE       : arithmetic mean
        everything else (sw_dir_cor, svf, …) : median
    """
    xds = xds.where(xds[mask_var] == 1, drop=True)

    if bins is None or sorted_labels is None:
        print("Calculating bins automatically.")
        full_range    = xds[elevation_var].values[xds[mask_var] == 1]
        bins          = np.arange(np.nanmin(full_range),
                                  np.nanmax(full_range) + elev_bandsize,
                                  elev_bandsize)
        sorted_labels = bins[:-1] + elev_bandsize / 2
        raw_labels    = sorted_labels

    bins          = np.asarray(bins)
    sorted_labels = np.asarray(sorted_labels)

    if var_of_interest in ("lat", "lon"):
        result_sorted = np.array([
            np.nanmean(
                xds.where(
                    (xds[mask_var] == 1) &
                    (xds[elevation_var] >= bins[i]) &
                    (xds[elevation_var] <  bins[i] + elev_bandsize),
                    drop=True
                )[var_of_interest].values
            )
            for i in range(len(sorted_labels))
        ])

    elif var_of_interest == "ASPECT":
        elvs    = xds[elevation_var].values.flatten()[
                      xds[mask_var].values.flatten() == 1]
        aspects = xds[var_of_interest].values.flatten()[
                      xds[mask_var].values.flatten() == 1]
        result_sorted = np.array([
            aspect_means(aspects[np.logical_and(elvs >= bins[i],
                                                elvs <  bins[i] + elev_bandsize)])
            for i in range(len(sorted_labels))
        ])

    elif var_of_interest == mask_var:
        values        = xds[var_of_interest].groupby_bins(
            xds[elevation_var], bins, labels=sorted_labels, include_lowest=True
        ).sum(skipna=True, min_count=1)
        result_sorted = values.reindex(
            {"HGT_bins": sorted_labels}, fill_value=0
        ).fillna(0).values

    elif var_of_interest == "SLOPE":
        values        = xds[var_of_interest].groupby_bins(
            xds[elevation_var], bins, labels=sorted_labels, include_lowest=True
        ).mean(skipna=True)
        result_sorted = values.reindex({"HGT_bins": sorted_labels}).values

    else:
        # Default: median — for sw_dir_cor, svf, and any other numeric field
        values        = xds[var_of_interest].groupby_bins(
            xds[elevation_var], bins, labels=sorted_labels, include_lowest=True
        ).median(skipna=True)
        result_sorted = values.reindex({"HGT_bins": sorted_labels}).values

    result_sorted = np.array(result_sorted).flatten()

    # Re-align from sorted to raw (unsorted) label order
    if len(result_sorted) == len(raw_labels):
        indices = np.searchsorted(sorted_labels, raw_labels)
        return result_sorted[indices]

    print(f"Warning: length mismatch for '{var_of_interest}'. "
          f"Sorted={len(result_sorted)}, Raw={len(raw_labels)}. "
          "Returning sorted order.")
    return result_sorted


def construct_1d_dataset(df):
    """Convert the per-timestep elevation-band DataFrame to an xarray Dataset."""
    elev_ds = df.to_xarray()
    elev_ds.lon.attrs.update(standard_name="lon", long_name="longitude",
                              units="Average longitude of elevation bands")
    elev_ds.lat.attrs.update(standard_name="lat", long_name="latitude",
                              units="Average latitude of elevation bands")
    assign_attrs(elev_ds, "HGT",        "m",       "Mean elevation of band (m a.s.l.)")
    assign_attrs(elev_ds, "ASPECT",     "degrees", "Circular mean aspect")
    assign_attrs(elev_ds, "SLOPE",      "degrees", "Mean terrain slope")
    assign_attrs(elev_ds, "MASK",       "-",       "Glacier mask (1 = glacier)")
    assign_attrs(elev_ds, "N_Points",   "count",   "Number of 30-m source cells in band")
    assign_attrs(elev_ds, "sw_dir_cor", "-",
                 "Median direct-beam SW correction factor per elevation band")
    assign_attrs(elev_ds, "svf",        "-",
                 "Median sky view factor per elevation band (diffuse SW correction)")
    return elev_ds


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_and_slice(latitudes, longitudes, mask_obj):
    """
    Derive inner-domain slice and glacier-buffer mask.

    The ±11-cell buffer around the glacier ensures that regridding to coarser
    grids does not lose boundary cells.
    """
    slice_in = (slice(1, latitudes.shape[0] - 1),
                slice(1, longitudes.shape[0] - 1))

    mask_glacier_original                             = mask_obj.copy()
    mask_glacier_original[np.isnan(mask_glacier_original)] = 0
    mask_glacier                                      = mask_glacier_original.astype(bool)[slice_in]

    ilist, jlist = zip(*[
        (i, j)
        for i in range(mask_glacier.shape[0])
        for j in range(mask_glacier.shape[1])
        if mask_glacier[i, j]
    ])
    slice_buffer = (
        slice(np.min(ilist) - 11, np.max(ilist) + 11),
        slice(np.min(jlist) - 11, np.max(jlist) + 11),
    )
    mask_glacier[slice_buffer] = True
    return slice_in, slice_buffer, mask_glacier, mask_glacier_original


def compute_coords(lat, lon, elevation, slice_in):
    """
    Compute ECEF → ENU transformation and ENU vertex grid.

    Returns
    -------
    vert_grid        : padded vertex grid for both shadow and horizon modules
    vec_norm_enu     : surface-normal unit vectors, inner domain, global ENU
    vec_north_enu    : north-direction unit vectors, inner domain, global ENU
                       Caller must delete after use (needed for horizon_gridded).
    vec_tilt_enu     : global-ENU tilt from slope_vector_meth(output_rot=False)
                       → passed to hray.shadow.Terrain.initialise()
    trans_ecef2enu   : coordinate transformer (Skyfield sun positions)
    x_enu, y_enu, z_enu : full-domain ENU coords (needed for slope_plane_meth)
    rot_mat_glob2loc : rotation matrix global → local ENU
                       (needed for slope_plane_meth; caller deletes after use)
    """
    x_ecef, y_ecef, z_ecef = hray.transform.lonlat2ecef(
        *np.meshgrid(lon, lat), elevation, ellps=ELLPS
    )

    trans_ecef2enu = hray.transform.TransformerEcef2enu(
        lon_or=lon[len(lon) // 2], lat_or=lat[len(lat) // 2], ellps=ELLPS
    )
    x_enu, y_enu, z_enu = hray.transform.ecef2enu(
        x_ecef, y_ecef, z_ecef, trans_ecef2enu
    )

    vec_norm_ecef  = hray.direction.surf_norm(
        *np.meshgrid(lon[slice_in[1]], lat[slice_in[0]])
    )
    vec_north_ecef = hray.direction.north_dir(
        x_ecef[slice_in], y_ecef[slice_in], z_ecef[slice_in],
        vec_norm_ecef, ellps=ELLPS
    )
    del x_ecef, y_ecef, z_ecef

    vec_norm_enu  = hray.transform.ecef2enu_vector(vec_norm_ecef,  trans_ecef2enu)
    vec_north_enu = hray.transform.ecef2enu_vector(vec_north_ecef, trans_ecef2enu)
    del vec_norm_ecef, vec_north_ecef

    vert_grid = hray.auxiliary.rearrange_pad_buffer(x_enu, y_enu, z_enu)

    # Rotation matrix — kept in return value; caller deletes after slope_plane_meth
    rot_mat_glob2loc = hray.transform.rotation_matrix_glob2loc(
        vec_north_enu, vec_norm_enu
    )
    # vec_north_enu intentionally NOT deleted here — needed for horizon_gridded

    # Global-ENU tilt for hray.shadow.Terrain.initialise() — keep slope_vector_meth
    slice_in_a1 = (
        slice(slice_in[0].start - 1, slice_in[0].stop + 1),
        slice(slice_in[1].start - 1, slice_in[1].stop + 1),
    )
    vec_tilt_enu = np.ascontiguousarray(
        hray.topo_param.slope_vector_meth(
            x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
            rot_mat=rot_mat_glob2loc, output_rot=False
        )[1:-1, 1:-1]
    )

    return (vert_grid, vec_norm_enu, vec_north_enu, vec_tilt_enu,
            trans_ecef2enu, x_enu, y_enu, z_enu, rot_mat_glob2loc)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def merge_timestep_files(datasets, regrid, ds_coarse, static_ds,
                         elevation_profile, mask_glacier_original,
                         slice_in, slice_buffer):
    """Concatenate per-timestep datasets and merge with static fields."""
    t0        = time.time()
    ds_sw_cor = xr.concat(datasets, dim="time")
    ds_sw_cor["time"] = pd.to_datetime(ds_sw_cor["time"].values)

    if regrid:
        ds_sw_cor["MASK"] = ds_coarse["MASK"]
    else:
        if elevation_profile:
            for v in ("HGT", "ASPECT", "SLOPE", "MASK", "N_Points", "svf"):
                if v in ds_sw_cor:
                    ds_sw_cor[v] = ds_sw_cor[v].isel(time=0)
        else:
            mask_holder = mask_glacier_original[slice_in]
            add_variable_along_latlon(ds_sw_cor, mask_holder[slice_buffer],
                                      "MASK", "-", "Actual glacier mask")

    ds_sw_cor["MASK"] = (
        ("lat", "lon"),
        np.where(ds_sw_cor["MASK"] == 1, ds_sw_cor["MASK"], np.nan)
    )
    if not elevation_profile:
        ds_sw_cor = ds_sw_cor[["sw_dir_cor", "MASK"]]

    print(f"Concat took {time.time() - t0:.1f} s")

    if regrid:
        regridder = xe.Regridder(static_ds, ds_coarse[["HGT"]],
                                 method="conservative_normed")
        regridded = regridder(static_ds)
        # SVF was stored without a glacier mask so conservative regridding has
        # clean (non-NaN) inputs at glacier boundaries.  Apply mask now.
        if "svf" in regridded:
            regridded["svf"] = regridded["svf"].where(ds_coarse["MASK"] == 1)
        combined = xr.merge([ds_sw_cor, regridded])
    else:
        combined = (ds_sw_cor.copy() if elevation_profile
                    else xr.merge([ds_sw_cor, static_ds]))

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_horayzon_scheme(static_file, file_sw_dir_cor,
                        coarse_static_file=None, regrid=False,
                        elevation_profile=False, elev_bandsize=20,
                        elev_stat_file=None, elev_bins_file=None):
    """
    Compute sw_dir_cor (hourly, 2020) and SVF (static) from the static file,
    optionally aggregate to 1-D elevation bands or regrid to a coarser grid.
    """
    if elevation_profile:
        print(f"Elevation-profile mode: regrid forced off, "
              f"band size = {elev_bandsize} m.")
        regrid = False

    # ── Reference elevation bins ───────────────────────────────────────────
    ref_bins = ref_raw_labels = ref_sorted_labels = ref_lats = ref_lons = None
    if elev_bins_file is not None and elevation_profile:
        (elev_bandsize, ref_bins, ref_raw_labels,
         ref_sorted_labels, ref_lats, ref_lons) = load_elev_bins(elev_bins_file)

    # ── Load DEM ──────────────────────────────────────────────────────────
    ds                 = xr.open_dataset(static_file)
    elevation          = ds["HGT"].values.copy()
    elevation_original = ds["HGT"].values.copy()
    lon                = ds["lon"].values
    lat                = ds["lat"].values

    (slice_in, slice_buffer,
     mask_glacier, mask_glacier_original) = compute_and_slice(
        lat, lon, ds["MASK"].values
    )
    print(f"Inner domain size: {elevation[slice_in].shape}")

    elevation_ortho = np.ascontiguousarray(elevation[slice_in])
    elevation      += hray.geoid.undulation(lon, lat, geoid="EGM96")

    offset_0, offset_1   = slice_in[0].start, slice_in[1].start
    dem_dim_0, dem_dim_1 = elevation.shape

    # ── Coordinate setup ──────────────────────────────────────────────────
    (vert_grid, vec_norm_enu, vec_north_enu, vec_tilt_enu,
     trans_ecef2enu, x_enu, y_enu, z_enu,
     rot_mat_glob2loc) = compute_coords(lat, lon, elevation, slice_in)

    surf_enl_fac = 1.0 / (vec_norm_enu * vec_tilt_enu).sum(axis=2)
    print(f"Surface enlargement factor: "
          f"min={surf_enl_fac.min():.3f}, max={surf_enl_fac.max():.3f}")

    # Glacier mask for the inner domain (uint8 for HORAYZON)
    mask = np.ones(vec_tilt_enu.shape[:2], dtype=np.uint8)
    mask[~mask_glacier] = 0

    # ── Compute terrain horizon FIRST (before terrain.initialise) ─────────
    # Both hray.horizon.horizon_gridded and hray.shadow.Terrain build an
    # Embree BVH internally.  Running them sequentially (horizon first, then
    # shadow) avoids a segfault caused by two BVHs sharing the same vertex
    # data being alive simultaneously.
    #
    # azim_num is left at the HORAYZON default (360 directions).
    # Uses vec_norm_enu and vec_north_enu — deleted after this block.
    print(f"Computing terrain horizon (dist={SVF_DIST_SEARCH} km)...")
    t0 = time.time()
    hori, azim = hray.horizon.horizon_gridded(
        vert_grid, dem_dim_0, dem_dim_1,
        vec_norm_enu, vec_north_enu,
        offset_0, offset_1,
        SVF_DIST_SEARCH,
        mask=mask,
    )
    print(f"Horizon: {time.time() - t0:.1f} s")

    # ── Two local-ENU tilt vectors ────────────────────────────────────────
    # slope_plane_meth(output_rot=True) : used for SVF — matches HORAYZON examples
    # slope_vector_meth(output_rot=True): used for slope/aspect — matches old behaviour
    # Both need x_enu, y_enu, z_enu and rot_mat_glob2loc, so compute together.
    slice_in_a1 = (
        slice(slice_in[0].start - 1, slice_in[0].stop + 1),
        slice(slice_in[1].start - 1, slice_in[1].stop + 1),
    )
    vec_tilt_plane = np.ascontiguousarray(
        hray.topo_param.slope_plane_meth(
            x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
            rot_mat=rot_mat_glob2loc,
            output_rot=True,
        )[1:-1, 1:-1]
    )
    vec_tilt_vec = np.ascontiguousarray(
        hray.topo_param.slope_vector_meth(
            x_enu[slice_in_a1], y_enu[slice_in_a1], z_enu[slice_in_a1],
            rot_mat=rot_mat_glob2loc,
            output_rot=True,
        )[1:-1, 1:-1]
    )
    del rot_mat_glob2loc
    del vec_north_enu   # no longer needed after horizon_gridded
    del x_enu, y_enu, z_enu

    # ── Sky view factor (uses plane-method tilt) ──────────────────────────
    # Exact API from HORAYZON gridded_curved_DEM[_masked].py:
    #   svf = hray.topo_param.sky_view_factor(azim, hori, vec_tilt)
    print("Computing sky view factor...")
    t0       = time.time()
    svf_full = hray.topo_param.sky_view_factor(azim, hori, vec_tilt_plane)
    del hori, azim, vec_tilt_plane
    print(f"SVF: {time.time() - t0:.1f} s  |  inner domain "
          f"min={svf_full.min():.3f} mean={svf_full.mean():.3f} "
          f"max={svf_full.max():.3f}")

    # Glacier-masked SVF for:
    #   (a) diagnostic print
    #   (b) 1-D elevation-band aggregation (elevation_profile mode)
    # The full unmasked svf_full is used for the 2-D static_ds so that
    # conservative regridding is not biased by NaN at glacier boundaries.
    glacier_mask_buf = (mask_glacier_original[slice_in][slice_buffer] == 1)
    svf_glaciated    = np.where(glacier_mask_buf, svf_full[slice_buffer], np.nan)
    print(f"SVF (glacier only): min={np.nanmin(svf_glaciated):.3f} "
          f"mean={np.nanmean(svf_glaciated):.3f} "
          f"max={np.nanmax(svf_glaciated):.3f}")

    # ── Slope and aspect (uses vector-method tilt — matches old behaviour) ─
    slope  = np.arccos(vec_tilt_vec[:, :, 2].clip(max=1.0))
    aspect = np.pi / 2.0 - np.arctan2(vec_tilt_vec[:, :, 1],
                                        vec_tilt_vec[:, :, 0])
    aspect[aspect < 0.0] += 2.0 * np.pi   # wrap to [0, 2π]
    del vec_tilt_vec

    # ── Initialise shadow terrain (sw_dir_cor) ────────────────────────────
    # Runs AFTER horizon/SVF so only one Embree BVH is active at a time.
    # Uses vec_tilt_enu (global ENU, slope_vector_meth) — do not change.
    # vec_norm_enu still alive here (needed for terrain.initialise).
    terrain = hray.shadow.Terrain()
    terrain.initialise(
        vert_grid, dem_dim_0, dem_dim_1,
        offset_0, offset_1,
        vec_tilt_enu, vec_norm_enu,
        surf_enl_fac,
        mask=mask,
        elevation=elevation_ortho,
        refrac_cor=False,   # atmospheric refraction negligible at alpine elevation
    )
    # vec_norm_enu intentionally NOT deleted — terrain holds a raw C++ pointer
    # to the underlying array data and will segfault if the array is freed.
    # Python will garbage-collect it when run_horayzon_scheme returns.

    # ── Static 2-D dataset ────────────────────────────────────────────────
    static_ds = xr.Dataset()
    static_ds.coords["lat"] = lat[slice_buffer[0]]
    static_ds.coords["lon"] = lon[slice_buffer[1]]
    add_variable_along_latlon(static_ds, elevation_ortho[slice_buffer],
                              "elevation",    "m",      "Orthometric height")
    add_variable_along_latlon(static_ds, np.rad2deg(slope)[slice_buffer],
                              "slope",        "degree", "Terrain slope")
    add_variable_along_latlon(static_ds, np.rad2deg(aspect)[slice_buffer],
                              "aspect",       "degree", "Aspect (clockwise from N)")
    add_variable_along_latlon(static_ds, surf_enl_fac[slice_buffer],
                              "surf_enl_fac", "-",      "Surface enlargement factor")
    # Store FULL (unmasked) SVF so conservative regridding is not biased by
    # NaN values at glacier boundaries.  Glacier mask applied post-regrid in
    # merge_timestep_files.  For elevation-profile output the 1-D aggregation
    # already restricts to glacier cells via mask_real.
    add_variable_along_latlon(static_ds, svf_full[slice_buffer],
                              "svf",          "-",
                              "Sky view factor (diffuse SW correction)")
    del svf_full   # static_ds holds its own copy; safe to free now

    # ── Skyfield ──────────────────────────────────────────────────────────
    load.directory = _cfg.paths["static_folder"]
    planets        = load("de421.bsp")
    sun            = planets["sun"]
    earth          = planets["earth"]
    loc_or         = earth + wgs84.latlon(trans_ecef2enu.lat_or,
                                          trans_ecef2enu.lon_or)

    # ── Hourly time axis for 2020 UTC ─────────────────────────────────────
    time_dt_beg = dt.datetime(2020, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    time_dt_end = dt.datetime(2021, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    dt_step     = dt.timedelta(hours=1)
    num_ts      = int((time_dt_end - time_dt_beg) / dt_step)
    ta          = [time_dt_beg + dt_step * i for i in range(num_ts)]

    # ── Coarse grid / regridder ───────────────────────────────────────────
    ds_coarse = None
    if coarse_static_file is not None:
        ds_coarse         = xr.open_dataset(coarse_static_file)
        ds_coarse["mask"] = ds_coarse["MASK"]

    sw_dir_cor = np.zeros(vec_tilt_enu.shape[:2], dtype=np.float32)

    if regrid:
        ts           = load.timescale()
        t            = ts.from_datetime(ta[0])
        alt, az, d   = loc_or.at(t).observe(sun).apparent().altaz()
        sun_pos      = np.array([
            d.m * np.cos(alt.radians) * np.sin(az.radians),
            d.m * np.cos(alt.radians) * np.cos(az.radians),
            d.m * np.sin(alt.radians),
        ], dtype=np.float32)
        terrain.sw_dir_cor(sun_pos, sw_dir_cor)

        tmpl = xr.Dataset(
            coords=dict(time=[pd.to_datetime(ta[0])],
                        lat=lat[slice_buffer[0]],
                        lon=lon[slice_buffer[1]])
        )
        sw_h    = np.zeros((1, lat[slice_buffer[0]].shape[0],
                                lon[slice_buffer[1]].shape[0]))
        sw_h[0] = sw_dir_cor[slice_buffer]
        add_variable_along_timelatlon(tmpl, sw_h, "sw_dir_cor", "-",
                                      "Direct-beam SW correction factor")
        add_variable_along_latlon(tmpl, mask[slice_buffer], "mask", "-",
                                  "Glacier mask")
        regrid_fn = xe.Regridder(tmpl, ds_coarse, method="conservative_normed")
        tmpl.close()

    # ── Pre-compute static elevation-band fields (ONCE, before time loop) ──
    # SLOPE, ASPECT, lat, lon and SVF are time-invariant.  Computing them
    # inside the 8760-step loop was a significant performance bug.
    if elevation_profile:
        if ref_bins is not None:
            bins, raw_labels, sorted_labels = (ref_bins, ref_raw_labels,
                                               ref_sorted_labels)
        else:
            full_range    = ds["HGT"].values[ds["MASK"].values == 1]
            bins          = np.arange(np.nanmin(full_range),
                                      np.nanmax(full_range) + elev_bandsize,
                                      elev_bandsize)
            sorted_labels = bins[:-1] + elev_bandsize / 2
            raw_labels    = sorted_labels

        static_vars = ["SLOPE", "ASPECT"]
        if ref_lats is None or ref_lons is None:
            static_vars += ["lat", "lon"]

        static_placeholder = {
            v: calculate_1d_elevationband(
                ds, "HGT", "MASK", v, elev_bandsize,
                bins=bins, sorted_labels=sorted_labels, raw_labels=raw_labels
            )
            for v in static_vars
        }

        # SVF aggregation (static — uses the 2-D arrays already in memory)
        svf_tmp_ds = xr.Dataset(coords=dict(lat=lat[slice_buffer[0]],
                                             lon=lon[slice_buffer[1]]))
        add_variable_along_latlon(svf_tmp_ds, elevation_original[slice_in][slice_buffer],
                                  "HGT",       "m asl", "Surface elevation")
        add_variable_along_latlon(svf_tmp_ds, mask_glacier_original[slice_in][slice_buffer],
                                  "mask_real", "-",      "Actual glacier mask")
        add_variable_along_latlon(svf_tmp_ds, svf_glaciated, "svf", "-",
                                  "Sky view factor")
        static_placeholder["svf"] = calculate_1d_elevationband(
            svf_tmp_ds, "HGT", "mask_real", "svf", elev_bandsize,
            bins=bins, sorted_labels=sorted_labels, raw_labels=raw_labels,
        )
        svf_tmp_ds.close()

        # Resolve lat/lon for DataFrame index
        if ref_lats is not None and ref_lons is not None:
            final_lats    = ref_lats
            final_lons_val = np.mean(ref_lons)
        else:
            final_lats    = static_placeholder["lat"]
            final_lons_val = np.mean(static_placeholder["lon"])
        if hasattr(final_lons_val, "size") and final_lons_val.size == 1:
            final_lons_val = final_lons_val.item()
        elif isinstance(final_lons_val, list) and len(final_lons_val) == 1:
            final_lons_val = final_lons_val[0]

        if len(final_lats) != len(raw_labels):
            raise ValueError(
                f"Length mismatch: final_lats ({len(final_lats)}) vs "
                f"raw_labels ({len(raw_labels)}). Check elevation bins."
            )

    # ── Time loop: sw_dir_cor only ─────────────────────────────────────────
    datasets         = []
    comp_time_shadow = []

    for i, ta_i in enumerate(ta):
        t_beg = time.time()

        ts           = load.timescale()
        t            = ts.from_datetime(ta_i)
        alt, az, d   = loc_or.at(t).observe(sun).apparent().altaz()
        sun_pos      = np.array([
            d.m * np.cos(alt.radians) * np.sin(az.radians),
            d.m * np.cos(alt.radians) * np.cos(az.radians),
            d.m * np.sin(alt.radians),
        ], dtype=np.float32)

        terrain.sw_dir_cor(sun_pos, sw_dir_cor)
        comp_time_shadow.append(time.time() - t_beg)

        # Per-timestep dataset
        result = xr.Dataset(
            coords=dict(time=[pd.to_datetime(ta_i)],
                        lat=lat[slice_buffer[0]],
                        lon=lon[slice_buffer[1]])
        )
        sw_h    = np.zeros((1, lat[slice_buffer[0]].shape[0],
                                lon[slice_buffer[1]].shape[0]))
        sw_h[0] = sw_dir_cor[slice_buffer]
        add_variable_along_timelatlon(result, sw_h, "sw_dir_cor", "-",
                                      "Direct-beam SW correction factor")
        add_variable_along_latlon(result, mask[slice_buffer], "mask", "-",
                                  "Glacier mask")

        if elevation_profile:
            add_variable_along_latlon(result,
                                      elevation_original[slice_in][slice_buffer],
                                      "HGT",       "m asl", "Surface elevation")
            add_variable_along_latlon(result,
                                      mask_glacier_original[slice_in][slice_buffer],
                                      "mask_real", "-",      "Actual glacier mask")

            # Time-varying fields only (sw_dir_cor, N_Points)
            placeholder = dict(static_placeholder)
            for var in ("sw_dir_cor", "mask_real"):
                placeholder[var] = calculate_1d_elevationband(
                    result, "HGT", "mask_real", var, elev_bandsize,
                    bins=bins, sorted_labels=sorted_labels,
                    raw_labels=raw_labels,
                )

            df = pd.DataFrame({
                "lat":        final_lats,
                "lon":        final_lons_val,
                "time":       pd.to_datetime(ta_i),
                "HGT":        raw_labels,
                "ASPECT":     placeholder["ASPECT"],
                "SLOPE":      placeholder["SLOPE"],
                "MASK":       np.ones_like(final_lats),
                "N_Points":   placeholder["mask_real"],
                "sw_dir_cor": placeholder["sw_dir_cor"],
                "svf":        placeholder["svf"],
            })
            df["time"] = df["time"].dt.tz_localize(None)
            df.sort_values(by=["time", "lat", "lon"], inplace=True)

            try:
                df["lat"] = df["lat"] + df["HGT"] * 1e-9
                df.set_index(["time", "lat", "lon"], inplace=True)
                elev_ds = construct_1d_dataset(df)
            except Exception:
                df["lat"] = df["lat"] + df["HGT"] * 1e-8
                df.set_index(["time", "lat", "lon"], inplace=True)
                elev_ds = construct_1d_dataset(df)

        if regrid:
            datasets.append(regrid_fn(result))
        elif elevation_profile:
            datasets.append(elev_ds)
            elev_ds.close()
            del elev_ds, df
        else:
            datasets.append(result)

        result.close()
        del result

        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Timestep {i + 1}/{num_ts}")

    time_tot = np.sum(comp_time_shadow)
    print(f"Shadow: total={time_tot:.1f} s, per timestep={time_tot / num_ts:.4f} s")

    # ── Merge and write ───────────────────────────────────────────────────
    combined = merge_timestep_files(
        datasets=datasets, regrid=regrid, ds_coarse=ds_coarse,
        static_ds=static_ds, elevation_profile=elevation_profile,
        mask_glacier_original=mask_glacier_original,
        slice_in=slice_in, slice_buffer=slice_buffer,
    )

    if elevation_profile:
        combined.to_netcdf(file_sw_dir_cor)
        if elev_stat_file:
            combined[["HGT", "ASPECT", "SLOPE", "MASK",
                       "N_Points", "svf"]].to_netcdf(elev_stat_file)
    else:
        # Crop to minimal glacier bounding box
        valid_lats = (combined.MASK == 1).any(dim="lon")
        valid_lons = (combined.MASK == 1).any(dim="lat")
        lat_idx    = np.where(valid_lats.values)[0]
        lon_idx    = np.where(valid_lons.values)[0]

        if not np.any(np.diff(lat_idx) > 1) and not np.any(np.diff(lon_idx) > 1):
            combined = combined.where(combined.MASK == 1, drop=True)
        else:
            print("Gap in valid lat/lon extent — skipping drop=True crop.")
            combined = combined.where(combined.MASK == 1)

        combined.to_netcdf(file_sw_dir_cor)

    print(f"Output written → {file_sw_dir_cor}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def get_user_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.description = "Compute HORAYZON sw_dir_cor LUT and SVF for COSIPY."
    parser.prog        = __package__
    parser.add_argument("-s",  "--static",        dest="static_file",        type=str,      required=True)
    parser.add_argument("-o",  "--output",         dest="file_sw_dir_cor",    type=str,      required=True)
    parser.add_argument("-c",  "--coarse-static",  dest="coarse_static_file", type=str,      default=None)
    parser.add_argument("-r",  "--regridding",     dest="regrid",             type=str2bool, default=False)
    parser.add_argument("-e",  "--elevation_prof", dest="elevation_profile",  type=str2bool, default=False)
    parser.add_argument("-es", "--elevation_size", dest="elev_bandsize",      type=int,      default=20)
    parser.add_argument("-d",  "--elev_data",      dest="elev_stat_file",     type=str,      default=None)
    parser.add_argument("-eb", "--elev_bins",      dest="elev_bins_file",     type=str,      default=None)
    return parser.parse_args()


def load_config(module_name: str) -> tuple:
    params    = UtilitiesConfig()
    arguments = get_user_arguments(params.parser)
    params.load(arguments.utilities_path)
    params = params.get_config_expansion(name=module_name)
    return arguments, params


def main():
    global _args, _cfg
    _args, _cfg = load_config(module_name="create_static")
    run_horayzon_scheme(
        _args.static_file,
        _args.file_sw_dir_cor,
        _args.coarse_static_file,
        _args.regrid,
        _args.elevation_profile,
        _args.elev_bandsize,
        _args.elev_stat_file,
        _args.elev_bins_file,
    )


if __name__ == "__main__":
    main()
