"""
srf_weight_test.py -- which weight w for the 5 km SRF term do the observations support?

    SRF_w = normalise( (1 - w) * srf_50 + w * srf_5000 )
    w = 0   -> 50 m term only          (toposcale2cosipy --srf-variant 50m)
    w = 0.5 -> Hanzer et al. 2016       (current "combined" SRF)

Idea
----
Snow that lasts longer into late summer marks pixels that got more snow. Across
elevation, snow persistence is dominated by melt (temperature, radiation), so
the band-to-band profile cannot separate accumulation from ablation. WITHIN an
elevation band temperature is the same, so the pixel-to-pixel differences in
snow persistence carry the redistribution signal. The test

  1. demeans observation and predictors inside each elevation band,
  2. removes slope / aspect (and optional k_ava) from both, because shaded,
     north-facing pixels also keep snow longer for radiation reasons, and the
     elevation inside each band (own slope per band),
  3. finds the w whose SRF_w correlates best with what is left,
  4. gets an uncertainty from a spatial block bootstrap (tiles, not pixels,
     because neighbouring pixels are not independent).

A second estimate comes from one regression with both raw terms side by side:
    obs ~ a * srf_50 + b * srf_5000 (+ covariates)   ->   w_reg = b / (a + b)

Caveats (for the paper)
-----------------------
- Only bands around the late-summer snowline carry information; bands that are
  always snow-covered or always bare have no within-band contrast and drop out.
  The plateau bands where the 5 km term sits at the 0.2 floor are often among
  them -> the test constrains w by the horizontal pattern of the SAME field.
- Slope / aspect do not capture horizon shading by distant terrain, and that
  shading is correlated with the 5 km openness. What is left of it pushes w
  UP, so read the estimate as an upper bound for w.
- Observation: the QC-filtered Landsat / Sentinel-2 albedo scenes of the
  glacier (Landsat + Sentinel-2 NetCDF stacks), late ablation season only
  (MONTHS per glacier), all years. Three summaries are
  tested side by side: mean albedo, and the share of scenes with albedo above
  0.40 and 0.50 (snow/firn vs ice). If they agree, the threshold does not drive
  the answer.
- Albedo also carries dust and debris. Avalanche cones and headwall feet are
  concave AND get rockfall debris, which lowers albedo where the SRF is high
  -> this biases w DOWN for the 50 m term's pixels; k_ava as covariate takes
  out part of it.

Put this file next to createSRFmap.py (it reuses its functions). Outputs per
glacier in OUT_ROOT/<Cap>/, and srf_weight_summary_all.csv over all glaciers.
In a notebook: `import srf_weight_test as T; T.GLACIERS = ["Mera"]; T.run_all()`.
"""
import os
import sys

import numpy as np
import pandas as pd

# ============================================================================
# CONFIG
# ============================================================================
SRF_DIR    = "/data/scratch/richteny/thesis/cosipy_test_space/cosipy/utilities/aws2cosipy"
STATIC     = "/data/scratch/richteny/thesis/cosipy_test_space/data/static"
CONFIG_TXT = "/data/scratch/richteny/thesis/cosipy_test_space/cosipy/utilities/aws2cosipy/glacier_config.txt"
ALB_DIR    = "/data/scratch/richteny/albedo"         # {ALB_DIR}/{Cap}/{Cap}_LS|S2_filtered_albedos.nc
SENSORS    = ("LS", "S2")
ALB_VAR    = None                 # None -> 'albedo' if present, else the first variable with time
GLACIERS   = None                 # None = all in glacier_config.txt, or e.g. ["Mera", "Dongkemadi"]
# late ablation season per glacier (month numbers) -- PLEASE CHECK
MONTHS = {
    "Abramov": (8, 9), "Bordu": (8, 9), "Urumqi": (8, 9), "Guliya": (8, 9),
    "InteriorGlacier": (8, 9), "Dongkemadi": (8, 9), "Halji": (8, 9, 10),
    "Kolahoi": (8, 9), "Mera": (9, 10, 11), "ParlungNo4": (9, 10),
}
# HORAYZON 30 m LUT (cropped to the glacier): {Cap}_{outline}_HORAYZON-LUT_30m.nc
# Its sky-view factor and mean terrain correction of direct SW are added as
# shading covariates in a second pass ("+HORAYZON"), to see whether the 5 km
# effect is snow redistribution or only shading by the surrounding ridges.
HRZ_SVF, HRZ_SWCOR = "svf", "sw_dir_cor"
HRZ_MONTHS = (5, 6, 7, 8, 9)      # months for the mean sw_dir_cor (melt season)
MIN_SCENES = 5                    # pixels with fewer valid scenes are dropped
OBS_VARIANTS = [("mean albedo", None), ("share > 0.40", 0.40), ("share > 0.50", 0.50)]
KNEIB_DIR  = "/data/scratch/richteny/Kneib_HMA"      # None -> k_ava not used as covariate
OUT_ROOT   = "/data/scratch/richteny/srf_weight_test"
# set per glacier by set_glacier()
GLACIER, STATIC_2D, OUT_DIR, HRZ_LUT = None, None, None, None

BAND_M        = 20.0      # elevation band width (m), as in the 1D setup
MIN_PX_BAND   = 20        # bands with fewer glacier pixels are dropped
SNOWLINE_SHARE = (0.15, 0.85)  # bands used: band-mean share of snow scenes (albedo > SNOW_ALB)
SNOW_ALB      = 0.45      # in this range = the late-summer snowline zone; the SAME bands
                          # are used for every observation summary
RANK_IN_BAND  = True      # use within-band ranks of the observation (robust to saturation)
CLIP          = (0.2, 1.6)
W_GRID        = np.round(np.arange(0.0, 1.0001, 0.05), 2)
TILE_M        = 500.0     # bootstrap block size (m)
N_BOOT        = 1000
SEED          = 42
# psi coefficients exactly as in createSRFmap.build_topographic_srf
PSI_50, PSI_5000 = (3.0, 1.2), (3.0, 1.0)
# ============================================================================

sys.path.insert(0, SRF_DIR)
import createSRFmap as cs  # noqa: E402


def raw_terms(dem, res):
    """Clipped raw 50 m and 5000 m terms (before averaging / normalising)."""
    n50 = cs.openness(dem, res, L=50.0, negative=True)
    n5k = cs.openness(dem, res, L=5000.0, negative=True)
    s50 = np.clip(PSI_50[0] * (n50 - PSI_50[1]), 0.0, 5.0)
    s5k = np.clip(PSI_5000[0] * (n5k - PSI_5000[1]), 0.0, 5.0)
    return s50, s5k


def slope_aspect(dem, res):
    gy, gx = np.gradient(dem, res)          # rows: file order, sign fixed below
    return gy, gx


def set_glacier(cap, outline):
    """Point the module-level paths at one glacier (latest outline = mask)."""
    global GLACIER, STATIC_2D, OUT_DIR, HRZ_LUT
    GLACIER = cap
    f = f"{STATIC}/{cap}/{cap}_{outline}_static_30m_new.nc"
    if not os.path.exists(f):                       # Halji2014_static_30m_new.nc
        f = f"{STATIC}/{cap}/{cap}{outline}_static_30m_new.nc"
    STATIC_2D = f
    f = f"{STATIC}/{cap}/{cap}_{outline}_HORAYZON-LUT_30m.nc"
    if not os.path.exists(f):
        f = f"{STATIC}/{cap}/{cap}{outline}_HORAYZON-LUT_30m.nc"
    HRZ_LUT = f
    OUT_DIR = f"{OUT_ROOT}/{cap}"


def read_config():
    """(cap, latest outline) per glacier from glacier_config.txt."""
    rows = []
    for line in open(CONFIG_TXT):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        outl = p[4] if p[5] == "-" else p[5].split(",")[-1]
        rows.append((p[1], outl))
    return rows


def _cell_index(src, tgt, step):
    """Nearest index of each src coordinate in tgt; -1 if more than half a cell away."""
    order = np.argsort(tgt)
    pos = np.clip(np.searchsorted(tgt[order], src), 1, len(tgt) - 1)
    lo, hi = order[pos - 1], order[pos]
    idx = np.where(np.abs(src - tgt[lo]) <= np.abs(src - tgt[hi]), lo, hi)
    idx[np.abs(src - tgt[idx]) > 0.5 * step] = -1
    return idx


def _cos_zenith(times, lat, lon):
    """Cosine of the solar zenith angle (NOAA approximation), times in UTC."""
    t = pd.DatetimeIndex(times)
    doy = t.dayofyear.values
    hr = t.hour.values + t.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.deg2rad((hr * 60 + eqt + 4 * lon) / 4.0 - 180.0)
    la = np.deg2rad(lat)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)


def horayzon_on_grid(ds, y_dim, x_dim):
    """svf and the melt-season mean of sw_dir_cor on the static grid (NaN
    outside the cropped LUT). sw_dir_cor is averaged over daylight steps,
    weighted by cos(zenith), so low-sun hours with extreme ratios do not
    dominate. Returns {} if the LUT is missing."""
    import xarray as xr
    if not HRZ_LUT or not os.path.exists(HRZ_LUT):
        print(f"  (no HORAYZON LUT: {HRZ_LUT}) -> '+HORAYZON' pass skipped")
        return {}
    d = xr.open_dataset(HRZ_LUT)
    ylat, xlon = ds[y_dim].values, ds[x_dim].values
    ny, nx = len(ylat), len(xlon)
    ref = d[HRZ_SVF] if HRZ_SVF in d else d[HRZ_SWCOR]
    la = next(c for c in ref.dims if c.lower().startswith("lat") or c == "y")
    lo = next(c for c in ref.dims if c.lower().startswith("lon") or c == "x")
    iy = _cell_index(d[la].values, ylat, abs(ylat[1] - ylat[0]))
    ix = _cell_index(d[lo].values, xlon, abs(xlon[1] - xlon[0]))

    def to_grid(a2d):
        out = np.full((ny, nx), np.nan)
        IY, IX = np.meshgrid(iy, ix, indexing="ij")
        m = (IY >= 0) & (IX >= 0)
        out[IY[m], IX[m]] = a2d[m]
        return out

    res = {}
    if HRZ_SVF in d:
        sv = d[HRZ_SVF]
        extra = [c for c in sv.dims if c not in (la, lo)]
        if extra:                                      # e.g. a time axis: take the mean
            sv = sv.mean(extra)
        res["svf"] = to_grid(sv.transpose(la, lo).values.astype(float))
    if HRZ_SWCOR in d:
        sw = d[HRZ_SWCOR]
        tdim = [c for c in sw.dims if c not in (la, lo)]
        if len(tdim) == 1:
            tdim = tdim[0]
            sw = sw.transpose(tdim, la, lo)
            tv = sw[tdim].values
            if np.issubdtype(tv.dtype, np.datetime64):
                months = pd.DatetimeIndex(tv).month
                keep = np.isin(months, HRZ_MONTHS)
                cz = _cos_zenith(tv, float(np.mean(ylat)), float(np.mean(xlon)))
                wts = np.where(keep & (cz > 0.05), cz, 0.0)
                how = f"months {HRZ_MONTHS}, cos-zenith weighted"
            else:                                      # no dates: plain mean of daylight steps
                wts = np.ones(len(tv))
                how = "all steps (no dates in the LUT), unweighted"
            num = np.zeros(sw.shape[1:])
            den = np.zeros(sw.shape[1:])
            for k in np.where(wts > 0)[0]:
                a = sw.isel({tdim: int(k)}).values.astype(float)
                good = np.isfinite(a) & (a > 0)
                num[good] += wts[k] * a[good]
                den[good] += wts[k]
            with np.errstate(invalid="ignore", divide="ignore"):
                res["sw_dir_cor"] = to_grid(num / den)
            print(f"  HORAYZON: svf + mean sw_dir_cor over {int((wts > 0).sum())} steps ({how})")
        elif len(tdim) == 0:
            res["sw_dir_cor"] = to_grid(sw.transpose(la, lo).values.astype(float))
    d.close()
    return res


def albedo_stack(ds, y_dim, x_dim):
    """All scenes (LS + S2) of the glacier's late-season months, each scene
    averaged onto the 30 m static grid (finer S2 pixels are pooled per cell,
    so every scene counts once per cell)."""
    import xarray as xr
    months = MONTHS[GLACIER]
    ylat, xlon = ds[y_dim].values, ds[x_dim].values
    ny, nx = len(ylat), len(xlon)
    dy, dx = abs(ylat[1] - ylat[0]), abs(xlon[1] - xlon[0])

    cell_index = _cell_index

    stack, info = [], []
    for sen in SENSORS:
        f = f"{ALB_DIR}/{GLACIER}/{GLACIER}_{sen}_filtered_albedos.nc"
        if not os.path.exists(f):
            print(f"  (missing, skipped) {f}")
            continue
        d = xr.open_dataset(f)
        var = ALB_VAR or ("albedo" if "albedo" in d else
                          next(v for v in d.data_vars if "time" in d[v].dims))
        da = d[var]
        la = next(c for c in da.dims if c.lower().startswith("lat") or c == "y")
        lo = next(c for c in da.dims if c.lower().startswith("lon") or c == "x")
        da = da.sel(time=da["time"].dt.month.isin(months)).transpose("time", la, lo)
        iy = cell_index(da[la].values, ylat, dy)
        ix = cell_index(da[lo].values, xlon, dx)
        IY, IX = np.meshgrid(iy, ix, indexing="ij")
        inside = (IY >= 0) & (IX >= 0)
        cell = (IY * nx + IX)[inside]
        for t in range(da.sizes["time"]):
            a = da.isel(time=t).values.astype("float64")[inside]
            good = np.isfinite(a)
            s_ = np.bincount(cell[good], a[good], minlength=ny * nx)
            n_ = np.bincount(cell[good], minlength=ny * nx)
            with np.errstate(invalid="ignore", divide="ignore"):
                stack.append((s_ / n_).reshape(ny, nx).astype("float32"))
        yrs = da["time"].dt.year.values
        info.append(f"{sen} {da.sizes['time']} scenes"
                    + (f" {yrs.min()}-{yrs.max()}" if len(yrs) else ""))
        d.close()
    if not stack:
        sys.exit(f"No albedo scenes for {GLACIER} in months {months}.")
    stack = np.stack(stack)
    mx = np.nanpercentile(stack, 99)
    if mx > 150:
        stack /= 10000.0
    elif mx > 1.5:
        stack /= 100.0
    stack[(stack < 0) | (stack > 1.2)] = np.nan
    print(f"  albedo, months {months}: " + ", ".join(info)
          + f"; on the glacier grid: {np.isfinite(stack).any(axis=(1, 2)).sum()} scenes")
    return stack


def summarise(stack, thr):
    """Mean albedo (thr None) or share of scenes above thr, per pixel."""
    n = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore"):
        if thr is None:
            out = np.nanmean(stack, axis=0)
        else:
            out = np.nansum(stack >= thr, axis=0) / np.where(n > 0, n, np.nan)
    out[n < MIN_SCENES] = np.nan
    return out


def demean(x, g):
    return x - pd.Series(x).groupby(g).transform("mean").values


def residualise(y, X):
    if X.shape[1] == 0:
        return y
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def mix_weight(a, b):
    """w = b / (a + b); a term with a non-positive effect gets weight 0."""
    if a > 0 and b > 0:
        return b / (a + b)
    if a > 0:
        return 0.0
    if b > 0:
        return 1.0
    return np.nan


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def prepare():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"== {GLACIER}  ({os.path.basename(STATIC_2D)})")
    ds, dem2d, mask2d, y_dim, x_dim, res = cs.load_2d_static(STATIC_2D)
    dem = cs.filled_dem(dem2d)
    glac = (mask2d == 1) & np.isfinite(dem2d)
    print(f"  grid {dem.shape}, {res:.1f} m, {glac.sum()} glacier pixels")

    # extent check: the 5 km scan needs >= 5 km of terrain around the glacier
    rows, cols = np.where(glac)
    margin = res * np.array([rows.min(), dem.shape[0] - 1 - rows.max(),
                             cols.min(), dem.shape[1] - 1 - cols.max()])
    print(f"  terrain margin around the glacier (N/S/W/E order of the file): "
          + ", ".join(f"{m/1000:.1f} km" for m in margin)
          + ("   ! < 5 km: 5 km openness truncated at the edge" if margin.min() < 5000 else ""))

    print("  openness 50 m / 5000 m ...")
    s50, s5k = raw_terms(dem, res)

    # covariates: slope, northness, eastness (+ log k_ava)
    gy, gx = slope_aspect(dem, res)
    lat = ds[y_dim].values
    if lat[0] < lat[-1]:                     # rows run south->north
        gy = -gy
    # gy is now d(z)/d(row going south); north-facing = surface drops to the north
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)             # 0 = north, pi/2 = east
    cov = {"slope": slope, "northness": np.cos(aspect) * np.sin(slope),
           "eastness": np.sin(aspect) * np.sin(slope)}
    if KNEIB_DIR:
        kava = cs.kava_on_static(ds, y_dim, x_dim, glac, KNEIB_DIR)
        cov["log_kava"] = np.log(np.clip(kava, 0.05, None))
    hrz = horayzon_on_grid(ds, y_dim, x_dim)
    return dict(ds=ds, y_dim=y_dim, x_dim=x_dim, res=res, dem2d=dem2d, glac=glac,
                s50=s50, s5k=s5k, cov=cov, lat=lat, hrz=hrz)


def test(ctx, obs, label, use_hrz=False):
    """Weight test for one observation summary. use_hrz adds the HORAYZON
    shading covariates. Both passes use the same pixels (those inside the
    HORAYZON LUT), so they differ only in the covariates."""
    res, dem2d, glac, s50, s5k, lat = (ctx[k] for k in
                                       ("res", "dem2d", "glac", "s50", "s5k", "lat"))
    cov = dict(ctx["cov"])
    hrz_ok = np.ones_like(glac, bool)
    for k, v in ctx["hrz"].items():
        hrz_ok &= np.isfinite(v)
        if use_hrz:
            cov[k] = v
    covset = "+HORAYZON" if use_hrz else "local"
    tag = (label.replace(" ", "").replace(">", "gt").replace(".", "")
           + ("_hrz" if use_hrz else ""))
    print(f"\n--- {label}  [covariates: {covset}] ---")
    ok = glac & np.isfinite(obs) & np.isfinite(ctx["snow_share"]) & hrz_ok
    band = np.floor(dem2d[ok] / BAND_M).astype(int)
    y = obs[ok].astype(float)
    sc = ctx["snow_share"][ok]
    stats = pd.DataFrame({"band": band, "sc": sc}).groupby("band")["sc"].agg(["size", "mean"])
    keep_b = stats.index[(stats["size"] >= MIN_PX_BAND) & (stats["mean"] >= SNOWLINE_SHARE[0])
                         & (stats["mean"] <= SNOWLINE_SHARE[1])]
    sel = np.isin(band, keep_b)
    print(f"  {ok.sum()} pixels with observation; snowline bands: {len(keep_b)} of "
          f"{len(stats)} ({sel.sum()} px, {BAND_M * keep_b.min():.0f}-{BAND_M * (keep_b.max() + 1):.0f} m)")
    if sel.sum() < 200:
        print("  too few pixels in the snowline bands -- skipped (widen SNOWLINE_SHARE?)")
        return None

    g = band[sel]
    if RANK_IN_BAND:
        # normal scores of the within-band ranks: robust to the S-shaped,
        # saturating response of snow cover / albedo near 0 and 1
        from scipy.stats import norm
        rk = pd.Series(y[sel]).groupby(g).rank(method="average").values
        nb = pd.Series(g).map(pd.Series(g).value_counts()).values
        ya = demean(norm.ppf((rk - 0.5) / nb), g)
    else:
        ya = demean(y[sel], g)
    # elevation inside the band, with its own slope per band: near the snowline
    # snow cover changes fast with z, and the 5 km term also falls with z, so
    # without this the +-10 m inside a band leak in as a fake 5 km effect
    za = demean(dem2d[ok][sel].astype(float), g)
    Z = np.column_stack([np.where(g == bb, za, 0.0) for bb in keep_b])
    C = np.column_stack([demean(v[ok][sel], g) for v in cov.values()] + [Z])
    ry = residualise(ya, C)

    # tiles for the block bootstrap
    tpx = max(1, int(round(TILE_M / res)))
    rr, cc = np.where(ok)
    tile = (rr[sel] // tpx) * 100000 + (cc[sel] // tpx)
    utile, tinv = np.unique(tile, return_inverse=True)
    print(f"  bootstrap: {len(utile)} tiles of {tpx * res:.0f} m")

    # --- grid over w --------------------------------------------------------
    RX = []
    for w in W_GRID:
        srf, _ = cs.normalize_mass_conserving((1 - w) * s50 + w * s5k, glac.astype(int),
                                              CLIP[0], CLIP[1], verbose=False)
        xa = demean(np.log(srf[ok][sel]), g)
        RX.append(residualise(xa, C))
    RX = np.array(RX)
    r_w = np.array([corr(rx, ry) for rx in RX])
    w_best = W_GRID[np.argmax(r_w)]

    # --- regression with both raw terms --------------------------------------
    A = np.column_stack([demean(s50[ok][sel], g), demean(s5k[ok][sel], g), C])
    coef, *_ = np.linalg.lstsq(A, ya, rcond=None)
    a, b = coef[0], coef[1]
    w_reg = mix_weight(a, b)

    # --- block bootstrap ------------------------------------------------------
    rng = np.random.default_rng(SEED)
    idx_by_tile = [np.where(tinv == k)[0] for k in range(len(utile))]
    w_boot, wreg_boot, r_boot, a_boot, b_boot = [], [], [], [], []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(utile), len(utile))
        ii = np.concatenate([idx_by_tile[k] for k in pick])
        rb = np.array([corr(RX[j, ii], ry[ii]) for j in range(len(W_GRID))])
        r_boot.append(rb)
        w_boot.append(W_GRID[np.argmax(rb)])
        cb, *_ = np.linalg.lstsq(A[ii], ya[ii], rcond=None)
        wreg_boot.append(mix_weight(cb[0], cb[1]))
        a_boot.append(cb[0])
        b_boot.append(cb[1])
    w_boot, wreg_boot, r_boot = np.array(w_boot), np.array(wreg_boot), np.array(r_boot)
    p05, p50, p95 = np.percentile(w_boot, [5, 50, 95])
    j0, j5 = 0, int(np.argmin(np.abs(W_GRID - 0.5)))
    p_50_better = np.mean(r_boot[:, j0] > r_boot[:, j5])

    # --- per-band partial correlations (where does the information sit?) -----
    rows_b = []
    for bb in keep_b:
        m = g == bb
        if m.sum() < MIN_PX_BAND:
            continue
        Cb = C[m] - C[m].mean(0)
        Cb = Cb[:, np.abs(Cb).sum(0) > 0]
        ryb = residualise(ya[m], Cb)
        rows_b.append({"z": BAND_M * (bb + 0.5), "n_px": int(m.sum()),
                       "obs_mean": float(y[sel][m].mean()),
                       "r_50": corr(residualise(demean(s50[ok][sel][m], np.zeros(m.sum())), Cb), ryb),
                       "r_5k": corr(residualise(demean(s5k[ok][sel][m], np.zeros(m.sum())), Cb), ryb)})
    per_band = pd.DataFrame(rows_b)

    # --- report ----------------------------------------------------------------
    print("\n  partial r (within bands, after " + ", ".join(cov) + ", z) by w:")
    for w, r in zip(W_GRID, r_w):
        print(f"    w={w:4.2f}  r={r:+.3f}" + ("   <- best" if w == w_best else ""))
    print(f"\n  best w (grid):        {w_best:.2f}   bootstrap 5-95 %: {p05:.2f}-{p95:.2f} (median {p50:.2f})")
    print(f"  w from regression:    {w_reg:.2f}   (a_50={a:+.3f}, b_5k={b:+.3f}; "
          f"5-95 %: {np.nanpercentile(wreg_boot, 5):.2f}-{np.nanpercentile(wreg_boot, 95):.2f}, "
          f"{np.mean(np.isnan(wreg_boot)):.0%} of draws with both terms <= 0)")
    print(f"  effect of each term, 5-95 %:  50 m {np.percentile(a_boot, 5):+.2f} to "
          f"{np.percentile(a_boot, 95):+.2f},  5 km {np.percentile(b_boot, 5):+.2f} to "
          f"{np.percentile(b_boot, 95):+.2f}   (range above 0 = supported)")
    print(f"  P(50 m only beats w=0.5): {p_50_better:.2f}")
    print("  note: w is an upper bound (distant-horizon shading not controlled).")

    pd.DataFrame({"w": W_GRID, "r": r_w,
                  "r_p05": np.percentile(r_boot, 5, axis=0),
                  "r_p95": np.percentile(r_boot, 95, axis=0)}
                 ).to_csv(f"{OUT_DIR}/{GLACIER}_srf_weight_curve_{tag}.csv", index=False)
    per_band.to_csv(f"{OUT_DIR}/{GLACIER}_srf_weight_per_band_{tag}.csv", index=False)
    summary = pd.DataFrame([{"glacier": GLACIER, "observation": label, "covariates": covset, "n_px": int(sel.sum()), "n_tiles": len(utile),
                             "n_bands": len(keep_b), "w_best": w_best, "w_p05": p05, "w_p50": p50,
                             "w_p95": p95, "w_reg": w_reg, "a_50": a, "b_5k": b,
                             "a_50_p05": np.percentile(a_boot, 5), "a_50_p95": np.percentile(a_boot, 95),
                             "b_5k_p05": np.percentile(b_boot, 5), "b_5k_p95": np.percentile(b_boot, 95),
                             "r_at_0": r_w[j0], "r_at_0.5": r_w[j5], "r_best": r_w.max(),
                             "p_50m_better_than_0.5": p_50_better}])

    # --- figure -----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    ax[0].fill_between(W_GRID, np.percentile(r_boot, 5, 0), np.percentile(r_boot, 95, 0),
                       color="0.85", label="bootstrap 5-95 %")
    ax[0].plot(W_GRID, r_w, "k-o", ms=3, label="partial r")
    ax[0].axvline(0.5, color="tab:red", ls=":", label="Hanzer (w = 0.5)")
    ax[0].axvspan(p05, p95, color="tab:blue", alpha=0.15, label="best w, 5-95 %")
    ax[0].set(xlabel="weight of the 5 km term w", ylabel="within-band partial r",
              title=f"{GLACIER}: {label}, {covset}")
    ax[0].legend(fontsize=7)
    shown = np.where(ok, obs, np.nan)
    im = ax[1].imshow(shown[::-1] if lat[0] < lat[-1] else shown, cmap="Blues",
                      vmin=np.nanpercentile(shown, 2), vmax=np.nanpercentile(shown, 98))
    plt.colorbar(im, ax=ax[1], shrink=0.8, label=label)
    ax[1].set(title=f"months {MONTHS[GLACIER]}", xticks=[], yticks=[])
    if len(per_band):
        ax[2].plot(per_band.r_50, per_band.z, "-o", ms=3, label="50 m term")
        ax[2].plot(per_band.r_5k, per_band.z, "-o", ms=3, label="5 km term")
        ax[2].axvline(0, color="0.5", lw=0.8)
    ax[2].set(xlabel="partial r within the band", ylabel="elevation (m)",
              title="information per band")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GLACIER}_srf_weight_test_{tag}.png", dpi=200)
    plt.close(fig)
    return summary


def main():
    ctx = prepare()
    stack = albedo_stack(ctx["ds"], ctx["y_dim"], ctx["x_dim"])
    ctx["snow_share"] = summarise(stack, SNOW_ALB)
    passes = [False, True] if ctx["hrz"] else [False]
    out = [test(ctx, summarise(stack, thr), label, h)
           for h in passes for label, thr in OBS_VARIANTS]
    out = pd.concat([o for o in out if o is not None], ignore_index=True)
    out.to_csv(f"{OUT_DIR}/{GLACIER}_srf_weight_summary.csv", index=False)
    cols = ["observation", "covariates", "n_px", "n_bands", "w_best", "w_p05", "w_p95", "w_reg",
            "p_50m_better_than_0.5"]
    print(f"\n== {GLACIER} summary (w = weight of the 5 km term; 0.5 = current SRF)")
    print(out[cols].round(2).to_string(index=False))
    print(f"  written to {OUT_DIR}")
    return out


def run_all():
    """All glaciers of glacier_config.txt (or GLACIERS); one table at the end."""
    out = []
    for cap, outline in read_config():
        if GLACIERS and cap not in GLACIERS:
            continue
        set_glacier(cap, outline)
        try:
            out.append(main())
        except Exception as e:                     # keep going with the next glacier
            print(f"!!! {cap}: {e}")
    if out:
        out = pd.concat(out, ignore_index=True)
        out.to_csv(f"{OUT_ROOT}/srf_weight_summary_all.csv", index=False)
        print("\n== ALL GLACIERS (w = weight of the 5 km term; 0.5 = current SRF)")
        print(out[["glacier", "observation", "covariates", "n_px", "n_bands", "w_best", "w_p05", "w_p95",
                   "w_reg", "a_50_p05", "a_50_p95", "b_5k_p05", "b_5k_p95"]].round(2).to_string(index=False))
        return out


if __name__ == "__main__":
    run_all()
