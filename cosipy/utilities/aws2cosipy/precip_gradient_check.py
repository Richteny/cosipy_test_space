"""
Precipitation vs elevation over each glacier: what the forcing has, what TopoPyScale's precipitation lapse rate would
give, what HMA_SR_D says (Oct-May, all ten glaciers), and our linear COSIPY lapse rate -- per elevation band.

1. Forcing (COSIPY file, RRR as used now = raw ERA5 tp, --precip-lr off): Oct-May and JJAS sums per band, relative
   to the glacier mean (N_Points-weighted). Expected ~flat.
2. TopoPyScale (topo_scale.py, precip_lapse_rate_flag=True):
       f = (1 + c*dz) / (1 - c*dz),  dz = (band z - ERA5 surface z) [km],
       c = 0.35 0.35 0.35 0.30 0.25 0.20 0.20 0.20 0.20 0.25 0.30 0.35 km-1 (Jan..Dec; Liston & Elder MicroMet,
       from Thornton et al. 1997, western US).
   Monotonic and accelerating with height (singular at dz = 1/c = 2.9-5 km). The ERA5 surface z is the SURF 'z'
   (surface geopotential) of the TopoPyScale project, interpolated to each pts_list point as TopoPyScale does.
   Shown relative to the glacier-mean band, Oct-May and JJAS (forcing-precipitation-weighted over the months).
3. HMA_SR_D (hmasr_precip_constraint.py output *_hmasr_per_year_bin.csv): Oct-May precipitation per 100 m bin of
   the seasonal-snow pixels within 15 km, all years, relative to the glacier-mean bin. Off-glacier pixels, winter only,
   few pixels in high bins (n_px printed) -- an independent, snow-cover-constrained estimate available at all glaciers.
   All bins are plotted (also above / below the glacier, open symbols) to see whether a maximum appears.
4. COSIPY linear lapse (precip_lapse_rate in constants.toml): f = 1 + (z - z_ref) * LAPSE_COSIPY, z_ref = mean HGT.

Gradients are also given as a log-linear fit in % per 100 m over the glacier's elevation range.

    python precip_gradient_check.py               (all glaciers found)
    python precip_gradient_check.py Dongkemadi    (one or several)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

# --- HPC layout -------------------------------------------------------------------------------------------------------
PROJ = "/data/scratch/richteny/projects"                                   # TopoPyScale projects (native structure)
INPUT = "/data/scratch/richteny/thesis/cosipy_test_space/data/input"      # COSIPY forcing (lapse rate off)
HMASR = "/data/scratch/richteny/HMA_SWE"                                   # HMA_SR_D raw + *_hmasr_per_year_bin.csv
FORCING = "*_ERA5_1D20m_HORAYZON_1987_2024-terrprinz.nc"
OUT_DIR = "/data/scratch/richteny/HMA_SWE/precip_gradient_check"
# glacier: (TopoPyScale project dir, names to try for the forcing folder and the HMA_SR_D csv)
GLACIERS = {"Abramov": ("abramov", ["Abramov"]), "Bordu": ("bordu", ["Bordu"]),
            "Dongkemadi": ("dongkemadi", ["Dongkemadi"]), "Guliya": ("guliya", ["Guliya"]),
            "Halji": ("halji", ["Halji"]), "InteriorGlacier": ("interiorglacier", ["InteriorGlacier"]),
            "Kolahoi": ("kolahoi", ["Kolahoi"]), "Mera": ("mera", ["Mera"]),
            "ParlungNo4": ("parlung", ["ParlungNo4", "Parlung"]), "Urumqi": ("urumqi", ["UrumqiNo1", "Urumqi"])}
GLACIER_LATLON = {"Abramov": (39.610, 71.570), "Bordu": (41.813, 78.175), "Dongkemadi": (33.082, 92.063),
                  "Guliya": (35.261, 81.462), "Halji": (30.265, 81.470), "InteriorGlacier": (32.251, 87.505),
                  "Kolahoi": (34.163, 75.315), "Mera": (27.725, 86.892), "ParlungNo4": (29.232, 96.924),
                  "Urumqi": (43.117, 86.801)}
LAPSE_COSIPY = 0.0004                                            # per m
C_MONTH = np.array([0.35, 0.35, 0.35, 0.30, 0.25, 0.20, 0.20, 0.20, 0.20, 0.25, 0.30, 0.35])
WINTER, SUMMER = [10, 11, 12, 1, 2, 3, 4, 5], [6, 7, 8, 9]
MIN_PX = 5


def tps_factor(dz_km, month):
    c = C_MONTH[np.asarray(month) - 1]
    return (1 + c * dz_km) / (1 - c * dz_km)


def fit_pct_per_100m(z, rel, w=None):
    """log-linear slope of rel against z, in % per 100 m."""
    ok = np.isfinite(rel) & (rel > 0) & np.isfinite(z)
    if ok.sum() < 3:
        return np.nan
    b = np.polyfit(z[ok], np.log(rel[ok]), 1, w=None if w is None else np.sqrt(w[ok]))[0]
    return 100 * (np.exp(100 * b) - 1)


def forcing(g):
    hits = []
    for name in GLACIERS[g][1]:
        hits += sorted(glob.glob(os.path.join(INPUT, name, "**", FORCING), recursive=True))
    if not hits:
        return None
    with xr.open_dataset(hits[0]) as fo:
        first = lambda v: (fo[v].isel(time=0) if "time" in fo[v].dims else fo[v]).values.ravel()
        w = np.nan_to_num(first("N_Points").astype(float))
        z = first("HGT").astype(float)
        p = np.nan_to_num(fo["RRR"].values.reshape(fo.sizes["time"], -1).astype(float))
        t = pd.DatetimeIndex(fo.time.values)
    sel = w > 0
    w, z, p = w[sel], z[sel], p[:, sel]
    nyr = len(np.unique(t.year))
    mon = pd.DataFrame(p, index=t).groupby(t.month).sum() / nyr        # mm per calendar month, per band
    return dict(z=z, w=w, mon=mon, file=os.path.basename(hits[0]))


def era5_z(g, band_z):
    """ERA5 surface height as TopoPyScale sees it: SURF 'z' (surface geopotential) of the project, interpolated
    linearly to each pts_list point (as surf_interp), matched to the forcing bands by elevation. Returns an array
    (one value per forcing band) and a description."""
    d = os.path.join(PROJ, GLACIERS[g][0])
    files = sorted(glob.glob(os.path.join(d, "inputs", "climate", "yearly", "SURF_*.nc")))
    if not files:
        files = sorted(glob.glob(os.path.join(d, "inputs", "climate", "SURF*.nc")))
    if not files:
        return None, f"no SURF files under {d}/inputs/climate"
    with xr.open_dataset(files[0]) as s_:
        zz = s_["z"]
        if "time" in zz.dims:
            zz = zz.isel(time=0)
        zz = zz.load()
    if float(zz.max()) > 2.0e4:                                     # geopotential (m2 s-2) -> m
        zz = zz / 9.80665
    pl = os.path.join(d, "pts_list.csv")
    if os.path.exists(pl):
        pts = pd.read_csv(pl)
        cl = {c.lower(): c for c in pts.columns}
        la, lo = cl.get("latitude", cl.get("lat")), cl.get("longitude", cl.get("lon"))
        if la and lo and "elevation" in cl:
            ze = np.array([float(zz.interp(latitude=r_[la], longitude=r_[lo])) for _, r_ in pts.iterrows()])
            pe = pts[cl["elevation"]].values.astype(float)
            ze_band = ze[np.abs(band_z[:, None] - pe[None, :]).argmin(1)]
            return ze_band, f"SURF z at {len(pts)} points ({np.nanmin(ze):.0f}-{np.nanmax(ze):.0f} m)"
    la, lo = GLACIER_LATLON[g]
    v = float(zz.interp(latitude=la, longitude=lo))
    return np.full(band_z.shape, v), "SURF z at the glacier coordinate"


def hmasr_profile(g):
    hits = []
    for name in [g] + GLACIERS[g][1]:
        hits += sorted(glob.glob(os.path.join(HMASR, "**", f"{name}_hmasr_per_year_bin.csv"), recursive=True))
    if not hits:
        return None
    D = pd.read_csv(hits[0])
    D = D[D.n_px >= MIN_PX]
    if D.empty:
        return None
    P = D.groupby("z").agg(hmasr=("hmasr_winter", "mean"), forc=("forcing_winter", "mean"), n_px=("n_px", "mean"),
                           years=("WY", "nunique"))
    return P


def main(want):
    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, panels = [], []
    for g in want:
        F = forcing(g)
        if F is None:
            print(f"{g}: no forcing file")
            continue
        z, w, mon = F["z"], F["w"], F["mon"]
        zmean = np.sum(z * w) / w.sum()
        o = np.argsort(z)
        r = {"glacier": g, "bands": f"{z.min():.0f}-{z.max():.0f}", "z_mean": zmean}
        prof = {"z": z[o]}

        # 1 forcing
        for lab, ms in (("Oct-May", WINTER), ("JJAS", SUMMER)):
            pb = mon.loc[ms].sum().values
            rel = pb / (np.sum(pb * w) / w.sum())
            prof[f"forcing {lab}"] = rel[o]
            r[f"forcing {lab} [%/100m]"] = fit_pct_per_100m(z, rel, w)

        # 2 TopoPyScale
        ze, src = era5_z(g, z)
        r["ERA5 z source"] = src
        if ze is not None and np.isfinite(ze).all():
            r["ERA5 z [m]"] = float(np.median(ze))
            dz = (z - ze) / 1000.0
            r["dz tongue / top [km]"] = f"{dz[o][0]:.2f} / {dz[o][-1]:.2f}"
            for lab, ms in (("Oct-May", WINTER), ("JJAS", SUMMER)):
                pm = mon.loc[ms].values                                        # months x bands
                fac = np.array([tps_factor(dz, m) for m in ms])                # months x bands
                pb = (pm * fac).sum(0) / np.maximum(pm.sum(0), 1e-12)          # precipitation-weighted factor
                rel = pb / (np.sum(pb * w) / w.sum())
                prof[f"TopoPyScale {lab}"] = rel[o]
                r[f"TopoPyScale {lab} [%/100m]"] = fit_pct_per_100m(z, rel, w)
                r[f"TopoPyScale {lab} factor tongue / top"] = f"{pb[o][0]:.2f} / {pb[o][-1]:.2f}"

        # 3 HMA_SR_D: all bins (15 km around the glacier, also above / below it), fit inside the glacier range
        P = hmasr_profile(g)
        if P is not None:
            pz = P.index.values
            inside = (pz >= z.min() - 50) & (pz <= z.max() + 50)
            if inside.sum() >= 2:
                Pi = P[inside]
                ref = np.average(Pi.hmasr.values, weights=Pi.n_px.values)
                prof["HMA_SR_D"] = (pz, P.hmasr.values / ref, P.n_px.values, inside)
                rel = Pi.hmasr.values / ref
                r["HMA_SR_D Oct-May [%/100m]"] = fit_pct_per_100m(Pi.index.values, rel, Pi.n_px.values)
                r["HMA_SR_D bins on glacier (n_px min-max)"] = f"{inside.sum()} ({Pi.n_px.min():.0f}-{Pi.n_px.max():.0f})"
                r["HMA_SR_D bin of max Oct-May precip [m]"] = float(pz[np.argmax(P.hmasr.values)])
                r["HMA_SR_D bins all (range)"] = f"{len(pz)} ({pz.min():.0f}-{pz.max():.0f})"
                relf = Pi.forc.values / np.average(Pi.forc.values, weights=Pi.n_px.values)
                r["forcing in HMA bins [%/100m]"] = fit_pct_per_100m(Pi.index.values, relf, Pi.n_px.values)

        # 4 COSIPY linear
        relc = np.maximum(1 + (z - zmean) * LAPSE_COSIPY, 0)
        prof["COSIPY linear"] = relc[o]
        r["COSIPY linear [%/100m]"] = fit_pct_per_100m(z, relc, w)
        rows.append(r)
        panels.append((g, prof))

    T = pd.DataFrame(rows).set_index("glacier")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(T.T.to_string(float_format=lambda v: f"{v:8.2f}"))
    T.to_csv(os.path.join(OUT_DIR, "precip_gradient_check.csv"))

    n = len(panels)
    if not n:
        return
    nc = min(n, 5)
    fig, axs = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(3.4 * nc, 4.2 * np.ceil(n / nc)), squeeze=False)
    for a, (g, p) in zip(axs.ravel(), panels):
        zz = p["z"]
        a.plot(p["forcing Oct-May"], zz, c="0.5", lw=1, label="forcing Oct-May")
        a.plot(p["forcing JJAS"], zz, c="0.5", lw=1, ls="--", label="forcing JJAS")
        if "TopoPyScale Oct-May" in p:
            a.plot(p["TopoPyScale Oct-May"], zz, c="C0", label="TopoPyScale Oct-May")
            a.plot(p["TopoPyScale JJAS"], zz, c="C0", ls="--", label="TopoPyScale JJAS")
        a.plot(p["COSIPY linear"], zz, c="C3", label=f"COSIPY {LAPSE_COSIPY * 1e4:.0f} %/100 m")
        if "HMA_SR_D" in p:
            hz, hr, hn, ins = p["HMA_SR_D"]
            sz = 10 + 40 * hn / hn.max()
            a.scatter(hr[ins], hz[ins], s=sz[ins], c="C2", zorder=3, label="HMA_SR_D Oct-May (glacier range)")
            a.scatter(hr[~ins], hz[~ins], s=sz[~ins], facecolors="none", edgecolors="C2", zorder=3,
                      label="HMA_SR_D outside glacier range")
        a.axvline(1, c="0.8", lw=0.8)
        a.set(title=g, xlabel="precipitation / glacier mean")
        a.set_xlim(0, max(2.5, a.get_xlim()[1]) if a.get_xlim()[1] < 4 else 4)
    axs[0, 0].set_ylabel("elevation (m)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=4)
    for a in axs.ravel()[n:]:
        a.set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, "precip_gradient_check.png"), dpi=150)
    plt.close(fig)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(GLACIERS))
