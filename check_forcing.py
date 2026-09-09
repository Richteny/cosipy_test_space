#!/usr/bin/env python3
"""
check_forcing.py — physical sanity check of a COSIPY band forcing file.

Uses no observations, so it runs on all ten glaciers, including the seven
without an AWS. Everything here is either an internal consistency relation
(pressure against barometry, RH against q/T/p) or a physical bound that cannot
be violated (emissivity, the extraterrestrial limit).

    python check_forcing.py --forcing .../Mera_ERA5_1D20m_..._LW-liu-cf.nc \
        --lat 27.72478 --lon 86.8920

    python check_forcing.py --forcing '.../data/input/*/*_LW-liu-cf.nc' --glob \
        --config .../glacier_config.txt          # all glaciers at once

Exit code is 1 if any FAIL is raised, so it can gate a pipeline.
"""
import argparse
import glob as _glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SIGMA = 5.670374419e-8
SOL0 = 1367.0

# Expected ranges. Deliberately generous -- these catch broken forcing, not
# subtle bias.
LIMITS = dict(
    dT_dz=(-1.10, -0.20),      # K / 100 m, mean over the profile
    dLW_dz=(-6.0, -1.0),       # W m-2 / 100 m
    eps_eff=(0.45, 1.00),      # LWin / (sigma T^4), band means
    pres_resid=3.0,            # hPa, max deviation from barometry
    rh_at_100=25.0,            # % of steps pinned at 100 % RH
)


def _prof(ds, var):
    """Band-mean profile of a variable, shape (band,)."""
    if var not in ds:
        return None
    a = np.squeeze(np.asarray(ds[var].values, dtype=float))
    if a.ndim == 3:
        a = a[:, :, 0]
    return np.nanmean(a, axis=0) if a.ndim == 2 else a


def _slope(y, z):
    """Least-squares slope per 100 m."""
    ok = np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(z[ok], y[ok], 1)[0] * 100.0)


def check_one(path, lat=None, lon=None, verbose=True):
    ds = xr.open_dataset(path)
    name = Path(path).name
    fails, warns = [], []
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")

    hgt = np.asarray(ds["HGT"].values, dtype=float).ravel()
    order = np.argsort(hgt)
    nb, nt = len(hgt), ds.sizes.get("time", 0)
    print(f"  {nb} bands, {hgt.min():.0f}-{hgt.max():.0f} m, {nt} steps")

    prof = {v: _prof(ds, v) for v in
            ("T2", "RH2", "U2", "G", "LWin", "PRES", "RRR")}
    T = prof["T2"]
    if T is not None and np.nanmedian(T) < 150:
        T = T + 273.15
    z = hgt

    # --- 1. NaNs and hard bounds -----------------------------------------
    BOUNDS = dict(T2=(223, 316), RH2=(0, 100), U2=(0, 60), G=(0, 1600),
                  LWin=(0, 500), PRES=(200, 1080), RRR=(0, 100))
    print("\n  range check")
    for v, (lo, hi) in BOUNDS.items():
        if v not in ds:
            continue
        a = np.asarray(ds[v].values, dtype=float)
        n_nan = int(np.sum(~np.isfinite(a)))
        out = int(np.sum((a < lo) | (a > hi)))
        tag = "ok"
        if n_nan:
            fails.append(f"{v}: {n_nan} NaN"); tag = "FAIL"
        elif out:
            frac = 100 * out / a.size
            tag = f"{out} outside ({frac:.3f} %)"
            (warns if frac < 0.01 else fails).append(f"{v}: {tag}")
        print(f"    {v:5s} {np.nanmin(a):9.2f} .. {np.nanmax(a):9.2f}   {tag}")

    # --- 2. Vertical gradients -------------------------------------------
    print("\n  vertical structure")
    if T is not None:
        g = _slope(T, z)
        ok = LIMITS["dT_dz"][0] <= g <= LIMITS["dT_dz"][1]
        print(f"    dT2/dz    {g:+7.3f} K/100m      "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"dT2/dz = {g:+.3f} K/100m outside "
                         f"{LIMITS['dT_dz']}")
        # monotonic in the mean?
        Ts = T[order]
        n_up = int(np.sum(np.diff(Ts) > 0))
        if n_up > 0.1 * nb:
            warns.append(f"T2 rises with elevation between {n_up} band pairs")
            print(f"      note: T2 rises between {n_up} of {nb-1} band pairs")

    if prof["U2"] is not None:
        gu = _slope(prof["U2"], z)
        print(f"    dU2/dz    {gu:+7.3f} m/s/100m     "
              f"{'ok' if abs(gu) < 0.30 else 'check'}")
        if abs(gu) >= 0.30:
            warns.append(f"dU2/dz = {gu:+.3f} m/s/100m is steep. TopoPyScale "
                         f"takes wind from pressure levels, so the upper bands "
                         f"sit close to the free troposphere where speeds are "
                         f"much higher than in a glacier boundary layer.")
            print("      note: pressure-level wind approaches free-troposphere "
                  "values\n            at the upper bands; the AWS measures a "
                  "decoupled surface layer.")

    if prof["LWin"] is not None:
        g = _slope(prof["LWin"], z)
        ok = LIMITS["dLW_dz"][0] <= g <= LIMITS["dLW_dz"][1]
        print(f"    dLWin/dz  {g:+7.3f} W/m2/100m   {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"dLWin/dz = {g:+.3f} outside {LIMITS['dLW_dz']}")
        print(f"      (Zhu et al. 2017 measure -3.3 W/m2/100m at Muztagata)")

    # --- 3. Pressure against barometry -----------------------------------
    if prof["PRES"] is not None and T is not None:
        p = prof["PRES"]
        i0 = int(np.argmin(z))
        Tm = float(np.nanmean(T))
        p_baro = p[i0] * np.exp(-9.81 * (z - z[i0]) / (287.05 * Tm))
        resid = float(np.nanmax(np.abs(p - p_baro)))
        ok = resid <= LIMITS["pres_resid"]
        print(f"\n  pressure vs barometry: max residual {resid:.2f} hPa   "
              f"{'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"PRES deviates {resid:.1f} hPa from barometry -- "
                         f"check the band-to-HGT assignment")

    # --- 4. Effective emissivity ------------------------------------------
    if prof["LWin"] is not None and T is not None:
        eps = prof["LWin"] / (SIGMA * T ** 4)
        lo, hi = LIMITS["eps_eff"]
        bad = int(np.sum((eps < lo) | (eps > hi)))
        print(f"\n  eps_eff = LWin/(sigma T^4): {np.nanmin(eps):.3f} .. "
              f"{np.nanmax(eps):.3f}   {'ok' if bad == 0 else 'FAIL'}")
        if bad:
            fails.append(f"eps_eff outside [{lo}, {hi}] in {bad} bands")
        print(f"      band mean {np.nanmean(eps):.3f} "
              f"(0.6-0.8 typical at these elevations)")

    # --- 5. RH consistency and saturation ---------------------------------
    if "RH2" in ds:
        rh = np.asarray(ds["RH2"].values, dtype=float)
        f100 = 100 * float(np.mean(rh >= 99.9))
        tag = "ok" if f100 <= LIMITS["rh_at_100"] else "check"
        print(f"\n  RH2 at >=99.9 %: {f100:.1f} % of steps   {tag}")
        if f100 > LIMITS["rh_at_100"]:
            warns.append(f"RH2 saturated in {f100:.0f} % of steps")

    # --- 6. Shortwave against the extraterrestrial limit ------------------
    if "G" in ds and lat is not None and lon is not None:
        t = pd.to_datetime(ds.time.values)
        doy = t.dayofyear.values.astype(float)
        hr = t.hour.values + t.minute.values / 60.0
        gg = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
        eqt = 229.18 * (0.000075 + 0.001868*np.cos(gg) - 0.032077*np.sin(gg)
                        - 0.014615*np.cos(2*gg) - 0.040849*np.sin(2*gg))
        dec = (0.006918 - 0.399912*np.cos(gg) + 0.070257*np.sin(gg)
               - 0.006758*np.cos(2*gg) + 0.000907*np.sin(2*gg)
               - 0.002697*np.cos(3*gg) + 0.00148*np.sin(3*gg))
        tst = (hr * 60.0 + eqt + 4.0 * lon) % 1440.0
        ha = np.radians(tst / 4.0 - 180.0)
        la = np.radians(lat)
        sh = np.sin(la)*np.sin(dec) + np.cos(la)*np.cos(dec)*np.cos(ha)
        ecc = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.25)
        toa = np.maximum(SOL0 * ecc * sh, 0.0)
        Gv = np.squeeze(np.asarray(ds["G"].values, dtype=float))
        if Gv.ndim == 3:
            Gv = Gv[:, :, 0]
        night = sh <= 0.0
        n_night = int(np.sum(Gv[night] > 5.0))
        # a slope can legitimately exceed the horizontal TOA, so only flag
        # values above the solar constant itself
        n_abs = int(np.sum(Gv > SOL0 * 1.05))
        print(f"\n  G at night > 5 W/m2 : {n_night}   "
              f"{'ok' if n_night == 0 else 'FAIL'}")
        print(f"  G above the solar constant: {n_abs}   "
              f"{'ok' if n_abs == 0 else 'check'}")
        if n_night:
            fails.append(f"{n_night} non-zero G values at night -- "
                         f"time zone or LUT indexing")
        if n_abs:
            warns.append(f"{n_abs} G values above {SOL0} W/m2")

    # --- verdict ----------------------------------------------------------
    print()
    if fails:
        print("  FAIL")
        for f in fails:
            print(f"    - {f}")
    if warns:
        print("  warnings")
        for w in warns:
            print(f"    - {w}")
    if not fails and not warns:
        print("  all checks passed")
    ds.close()
    return len(fails), prof, hgt


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--forcing", required=True, nargs="+",
                   help="one or more files. A shell glob expands before the "
                        "script sees it, which is fine -- all paths are taken. "
                        "Quote the pattern and add --glob to expand it here "
                        "instead.")
    p.add_argument("--glob", action="store_true",
                   help="expand --forcing as a pattern inside the script "
                        "(quote it in the shell)")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--plot", default=None,
                   help="write a profile figure to this path")
    a = p.parse_args()

    if a.glob:
        files = sorted({f for pat in a.forcing for f in _glob.glob(pat)})
    else:
        files = sorted(set(a.forcing))
    if not files:
        sys.exit(f"no files matching {a.forcing}")
    print(f"{len(files)} file(s) to check")

    total_fail, profiles = 0, {}
    for f in files:
        nf, prof, hgt = check_one(f, a.lat, a.lon)
        total_fail += nf
        profiles[Path(f).name] = (prof, hgt)

    if a.plot and profiles:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vs = ["T2", "RH2", "U2", "G", "LWin", "PRES"]
        UN = {"T2": "K", "RH2": "%", "U2": "m s$^{-1}$", "G": "W m$^{-2}$",
              "LWin": "W m$^{-2}$", "PRES": "hPa"}
        # One row per glacier: shared axes across ten glaciers would compress
        # every profile into a sliver, and the point is the SHAPE of each.
        names = list(profiles)
        nr, nc = len(names), len(vs)
        fig, ax = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.5 * nr),
                               squeeze=False)
        for r, nm in enumerate(names):
            prof, hgt = profiles[nm]
            short = nm.split("_")[0]
            for c, v in enumerate(vs):
                a_ = ax[r][c]
                y = prof.get(v)
                if y is None:
                    a_.set_visible(False); continue
                yy = y + 273.15 if (v == "T2" and np.nanmedian(y) < 150) else y
                a_.plot(yy, hgt, "o-", ms=2.5, lw=1.3, color="#377eb8")
                if v == "T2":
                    a_.axvline(273.15, color="0.5", ls="--", lw=1.0)
                a_.grid(True, ls=":", alpha=0.45)
                a_.tick_params(labelsize=7)
                if r == 0:
                    a_.set_title(f"{v} ({UN.get(v, '')})", fontsize=9,
                                 fontweight="bold")
                if r == nr - 1:
                    a_.set_xlabel(UN.get(v, ""), fontsize=8)
                if c == 0:
                    a_.set_ylabel(f"{short}\nelevation (m)", fontsize=8.5)
        fig.suptitle("Band-mean vertical profiles. Dashed line in T2: 273.15 K.",
                     fontweight="bold", y=1.002)
        fig.tight_layout()
        fig.savefig(a.plot, bbox_inches="tight", dpi=150)
        print(f"\nfigure written: {a.plot}")

    print(f"\n{'=' * 78}")
    print(f"{len(files)} file(s) checked, {total_fail} failure(s)")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
