#!/usr/bin/env python3
"""
era5_cell_check.py — Liegen alle Baender eines Gletschers in derselben
ERA5-Gitterzelle, und macht es einen Unterschied?

Hintergrund: toposcale2cosipy zieht TCC (und die Liu-LWin-Groessen) ueber
load_cloud() aus EINER Zelle -- der Zelle, die dem Gletscherschwerpunkt am
naechsten liegt. TopoPyScale interpoliert seine Punkte dagegen selbst
horizontal (nearest ODER idw, je nach Konfiguration). Wenn der Gletscher eine
Zellgrenze schneidet, sind das nicht mehr dieselben Luftsaeulen.

Das Skript beantwortet zwei Fragen mit den vorhandenen Daten:
  1. Auf wie viele ERA5-Zellen verteilen sich die Gletscherpunkte?
  2. Wie stark unterscheidet sich TCC zwischen diesen Zellen -- also wie viel
     Fehler entsteht ueberhaupt, wenn man nur eine nimmt?

Aufruf:
  python era5_cell_check.py --glacier mera \
      --pts /pfad/pts_list.csv --lat 27.72478 --lon 86.8920

  # ohne pts_list: Bounding-Box direkt angeben
  python era5_cell_check.py --glacier mera --lat 27.72478 --lon 86.8920 \
      --bbox 27.68 27.78 86.85 86.95
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJ = Path("/data/scratch/richteny/projects")


def load_points(args):
    """Gletscherpunkte als (lat, lon)-Arrays."""
    if args.pts:
        df = pd.read_csv(args.pts)
        cols = {c.lower(): c for c in df.columns}
        la = next((cols[k] for k in ("lat", "latitude", "y") if k in cols), None)
        lo = next((cols[k] for k in ("lon", "longitude", "x") if k in cols), None)
        if la is None or lo is None:
            sys.exit(f"FEHLER: keine lat/lon-Spalten in {args.pts}. "
                     f"Gefunden: {list(df.columns)}")
        print(f"{len(df)} Punkte aus {Path(args.pts).name}")
        return df[la].values, df[lo].values
    if args.bbox:
        y0, y1, x0, x1 = args.bbox
        gy, gx = np.meshgrid(np.linspace(y0, y1, 25), np.linspace(x0, x1, 25))
        print(f"Bounding-Box {y0}-{y1} N, {x0}-{x1} E (25x25 Raster)")
        return gy.ravel(), gx.ravel()
    sys.exit("FEHLER: --pts oder --bbox angeben.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glacier", required=True, help="Projektname (klein), z.B. mera")
    p.add_argument("--lat", type=float, required=True, help="Gletscher-Zentrum")
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--pts", default=None, help="pts_list.csv von TopoPyScale")
    p.add_argument("--bbox", nargs=4, type=float, default=None,
                   metavar=("LAT0", "LAT1", "LON0", "LON1"))
    a = p.parse_args()

    cdir = PROJ / a.glacier / "inputs" / "climate" / "yearly"
    files = sorted(glob.glob(str(cdir / "CLOUD_*.nc")))
    if not files:
        sys.exit(f"FEHLER: keine CLOUD_*.nc in {cdir}")

    ds = xr.open_dataset(files[0])
    latn = "latitude" if "latitude" in ds.coords else "lat"
    lonn = "longitude" if "longitude" in ds.coords else "lon"
    glat_grid = np.asarray(ds[latn].values, dtype=float)
    glon_grid = np.asarray(ds[lonn].values, dtype=float)
    dlat = abs(np.diff(glat_grid).mean()) if len(glat_grid) > 1 else 0.25
    dlon = abs(np.diff(glon_grid).mean()) if len(glon_grid) > 1 else 0.25
    print(f"ERA5-Gitter: {len(glat_grid)} x {len(glon_grid)} Zellen, "
          f"Aufloesung {dlat:.3f}deg x {dlon:.3f}deg")

    plat, plon = load_points(a)

    # --- 1) Auf welche Zellen fallen die Punkte? ---
    ilat = np.abs(plat[:, None] - glat_grid[None, :]).argmin(axis=1)
    ilon = np.abs(plon[:, None] - glon_grid[None, :]).argmin(axis=1)
    pairs, counts = np.unique(np.stack([ilat, ilon], 1), axis=0,
                              return_counts=True)

    print(f"\nAusdehnung der Punkte: "
          f"{plat.min():.4f}-{plat.max():.4f} N  ({(plat.max()-plat.min())*111:.1f} km), "
          f"{plon.min():.4f}-{plon.max():.4f} E  "
          f"({(plon.max()-plon.min())*111*np.cos(np.radians(a.lat)):.1f} km)")
    print(f"\nBetroffene ERA5-Zellen: {len(pairs)}")
    for (i, j), n in zip(pairs, counts):
        print(f"  ({glat_grid[i]:.2f} N, {glon_grid[j]:.2f} E): "
              f"{n} Punkte ({100*n/len(plat):.1f} %)")

    # Zelle, die load_cloud tatsaechlich nimmt
    ci = int(np.abs(glat_grid - a.lat).argmin())
    cj = int(np.abs(glon_grid - a.lon).argmin())
    print(f"\nload_cloud() nimmt: ({glat_grid[ci]:.2f} N, {glon_grid[cj]:.2f} E)")
    used = np.any((pairs[:, 0] == ci) & (pairs[:, 1] == cj))
    if len(pairs) == 1:
        print("  -> alle Punkte in dieser Zelle. Die Vereinfachung ist exakt.")
    else:
        share = counts[(pairs[:, 0] == ci) & (pairs[:, 1] == cj)]
        share = int(share[0]) if len(share) else 0
        print(f"  -> nur {100*share/len(plat):.1f} % der Punkte liegen darin. "
              f"Der Gletscher schneidet eine Zellgrenze.")

    # --- 2) Wie stark unterscheidet sich TCC zwischen den Zellen? ---
    if "tcc" not in ds.variables:
        print("\n(keine Variable 'tcc' -- Teil 2 uebersprungen)")
        return
    if len(pairs) == 1:
        print("\nTeil 2 entfaellt: nur eine Zelle betroffen.")
        return

    print("\nTCC-Vergleich zwischen den betroffenen Zellen "
          f"(Stichprobe: {Path(files[0]).name})")
    series = {}
    for (i, j) in pairs:
        v = ds["tcc"].isel({latn: int(i), lonn: int(j)}).values.astype(float)
        series[f"{glat_grid[i]:.2f}N_{glon_grid[j]:.2f}E"] = v
    sdf = pd.DataFrame(series)
    print(f"  Mittelwerte: "
          + ", ".join(f"{k}={v:.3f}" for k, v in sdf.mean().items()))
    print(f"  Spannweite der Mittelwerte: {sdf.mean().max()-sdf.mean().min():.3f}")
    print(f"  mittlere |Differenz| zur load_cloud-Zelle:")
    ref = f"{glat_grid[ci]:.2f}N_{glon_grid[cj]:.2f}E"
    if ref in sdf:
        for c in sdf.columns:
            if c == ref:
                continue
            d = (sdf[c] - sdf[ref]).abs()
            r = np.corrcoef(sdf[c], sdf[ref])[0, 1]
            print(f"    {c}: mean |dTCC| = {d.mean():.3f}, r = {r:.3f}")
    print("""
Einordnung:
  mean |dTCC| < 0.05 und r > 0.95 -> eine Zelle zu nehmen ist unkritisch,
      der Fehler liegt weit unter dem, was die Diffusaufteilung ohnehin
      an Unsicherheit hat.
  groessere Werte -> auf horizontale Interpolation umstellen (in load_cloud
      .sel(method='nearest') durch .interp() ersetzen).""")


if __name__ == "__main__":
    main()
