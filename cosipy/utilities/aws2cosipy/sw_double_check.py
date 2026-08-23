#!/usr/bin/env python3
"""
sw_double_check.py — Ist die down_pt-SW schon auf den Hang projiziert?

Hintergrund: toposcale2cosipy nimmt an, TopoPyScales `SW` sei rohes ssrd
(horizontal, nur hoehenkorrigiert), und wendet darauf HORAYZONs sw_dir_cor an
(Abschattung + Hangprojektion). Hat TopoPyScale den Direktstrahl aber bereits
projiziert, wird zweimal korrigiert -- auf sonnenabgewandten Baendern
quadriert sich ein Faktor < 1.

Drei Tests, vom schwaechsten zum staerksten:

  1. Verhaeltnis mean(down_pt SW) / mean(ssrd)
       ~1.0-1.15  -> plausibel nur hoehenkorrigiert
       ~0.6-0.85  -> etwas zieht Energie ab, bevor unser SW-Block laeuft

  2. Wie oft liegt SW ueber der EXTRATERRESTRISCHEN Einstrahlung auf die
     Horizontale (S0*ecc*sin_h)?
       Auf einer horizontalen Flaeche ist das UNMOEGLICH. Jeder Treffer
       beweist eine Projektion auf eine geneigte Flaeche.

  3. Korrelation SW_direct mit cos_illumination
       hoch -> der Direktstrahl traegt die Beleuchtungsgeometrie bereits.

Aufruf:
  python sw_double_check.py --glacier mera --aws-elev 5352
  python sw_double_check.py --glacier mera --all-bands
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJ = Path("/data/scratch/richteny/projects")
SOL0 = 1367.0

# lat/lon wie in toposcale2cosipy (lat = stationLat, lon = -tcart)
GLACIERS = {
    "abramov": (39.6097, 71.56986),   "parlung": (29.2324, 96.9237),
    "dongkemadi": (33.0820, 92.0630), "urumqi": (43.1170, 86.8010),
    "interiorglacier": (32.2510, 87.5050), "bordu": (41.81287, 78.17541),
    "guliya": (35.2610, 81.4620),     "kolahoi": (34.16268, 75.3151),
    "mera": (27.72478, 86.8920),      "halji": (30.26456, 81.47013),
}

DOWN_PT_DIRS = [
    "outputs/downscaled", "outputs", "downscaled", "outputs/down_pt",
]


def find_down_pt(glacier):
    base = PROJ / glacier
    for sub in DOWN_PT_DIRS:
        hits = sorted(glob.glob(str(base / sub / "down_pt_*.nc")))
        if hits:
            print(f"down_pt: {len(hits)} Dateien in {base / sub}")
            return hits
    hits = sorted(glob.glob(str(base / "**" / "down_pt_*.nc"), recursive=True))
    if hits:
        print(f"down_pt: {len(hits)} Dateien (rekursiv gefunden) in "
              f"{Path(hits[0]).parent}")
        return hits
    sys.exit(f"FEHLER: keine down_pt_*.nc unter {base}")


def sin_h(times, lat, lon):
    """Sinus der Sonnenhoehe, NOAA-Naeherung, times in UTC."""
    doy = times.dayofyear.values.astype(float)
    hr = times.hour.values + times.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hr - 12) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868*np.cos(g) - 0.032077*np.sin(g)
                    - 0.014615*np.cos(2*g) - 0.040849*np.sin(2*g))
    dec = (0.006918 - 0.399912*np.cos(g) + 0.070257*np.sin(g)
           - 0.006758*np.cos(2*g) + 0.000907*np.sin(2*g)
           - 0.002697*np.cos(3*g) + 0.00148*np.sin(3*g))
    tst = (hr * 60.0 + eqt + 4.0 * lon) % 1440.0
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    return np.sin(la)*np.sin(dec) + np.cos(la)*np.cos(dec)*np.cos(ha)


def pick_band(files, glacier, aws_elev):
    """Band, dessen Hoehe der AWS am naechsten liegt (via pts_list.csv)."""
    pl = PROJ / glacier / "pts_list.csv"
    if not pl.exists() or aws_elev is None:
        print(f"(kein pts_list.csv oder keine --aws-elev -> nehme {Path(files[len(files)//2]).name})")
        return [files[len(files) // 2]]
    df = pd.read_csv(pl)
    ecol = next((c for c in df.columns if c.lower() in
                 ("ele", "elev", "elevation", "hgt", "z")), None)
    if ecol is None:
        return [files[len(files) // 2]]
    i = int(np.abs(df[ecol].values - aws_elev).argmin())
    if i >= len(files):
        i = len(files) - 1
    print(f"AWS {aws_elev} m -> Band {i} ({df[ecol].values[i]:.0f} m): "
          f"{Path(files[i]).name}")
    return [files[i]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glacier", required=True)
    p.add_argument("--aws-elev", dest="aws_elev", type=float, default=None)
    p.add_argument("--all-bands", action="store_true",
                   help="alle Baender pruefen statt nur eines")
    p.add_argument("--year", default=None, help="nur ein Jahr, z.B. 2015")
    a = p.parse_args()

    g = a.glacier.lower()
    if g not in GLACIERS:
        sys.exit(f"FEHLER: {g} unbekannt. {sorted(GLACIERS)}")
    glat, glon = GLACIERS[g]
    print(f"{g}: ERA5-Zelle nearest zu ({glat:.4f}, {glon:.4f})")

    files = find_down_pt(g)
    targets = files if a.all_bands else pick_band(files, g, a.aws_elev)

    # ── SURF laden ─────────────────────────────────────────────────────────
    pat = f"SURF_{a.year}_*.nc" if a.year else "SURF_*.nc"
    sfiles = sorted(glob.glob(str(PROJ / g / "inputs" / "climate" / "yearly" / pat)))
    if not sfiles:
        sys.exit(f"FEHLER: keine {pat} gefunden")
    su = xr.open_mfdataset(sfiles, combine="by_coords", data_vars="minimal",
                           coords="minimal", compat="override", join="override")
    latn = "latitude" if "latitude" in su.coords else "lat"
    lonn = "longitude" if "longitude" in su.coords else "lon"
    lo = glon % 360.0 if float(su[lonn].min()) >= 0 and glon < 0 else glon
    su = su.sel({latn: glat, lonn: lo}, method="nearest")
    print(f"SURF-Zelle: {float(su[latn]):.2f} N, {float(su[lonn]):.2f} E "
          f"({len(sfiles)} Dateien)")

    rows = []
    for f in targets:
        dp = xr.open_dataset(f)
        t = dp.time.to_index().intersection(su.time.to_index())
        if len(t) < 100:
            print(f"  {Path(f).name}: zu wenig Zeitueberlapp"); continue

        sw = np.asarray(dp["SW"].sel(time=t).values, dtype=float).ravel()
        ssrd = np.asarray(su["ssrd"].sel(time=t).values, dtype=float).ravel() / 3600.0

        have_flat = ("SW_direct_flat" in dp.variables
                     and "SW_diffuse_flat" in dp.variables)
        if have_flat:
            swf = (np.asarray(dp["SW_direct_flat"].sel(time=t).values, float).ravel()
                   + np.asarray(dp["SW_diffuse_flat"].sel(time=t).values, float).ravel())
        else:
            swf = None

        # Test 2: extraterrestrische Obergrenze auf der Horizontalen
        sh = sin_h(t, glat, glon)
        ecc = 1.0 + 0.033 * np.cos(2*np.pi*t.dayofyear.values.astype(float)/365.25)
        toa = np.maximum(SOL0 * ecc * sh, 0.0)
        day = sh > 0.05
        over = int(np.sum(sw[day] > toa[day] * 1.02))     # 2 % Toleranz
        over_flat = (int(np.sum(swf[day] > toa[day] * 1.02))
                     if swf is not None else -1)

        # Test 3: traegt SW_direct die Beleuchtungsgeometrie?
        r_ci = np.nan
        if "SW_direct" in dp.variables and "cos_illumination" in dp.variables:
            d = np.asarray(dp["SW_direct"].sel(time=t).values, dtype=float).ravel()
            c = np.asarray(dp["cos_illumination"].sel(time=t).values, dtype=float).ravel()
            m = day & np.isfinite(d) & np.isfinite(c)
            if m.sum() > 100:
                r_ci = float(np.corrcoef(d[m], c[m])[0, 1])

        rows.append(dict(
            Datei=Path(f).name,
            SW_mean=sw.mean(), SW_max=sw.max(),
            ssrd_mean=ssrd.mean(), ssrd_max=ssrd.max(),
            Verhaeltnis=sw.mean() / max(ssrd.mean(), 1e-9),
            ueber_TOA=over,
            flat_Verh=(swf.mean()/max(ssrd.mean(),1e-9) if swf is not None else np.nan),
            ueber_TOA_flat=over_flat,
            r_dir_cosillu=r_ci))
        dp.close()

    if not rows:
        sys.exit("Nichts auswertbar.")
    res = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(res.round(3).to_string(index=False))

    v = res["Verhaeltnis"].mean()
    o = int(res["ueber_TOA"].sum())
    have_flat = bool(res["ueber_TOA_flat"].iloc[0] >= 0)
    of = int(res["ueber_TOA_flat"].clip(lower=0).sum())
    vf = res["flat_Verh"].mean()

    print("\n" + "=" * 92)
    print("BEFUND")
    print("  SW (altes, gelaendekorrigiertes Feld -- wird NICHT mehr verwendet):")
    print(f"     Verhaeltnis SW/ssrd    : {v:.3f}")
    print(f"     Zeitschritte ueber TOA : {o}")
    if have_flat:
        print("\n  SW_direct_flat + SW_diffuse_flat (das JETZT verwendete Feld):")
        print(f"     Verhaeltnis flat/ssrd  : {vf:.3f}")
        print(f"     Zeitschritte ueber TOA : {of}")
        if of == 0:
            print("\n  -> flat ueber TOA = 0: das horizontale Feld ueberschreitet die")
            print("     extraterrestrische Grenze nirgends. DOPPELKORREKTUR BEHOBEN.")
            print(f"     Dass SW (alt) weiter {o} Treffer hat, ist ERWARTET --")
            print("     dieses Feld liest toposcale2cosipy nicht mehr.")
        else:
            print("\n  -> WARNUNG: flat ueberschreitet die TOA-Grenze in "
                  f"{of} Faellen.")
            print("     SW_direct_flat soll horizontal + ungeschattet sein. Pruefen,")
            print("     ob es wirklich aus SW_direct_tmp (vor Projektion) kommt.")
    else:
        print("\n  Keine flat-Felder in den down_pt-Dateien -> alter Downscaling-")
        print("  Stand. Neu bauen, sonst laeuft toposcale2cosipy im Altpfad.")


if __name__ == "__main__":
    main()
