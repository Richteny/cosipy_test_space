#!/usr/bin/env python3
"""
era5raw2cosipy.py — ROHES ERA5 auf COSIPY-20m-Baender, OHNE Downscaling.

Zweck: Kontrollgruppe fuer den Forcing-Vergleich. Nimmt die naechstgelegene
ERA5-Gitterzelle (~31 km) und schreibt deren Werte UNVERAENDERT in alle
Hoehenbaender. Keine vertikale Interpolation, keine Lapse Rates, keine
HORAYZON-SW-Korrektur, kein Terrain-Term.

Damit beantwortet der Vergleich gegen die TopoPyScale-Version genau eine
Frage: was bringt das Downscaling? Jeder Unterschied ist Downscaling-Effekt,
weil sonst nichts variiert wurde.

Ausgabeformat identisch zu toposcale2cosipy.py (gleiche Statik aus der
SRF-Datei), damit beide Dateien im selben Notebook vergleichbar sind.

Beispiel:
  python era5raw2cosipy.py --glacier Mera \
      -o Mera_ERA5raw_1D20m_1987_2024.nc \
      -s .../Mera_combined_SRF_1D20m.nc \
      --stationLat 27.725 --tcart -86.892 -b 1987-01-01 -e 2024-12-31
"""
import argparse, glob, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJ = Path("/data/scratch/richteny/projects")

SIGMA_SB = 5.670374419e-8
G0 = 9.80665                      # ERA5-Schwerebeschleunigung (fuer z -> Hoehe)

# Gletscher-Tabelle aus der glacier-config.
# ACHTUNG: Projektordner (Schluessel, klein) und Static-Ordner (Grossschreibung)
# sind NICHT identisch -- parlung vs. ParlungNo4. Der Static-Name bestimmt auch
# den Dateinamen-Praefix, damit das Auswertungsnotebook die Datei findet.
GLACIERS = {
    "abramov":         dict(static="Abramov",         lat=39.6097,  lon=71.56986),
    "parlung":         dict(static="ParlungNo4",      lat=29.2324,  lon=96.9237),
    "dongkemadi":      dict(static="Dongkemadi",      lat=33.0820,  lon=92.0630),
    "urumqi":          dict(static="Urumqi",          lat=43.1170,  lon=86.8010),
    "interiorglacier": dict(static="InteriorGlacier", lat=32.2510,  lon=87.5050),
    "bordu":           dict(static="Bordu",           lat=41.81287, lon=78.17541),
    "guliya":          dict(static="Guliya",          lat=35.2610,  lon=81.4620),
    "kolahoi":         dict(static="Kolahoi",         lat=34.16268, lon=75.3151),
    "mera":            dict(static="Mera",            lat=27.72478, lon=86.8920),
    "halji":           dict(static="Halji",           lat=30.26456, lon=81.47013),
}

# Standardpfade -- ueber -s / -o ueberschreibbar
COSIPY_BASE = Path("/data/scratch/richteny/thesis/cosipy_test_space/data")
STATIC_DIR  = COSIPY_BASE / "static"
INPUT_DIR   = COSIPY_BASE / "input"


# Namensvarianten der ERA5-Felder (ARCO / CDS-NetCDF / GRIB)
CAND = {
    "t2m":  ["t2m", "2m_temperature", "temperature_2m", "2t"],
    "d2m":  ["d2m", "2m_dewpoint_temperature", "dewpoint_temperature_2m", "2d"],
    "q":    ["q", "specific_humidity", "sh2"],
    "sp":   ["sp", "surface_pressure", "ps"],
    "u10":  ["u10", "10m_u_component_of_wind", "u_component_of_wind_10m", "10u"],
    "v10":  ["v10", "10m_v_component_of_wind", "v_component_of_wind_10m", "10v"],
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "strd": ["strd", "surface_thermal_radiation_downwards"],
    "tp":   ["tp", "total_precipitation"],
}


def resolve(name):
    """--glacier tolerant aufloesen: Projektname (mera) oder Static-Name (Mera)."""
    key = name.lower()
    if key in GLACIERS:
        return key, GLACIERS[key]
    for k, v in GLACIERS.items():
        if v["static"].lower() == key:
            return k, v
    sys.exit(f"FEHLER: '{name}' unbekannt. Bekannt: {sorted(GLACIERS)}")


def pick(ds, key, required=True):
    """Erste vorhandene Namensvariante zurueckgeben."""
    for n in CAND[key]:
        if n in ds.variables:
            return n
    if required:
        sys.exit(
            f"FEHLER: keine Variable fuer '{key}' gefunden.\n"
            f"  Gesucht: {CAND[key]}\n"
            f"  Vorhanden: {sorted(ds.variables)}\n"
            f"  -> Namen oben in CAND ergaenzen."
        )
    return None


def sat_vp_Pa(T_K):
    """Saettigungsdampfdruck [Pa], Sonntag, Eis/Wasser-Umschaltung bei 273.16 K.
    Dieselbe Logik wie rh_from_q in toposcale2cosipy (COSIPY-Konvention)."""
    Tc = np.asarray(T_K) - 273.15
    ew = 611.2 * np.exp(17.62 * Tc / (243.12 + Tc))
    ei = 611.2 * np.exp(22.46 * Tc / (272.62 + Tc))
    return np.where(np.asarray(T_K) >= 273.16, ew, ei)


def cell_elevation(cdir, latn, lonn, glat, lon_q, files, sp_mean_hPa):
    """Orographiehoehe der ERA5-Gitterzelle [m].

    1. SURF-z (Oberflaechen-Geopotential) -- der direkte Weg.
    2. Faellt das aus (in unseren ARCO-Downloads ist z angelegt, aber nie
       gefuellt), aus den PLEV-Dateien: geopotentielle Hoehe der Druckflaechen
       auf den mittleren Bodendruck derselben Zelle interpoliert. Genau die
       Groesse, die TopoPyScale intern ohnehin verwendet -- deshalb ist sie
       verfuegbar und konsistent.
    """
    # --- 1) SURF-z ---
    z_present = False
    for f in files:
        d1 = xr.open_dataset(f)
        if "z" in d1.variables:
            z_present = True
            zv = d1["z"].sel({latn: glat, lonn: lon_q}, method="nearest").values
            zval = float(np.asarray(zv, dtype=float).ravel()[0])
            d1.close()
            if np.isfinite(zval):
                print(f"  ERA5-Zellhoehe = {zval/G0:.0f} m  [SURF-z, {Path(f).name}]")
                return zval / G0
            break            # z da, aber NaN -> nicht 456 Dateien durchprobieren
        d1.close()
    if z_present:
        print("  SURF-z ist vorhanden, aber NaN -> weiche auf PLEV aus")
    else:
        print("  kein SURF-z -> weiche auf PLEV aus")

    # --- 2) PLEV-z auf den Bodendruck interpolieren ---
    pfiles = sorted(glob.glob(str(cdir / "PLEV_*.nc")))
    if not pfiles:
        print(f"  WARNUNG: keine PLEV_*.nc in {cdir} -> Zellhoehe unbekannt")
        return np.nan
    dp = xr.open_dataset(pfiles[0])
    if "z" not in dp.variables or "level" not in dp.coords:
        print("  WARNUNG: PLEV ohne z/level -> Zellhoehe unbekannt")
        dp.close()
        return np.nan

    pl = dp["z"].sel({latn: glat, lonn: lon_q}, method="nearest")
    if "time" in pl.dims:
        pl = pl.mean("time")
    lev = np.asarray(dp["level"].values, dtype=float)              # hPa
    zh = np.asarray(pl.values, dtype=float).ravel() / G0           # m
    dp.close()

    o = np.argsort(lev)                    # steigender Druck = fallende Hoehe
    lev, zh = lev[o], zh[o]
    ok = np.isfinite(zh)
    if ok.sum() < 2:
        print("  WARNUNG: PLEV-z unbrauchbar -> Zellhoehe unbekannt")
        return np.nan
    lev, zh = lev[ok], zh[ok]
    if not (lev.min() <= sp_mean_hPa <= lev.max()):
        print(f"  WARNUNG: Bodendruck {sp_mean_hPa:.0f} hPa ausserhalb der Level "
              f"({lev.min():.0f}-{lev.max():.0f}) -> extrapoliert")
    elev = float(np.interp(sp_mean_hPa, lev, zh))
    print(f"  ERA5-Zellhoehe = {elev:.0f} m  [PLEV-z bei {sp_mean_hPa:.0f} hPa, "
          f"{Path(pfiles[0]).name}]")
    return elev


def load_surf(glacier, glat, glon, start_date, end_date, accum_hours):
    """ERA5-Oberflaechenfelder an der naechsten Gitterzelle als DataFrame."""
    cdir = PROJ / glacier / "inputs" / "climate" / "yearly"
    files = sorted(glob.glob(str(cdir / "SURF_*.nc")))
    if not files:
        sys.exit(f"FEHLER: keine SURF_*.nc in {cdir}")
    print(f"{glacier}: {len(files)} SURF-Dateien")

    # Gitter-Konsistenz pruefen: weichen lat/lon zwischen Monatsdateien auch nur
    # in der float32-Darstellung ab, erzeugt der Standard-Outer-Join ein
    # Vereinigungsgitter voller NaN. Bei identischem Gitter ist join="override"
    # korrekt UND verhindert genau das.
    d0 = xr.open_dataset(files[0])
    ln0 = "latitude" if "latitude" in d0.coords else "lat"
    lo0 = "longitude" if "longitude" in d0.coords else "lon"
    ref_lat, ref_lon = d0[ln0].values, d0[lo0].values
    d0.close()
    same_grid = True
    for f in files:
        dk = xr.open_dataset(f)
        # EXAKTER Vergleich: xarray richtet auf identischen Koordinatenwerten
        # aus, nicht auf "nah genug". np.allclose wuerde einen 1e-4-Versatz
        # durchwinken, der das Alignment trotzdem zerlegt.
        if (not np.array_equal(dk[ln0].values, ref_lat)
                or not np.array_equal(dk[lo0].values, ref_lon)):
            same_grid = False
            print(f"  WARNUNG: {Path(f).name} hat ein abweichendes Gitter")
        dk.close()

    if not same_grid:
        sys.exit("FEHLER: die SURF-Dateien haben uneinheitliche lat/lon-Gitter.\n"
                 "  xarray wuerde daraus ein Vereinigungsgitter voller NaN bauen\n"
                 "  (oder mit 'monotonic global indexes' abbrechen).\n"
                 "  -> SURF-Download pruefen: alle Monate muessen aus derselben\n"
                 "     Bounding-Box mit identischem Gitter stammen.")
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="minimal",
                           coords="minimal", compat="override", join="override")

    # Koordinatennamen tolerant behandeln
    latn = "latitude" if "latitude" in ds.coords else "lat"
    lonn = "longitude" if "longitude" in ds.coords else "lon"
    lon_q = glon
    if float(ds[lonn].min()) >= 0 and lon_q < 0:      # 0..360-Konvention
        lon_q = lon_q % 360.0

    di = ds.sel({latn: glat, lonn: lon_q}, method="nearest")
    print(f"  Gitterzelle: {latn}={float(di[latn]):.3f}, {lonn}={float(di[lonn]):.3f} "
          f"(angefragt {glat:.3f}, {glon:.3f})")
    if start_date or end_date:
        di = di.sel(time=slice(start_date, end_date))

    t = pd.to_datetime(di.time.values)
    out = pd.DataFrame(index=t)

    n_t2m = pick(di, "t2m")
    out["T2"] = np.asarray(di[n_t2m].values, dtype=float)

    n_sp = pick(di, "sp")
    sp = np.asarray(di[n_sp].values, dtype=float)
    out["PRES"] = sp / 100.0                                   # Pa -> hPa

    # Feuchte: bevorzugt Taupunkt, sonst spezifische Feuchte
    n_d2m = pick(di, "d2m", required=False)
    if n_d2m is not None:
        e = sat_vp_Pa(np.asarray(di[n_d2m].values, dtype=float))
        out["RH2"] = 100.0 * e / sat_vp_Pa(out["T2"].values)
        print(f"  Feuchte aus Taupunkt ({n_d2m})")
    else:
        n_q = pick(di, "q")
        q = np.asarray(di[n_q].values, dtype=float)
        e = q * sp / (0.622 + 0.378 * q)
        out["RH2"] = 100.0 * e / sat_vp_Pa(out["T2"].values)
        print(f"  Feuchte aus spezifischer Feuchte ({n_q})")

    n_u, n_v = pick(di, "u10"), pick(di, "v10")
    out["U2"] = np.hypot(np.asarray(di[n_u].values, dtype=float),
                         np.asarray(di[n_v].values, dtype=float))

    # Akkumulierte Felder -> Flussdichte / Summe pro Zeitschritt
    a = float(accum_hours) * 3600.0
    n_ssrd, n_strd, n_tp = pick(di, "ssrd"), pick(di, "strd"), pick(di, "tp")
    out["G"]    = np.asarray(di[n_ssrd].values, dtype=float) / a      # J/m2 -> W/m2
    out["LWin"] = np.asarray(di[n_strd].values, dtype=float) / a
    out["RRR"]  = np.asarray(di[n_tp].values, dtype=float) * 1000.0   # m -> mm

    # Zellhoehe braucht den mittleren Bodendruck -> hier, nach out["PRES"]
    z_elev = cell_elevation(cdir, latn, lonn, glat, lon_q, files,
                            float(out["PRES"].mean()))

    print(f"  Zeit: {t[0]} .. {t[-1]}  ({len(t)} Schritte)")

    # Akkumulationsfenster verifizieren statt annehmen: ein Faktor-24-Fehler
    # (taeglich statt stuendlich akkumuliert) faellt hier sofort auf.
    sw_mean, lw_mean = out["G"].mean(), out["LWin"].mean()
    print(f"  Akkumulationsfenster {accum_hours} h -> SWin-Mittel "
          f"{sw_mean:.1f} W/m2, LWin-Mittel {lw_mean:.1f} W/m2")
    if not (80.0 <= sw_mean <= 400.0):
        print(f"  WARNUNG: SWin-Tagesmittel {sw_mean:.1f} W/m2 ist unplausibel "
              f"(erwartet ~150-300). --accum-hours pruefen!")
    if not (120.0 <= lw_mean <= 400.0):
        print(f"  WARNUNG: LWin-Mittel {lw_mean:.1f} W/m2 ist unplausibel "
              f"(erwartet ~180-300). --accum-hours pruefen!")
    u_max = out["U2"].max()
    print(f"  Wind: Mittel {out['U2'].mean():.2f}, Max {u_max:.2f} m/s")
    if u_max < 8.0:
        print(f"  WARNUNG: Maximalwind {u_max:.2f} m/s ist fuer eine mehrjaehrige "
              f"Hochgebirgsreihe zu niedrig (ERA5 erreicht dort ueblich 15-25). "
              f"u10/v10 im SURF-Download pruefen!")
    return out, z_elev


def check_var(name, arr, lo, hi):
    x = np.asarray(arr)[np.isfinite(arr)]
    n_out = int(np.sum(x < lo) + np.sum(x > hi))
    tag = "[ok]" if n_out == 0 else f"AUSSERHALB: {n_out} ({100*n_out/x.size:.3f}%)  <-- PRUEFEN"
    print(f"  {name:5s} min={x.min():8.2f} max={x.max():8.2f}  {tag}")


def build(a):
    proj, info = resolve(a.glacier)
    cap = info["static"]
    if a.stationLat is not None and a.tcart is not None:
        glat, glon = a.stationLat, -a.tcart
    else:
        glat, glon = info["lat"], info["lon"]

    static_file = a.static_file or str(STATIC_DIR / cap / f"{cap}_combined_SRF_1D20m.nc")
    years = a.years or f"{(a.start_date or '1987')[:4]}_{(a.end_date or '2024')[:4]}"
    output = a.output or str(INPUT_DIR / cap / f"{cap}_ERA5raw_1D20m_{years}.nc")

    print(f"Gletscher: Projekt '{proj}', Static '{cap}', {glat}, {glon}")
    print(f"  Statik : {static_file}")
    print(f"  Ausgabe: {output}")
    if not os.path.exists(static_file):
        sys.exit(f"FEHLER: Statikdatei nicht gefunden: {static_file}")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    df, z_elev = load_surf(proj, glat, glon, a.start_date, a.end_date,
                           a.accum_hours)
    if a.era5_elev is not None:
        z_elev = float(a.era5_elev)
        print(f"  ERA5-Zellhoehe manuell gesetzt: {z_elev:.0f} m")

    ds_static = xr.open_dataset(static_file)
    n_band = ds_static.sizes["lat"]
    n_time = len(df)
    tvals = df.index.values
    hgt = ds_static["HGT"].values.flatten().astype(float)
    print(f"  {n_band} Baender, HGT {hgt.min():.0f}-{hgt.max():.0f} m "
          f"-> ALLE bekommen denselben ERA5-Zellwert (das ist der Punkt)")
    if np.isfinite(z_elev):
        print(f"  Hoehenversatz ERA5-Zelle ({z_elev:.0f} m) gegen Baender: "
              f"{hgt.min()-z_elev:+.0f} bis {hgt.max()-z_elev:+.0f} m "
              f"-- das ist der Fehler, den das Downscaling behebt")

    # Physikalische Grenzen wie in toposcale2cosipy
    df["RRR"]  = df["RRR"].clip(lower=0.0)
    df["U2"]   = df["U2"].clip(lower=0.0)
    df["RH2"]  = df["RH2"].clip(0.0, 100.0)
    df["LWin"] = df["LWin"].clip(lower=0.0)
    df["G"]    = df["G"].clip(lower=0.0)

    # NaN-Check: COSIPY bricht bei NaN im Forcing ab -- lieber hier scheitern.
    nan_report = {v: int(df[v].isna().sum()) for v in
                  ["T2", "RH2", "U2", "PRES", "RRR", "LWin", "G"]}
    if any(nan_report.values()):
        print("FEHLER: NaN im Forcing:")
        for v, n in nan_report.items():
            if n:
                print(f"    {v}: {n} von {len(df)} ({100*n/len(df):.2f}%)")
        sys.exit("Abbruch -- NaN deuten auf uneinheitliche SURF-Gitter hin.")
    print("  NaN-Check: keine fehlenden Werte")

    print("Wertebereiche (COSIPY-Grenzen):")
    for nm, lo, hi in [("T2",223.16,316.16), ("RH2",0.0,100.0), ("U2",0.0,50.0),
                       ("PRES",200.0,1080.0), ("RRR",0.0,20.0),
                       ("LWin",0.0,400.0), ("G",0.0,1600.0)]:
        check_var(nm, df[nm].values, lo, hi)

    dso = xr.Dataset()
    dso.coords["lat"] = ds_static["lat"]
    dso.coords["lon"] = ds_static["lon"]
    dso.coords["time"] = ("time", tvals)

    for v in ["HGT", "ASPECT", "SLOPE", "MASK"]:
        if v in ds_static:
            dso[v] = (("lat", "lon"), ds_static[v].values)
            dso[v].attrs = dict(ds_static[v].attrs)

    sim_times = pd.to_datetime(tvals)
    for v in ["SRF", "N_Points"]:
        if v not in ds_static:
            continue
        if "time" in ds_static[v].dims:
            sparse = pd.to_datetime(ds_static["time"].values)
            idx = np.clip(np.searchsorted(sparse, sim_times, side="right") - 1,
                          0, len(sparse) - 1)
            dso[v] = (("time", "lat", "lon"), ds_static[v].values[idx])
        else:
            dso[v] = (("lat", "lon"), ds_static[v].values)
        dso[v].attrs = dict(ds_static[v].attrs)

    units = {"T2": ("K", "Temperature at 2 m"),
             "RH2": ("%", "Relative humidity at 2 m"),
             "U2": ("m s-1", "Wind velocity at 2 m"),
             "G": ("W m-2", "Incoming shortwave radiation"),
             "PRES": ("hPa", "Atmospheric Pressure"),
             "RRR": ("mm", "Total precipitation"),
             "LWin": ("W m-2", "Incoming longwave radiation")}
    for v, (u, ln) in units.items():
        arr = np.repeat(df[v].values[:, None], n_band, axis=1)   # (time, band)
        dso[v] = (("time", "lat", "lon"), arr.reshape(n_time, n_band, 1))
        dso[v].attrs = {"units": u, "long_name": ln}

    dso.attrs["comment"] = (
        "RAW ERA5 grid cell replicated across all elevation bands. "
        "No vertical downscaling, no lapse rate, no HORAYZON terrain "
        "correction on SW, no LW terrain term. Control run for the "
        "downscaling comparison.")
    dso.attrs["era5_gridcell_lat"] = glat
    dso.attrs["era5_gridcell_lon"] = glon
    if np.isfinite(z_elev):
        dso.attrs["era5_gridcell_elevation_m"] = z_elev

    print(f"Schreibe {output}")
    enc = {v: {"zlib": True, "complevel": 4} for v in dso.data_vars}
    dso.to_netcdf(output, encoding=enc)
    print("Fertig.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glacier", required=True,
                   help="Projektname (mera) oder Static-Name (Mera)")
    p.add_argument("-o", "--output", default=None,
                   help="optional; Standard: data/input/<Static>/<Static>_ERA5raw_1D20m_<years>.nc")
    p.add_argument("-s", "--static_file", default=None,
                   help="optional; Standard: data/static/<Static>/<Static>_combined_SRF_1D20m.nc")
    p.add_argument("--years", default=None,
                   help="Jahres-Suffix im Ausgabenamen, z.B. 1987_2024")
    p.add_argument("-b", "--start_date", default=None)
    p.add_argument("-e", "--end_date", default=None)
    p.add_argument("--stationLat", type=float, default=None,
                   help="optional; sonst aus GLACIER_LATLON")
    p.add_argument("--tcart", type=float, default=None,
                   help="= -Laenge des Gletschers; optional, sonst GLACIER_LATLON")
    p.add_argument("--era5-elev", dest="era5_elev", type=float, default=None,
                   help="Orographiehoehe der ERA5-Zelle [m], falls z in den "
                        "SURF-Dateien leer ist. Nur fuer die Diagnose-Ausgabe "
                        "und das Datei-Attribut -- die Werte selbst aendert es nicht.")
    p.add_argument("--accum-hours", dest="accum_hours", type=float, default=1.0,
                   help="Akkumulationsfenster von ssrd/strd/tp in Stunden (ARCO: 1)")
    build(p.parse_args())


if __name__ == "__main__":
    main()
