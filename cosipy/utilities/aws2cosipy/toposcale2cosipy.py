#!/usr/bin/env python3
"""
toposcale2cosipy.py
===================

Baut ein COSIPY-Forcing-NetCDF (HEF-Format: dims lat=Baender, lon=1, time)
aus den TopoPyScale down_pt_*.nc Baendern eines Gletschers.

Unterschied zu cosmo2cosipy.py
-------------------------------
cosmo2cosipy nimmt EINE Punkt-Zeitreihe (CSV) und verteilt sie per lokaler
Lapse Rates ueber die Hoehenbaender. TopoPyScale hat die Baender aber SCHON
vertikal aufgeloest -> die Lapse-Rate-Verteilung entfaellt komplett.
Dieses Skript stapelt die down_pt-Baender direkt; der HORAYZON/Moelg-SW-Block
ist bit-identisch aus cosmo2cosipy uebernommen (Horayzon2022-Pfad).

Variablen-Mapping (down_pt -> COSIPY)
-------------------------------------
  t   (K)      -> T2   (K)          direkt
  q,t,p        -> RH2  (%)          via Sonntag Eis/Wasser (wie COSIPY intern)
  ws  (m/s)    -> U2   (m/s)        direkt
  p   (Pa)     -> PRES (hPa)        / 100   (Attribut 'bar' ist falsch)
  tp  (mm/hr)  -> RRR  (mm)         direkt
  LW  (W/m2)   -> LWin (W/m2)       direkt (ERA5, LW_terrain=False)
  SW  (W/m2)   -> G_meas            rohes ERA5-SWin (ssrd/tstep), Erbs-Split
                                    SW_direct/SW_diffuse wird VERWORFEN

SWin (Horayzon2022):
  G = sw_dir_cor * (1-f_dif) * SW  +  svf * f_dif * SW
  Moelg 2009 liefert nur das Verhaeltnis f_dif; die Energie SW ist ERA5.

Aufruf
------
  python toposcale2cosipy.py --glacier abramov \\
      -o Abramov_ERA5_1D20m_HORAYZON_1987_2024.nc \\
      -s data/static/Abramov/Abramov_RGI6_SRF_1D20m.nc \\
      --sw data/static/Abramov/Abramov_RGI6_HORAYZON-LUT_1D20m.nc \\
      -b 1987-10-01 -e 2024-12-31 \\
      --stationLat 39.61 --tcart -71.55

stationLat und tcart (= -glacier_lon) je Gletscher setzen -- am einfachsten
per CLI (unten) statt utilities_config.toml, damit das SLURM-Skript sie
einfach uebergeben kann. --forcing-utc-offset default 0 (ERA5 ist UTC).
"""

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# COSIPY-Umgebung fuer mod_radCor
sys.path.append("/data/scratch/richteny/thesis/cosipy_test_space/")
import cosipy.modules.radCor as mod_radCor

PROJ = Path("/data/scratch/richteny/projects")

# Moelg (2009) physikalische Konstanten -- identisch zu calcRad in radCor.py
SOL0   = 1367.0      # Solarkonstante [W m-2]
AESC1  = 0.87764     # Aerosol-Transmissivitaet Meereshoehe
AESC2  = 2.4845e-5   # Aerosol-Transmissivitaet Hoehengradient [m-1]
ALPHSS = 0.9         # Aerosol single-scattering albedo
DIROVC = 0.00        # Direktstrahlungsanteil bei Vollbedeckung
DIF1   = 4.6         # Min. Diffus [%] des Klarhimmel-Global bei cld=0
DIFRA  = 0.66        # Diffusstrahlungskonstante
CF     = 0.65        # Wolkeneinfluss-Konstante
SW_DIR_COR_MAX = 25.0
G_CAP = 1600.0        # Sicherheits-Cap fuer G [W m-2] = COSIPYs check-Grenze
                      # (laesst Backscattering ueber hochalbedischem Eis zu)


# ══════════════════════════════════════════════════════════════════════════
#  Feuchte: Sonntag Eis/Wasser (identisch zu COSIPY method_EW_Sonntag)
# ══════════════════════════════════════════════════════════════════════════
def rh_from_q(q, T_K, p_Pa):
    mr = q / (1.0 - q)
    e = mr * p_Pa / (0.62197 + mr)
    ew = 611.2 * np.exp(17.67 * (T_K - 273.16) / (T_K - 29.66))   # ueber Wasser
    ei = 611.2 * np.exp(22.46 * (T_K - 273.16) / (T_K - 0.55))    # ueber Eis
    es = np.where(T_K >= 273.16, ew, ei)
    return np.clip(100.0 * e / es, 0.0, 100.0)


def check_var(name, arr, lo, hi):
    """Zaehlt und meldet Werte ausserhalb [lo, hi] (COSIPY-check_data-Stil).

    Meldet Minimum, Maximum, wie viele Werte unter/ueber der Grenze liegen
    und den Anteil. Warnt deutlich, wenn Grenzen verletzt werden -- wirft aber
    NICHT ab (die Caps im SW-Block haben G bereits begrenzt; fuer die anderen
    Variablen ist es ein Hinweis auf Datenprobleme)."""
    x = arr[np.isfinite(arr)]
    n = x.size
    mn, mx = float(np.min(x)), float(np.max(x))
    n_lo = int(np.sum(x < lo))
    n_hi = int(np.sum(x > hi))
    n_out = n_lo + n_hi
    if n_out == 0:
        print(f"  {name:5s} min={mn:8.2f} max={mx:8.2f}  [ok]")
    else:
        pct = 100.0 * n_out / n
        print(f"  {name:5s} min={mn:8.2f} max={mx:8.2f}  "
              f"AUSSERHALB: {n_out} ({pct:.3f}%)  "
              f"[<{lo:g}: {n_lo}, >{hi:g}: {n_hi}]  <-- PRUEFEN")


# ══════════════════════════════════════════════════════════════════════════
#  down_pt-Baender laden und zu (time, band) stapeln
# ══════════════════════════════════════════════════════════════════════════
def load_bands(glacier, start_date, end_date):
    ddir = PROJ / glacier / "outputs" / "downscaled"
    files = sorted(glob.glob(str(ddir / "down_pt*.nc")),
                   key=lambda f: int(Path(f).stem.split("_")[-1]))
    if not files:
        sys.exit(f"FEHLER: keine down_pt-Dateien in {ddir}")
    print(f"{glacier}: {len(files)} Baender")

    varmap = ["t", "q", "p", "ws", "tp", "LW", "SW"]
    cols, time_ref = {}, None
    for k, f in enumerate(files):
        d = xr.open_dataset(f)
        if start_date or end_date:
            d = d.sel(time=slice(start_date, end_date))
        if time_ref is None:
            time_ref = pd.to_datetime(d.time.values)
            for v in varmap:
                cols[v] = np.full((len(time_ref), len(files)), np.nan)
        for v in varmap:
            if v not in d:
                sys.exit(f"FEHLER: '{v}' fehlt in {Path(f).name}")
            cols[v][:, k] = d[v].values
        d.close()
    print(f"  Zeit: {time_ref[0]} .. {time_ref[-1]}  ({len(time_ref)} Schritte)")
    return cols, time_ref, len(files)


# ══════════════════════════════════════════════════════════════════════════
#  SHORTWAVE -- Horayzon2022-Kern, bit-identisch aus cosmo2cosipy.py
#  (Debug-Prints entfernt; Konstanten als Modulkonstanten oben)
# ══════════════════════════════════════════════════════════════════════════
def compute_shortwave(sw, T_interp, RH_interp, P_interp, heights,
                      time_index_vals, corr_file, station_lat, tcart,
                      forcing_utc_offset, sw_starts, npoints_per_lut=None):
    """
    sw, T_interp, RH_interp, P_interp: (time, band, 1)
    heights: (band, 1)  -- aus static HGT
    Rueckgabe: G_interp (time, band, 1)
    """
    df_index = pd.to_datetime(time_index_vals)
    time_index = len(df_index)
    lat_index  = sw.shape[1]
    lon_index  = sw.shape[2]
    G_interp   = np.zeros((time_index, lat_index, lon_index), dtype=np.float64)
    svf_time   = np.zeros((time_index, lat_index, lon_index), dtype=np.float64)
    idx_used   = np.zeros(time_index, dtype=np.int32)   # welcher Umriss je Schritt

    if corr_file is None or len(corr_file) == 0:
        sys.exit("FEHLER: keine HORAYZON-LUT via --sw uebergeben.")

    lut_datasets = [xr.open_dataset(f) for f in corr_file]
    n_luts = len(lut_datasets)

    # --- Jahr->LUT-Mapping (--sw-starts) ---
    if n_luts == 1:
        sw_starts_sorted = []
    else:
        if not sw_starts or len(sw_starts) == 0:
            sys.exit(f"{n_luts} LUTs aber keine --sw-starts (brauche {n_luts-1}).")
        sw_starts_sorted = sorted(int(y) for y in sw_starts)
        if len(sw_starts_sorted) != n_luts - 1:
            sys.exit(f"--sw-starts hat {len(sw_starts_sorted)}, brauche {n_luts-1}.")

    # --- SVF pro LUT ---
    svf_per_lut = []
    for k, ds_lut in enumerate(lut_datasets):
        if "svf" in ds_lut:
            s = ds_lut["svf"].values.astype(np.float64)
            svf_per_lut.append(s)
            print(f"  LUT {k}: SVF-Mittel = {np.nanmean(s):.3f}")
        else:
            svf_per_lut.append(None)
    first_svf = next((s for s in svf_per_lut if s is not None), None)
    if first_svf is None:
        print("WARNUNG: kein SVF in LUT -- diffuse Strahlung nicht terrain-korrigiert.")
    else:
        svf_per_lut = [s if s is not None else first_svf for s in svf_per_lut]

    # --- NaN in SVF VERIFIZIERT auf 0 setzen ---
    # NaN entsteht nur in leeren Baendern (N_Points=0 nach Gletscherrueckzug),
    # wo die HORAYZON-Aggregation ueber eine leere Maske NaN gibt. Diese Baender
    # tragen nichts zur Massenbilanz bei -> G=0 dort korrekt. ABER: ein NaN bei
    # N_Points>0 waere ein ECHTER Datenfehler (Eis vorhanden, SVF kaputt) und
    # darf NICHT stillschweigend auf 0 gesetzt werden. Deshalb pro Umriss gegen
    # N_Points pruefen, bevor NaN->0. sw_dir_cor wird im Loop per nan_to_num
    # bereinigt (dieselben leeren Baender); die Band-genaue Verifikation
    # erfolgt an SVF (eindeutig band-aufgeloest).
    if npoints_per_lut is not None and len(npoints_per_lut) != n_luts:
        sys.exit(f"FEHLER: {len(npoints_per_lut)} SRF-Umrisse != {n_luts} LUTs "
                 f"— Reihenfolge/Anzahl passt nicht, N_Points-Verifikation "
                 f"nicht moeglich.")
    svf_clean = []
    for k in range(n_luts):
        if first_svf is None:
            svf_clean.append(None)
            continue
        svf_k = svf_per_lut[k].reshape(-1)
        nan_mask = np.isnan(svf_k)
        if nan_mask.any():
            npt = npoints_per_lut[k].reshape(-1) if npoints_per_lut is not None else None
            if npt is None:
                print(f"  WARNUNG LUT {k}: {int(nan_mask.sum())} NaN-SVF-Baender, "
                      f"keine N_Points zur Verifikation -> auf 0 gesetzt")
            else:
                bad = nan_mask & (npt > 0)        # NaN TROTZ Eis -> echter Fehler
                if bad.any():
                    bands = list(np.where(bad)[0])
                    sys.exit(f"FEHLER LUT {k}: NaN-SVF in Baendern {bands} MIT "
                             f"N_Points>0 — echter Datenfehler, nicht nur leeres "
                             f"Band! Abbruch statt stillem 0-Setzen.")
                empty = nan_mask & (npt == 0)
                print(f"  LUT {k}: {int(empty.sum())} NaN-SVF-Band(er) mit "
                      f"N_Points=0 (leer) -> auf 0 gesetzt: "
                      f"{list(np.where(empty)[0])}")
        svf_clean.append(np.nan_to_num(svf_per_lut[k], nan=0.0))

    # --- time_id-Index auf jeder LUT (Referenzjahr 2020, UTC) ---
    processed_luts = []
    for ds_lut in lut_datasets:
        clip = np.where(ds_lut["sw_dir_cor"].values > SW_DIR_COR_MAX,
                        SW_DIR_COR_MAX, ds_lut["sw_dir_cor"].values)
        ds_lut = ds_lut.assign(sw_dir_cor=(("time", "lat", "lon"), clip))
        time_id = (ds_lut.time.dt.dayofyear.values
                   + ds_lut.time.dt.hour.values / 100.0)
        ds_lut = ds_lut.assign_coords(time_id=("time", time_id))
        ds_lut = ds_lut.swap_dims({"time": "time_id"})
        processed_luts.append(ds_lut)

    solPars, timeCorr = mod_radCor.solpars(station_lat)

    # kein N (Bewoelkung) im Forcing -> Klarhimmel-Diffus-Verhaeltnis
    has_cloud = False
    print("Hinweis: keine Wolkenfraktion N im Forcing -> f_dif aus Klarhimmel "
          "(Dcs/grcs). Fuer bewoelkten Himmel leicht unterschaetzt.")

    for t in range(time_index):
        year = df_index[t].year
        hour = df_index[t].hour
        doy  = df_index[t].dayofyear

        # LUT-Lookup in UTC
        t_utc = df_index[t] - pd.Timedelta(hours=forcing_utc_offset)
        doy_lut, hour_lut = t_utc.dayofyear, t_utc.hour
        if t_utc.year % 4 != 0 and doy_lut > 59:
            doy_lut += 1
        time_id_val = doy_lut + hour_lut / 100.0

        if not sw_starts_sorted:
            idx = 0
        else:
            idx = min(int(np.searchsorted(sw_starts_sorted, year, side="right")),
                      n_luts - 1)
        idx_used[t] = idx
        sw_cor_val = processed_luts[idx].sel(time_id=time_id_val)["sw_dir_cor"].values
        # NaN in sw_dir_cor/svf treten in leeren Baendern (N_Points=0 nach
        # Rueckzug) auf -> wurde bereits VOR dem Loop verifiziert (jeder NaN
        # faellt mit N_Points==0 zusammen) und dort auf 0 gesetzt. Die
        # per-Umriss-bereinigten Arrays liegen in svf_clean/swcor_clean.
        sw_cor_val = np.nan_to_num(sw_cor_val, nan=0.0)
        svf_val = svf_clean[idx] if first_svf is not None else None
        if svf_val is not None:
            svf_time[t, :, :] = svf_val

        # Moelg-Solargeometrie (exakt wie calcRad)
        soldec = solPars[doy - 1, 3]
        eccorr = solPars[doy - 1, 2]
        tcorr  = timeCorr[doy - 1, 3]
        stime = (180.0 + (15.0 / 2.0) - hour * 15.0 - tcorr + tcart)
        sin_h = (math.sin(soldec) * math.sin(math.radians(station_lat))
                 + math.cos(soldec) * math.cos(math.radians(station_lat))
                 * math.cos(math.radians(stime)))

        if sin_h <= 0.01:
            G_interp[t, :, :] = 0.0
            continue

        # ERA5-ssrd hat vereinzelt unphysikalische Ausreisser (einzelne
        # Stundenwerte). Oberflaechen-SWin kann die Solarkonstante nicht
        # ueberschreiten. Bewusst FESTER Cap auf SOL0 (nicht Sol0*eccorr*sin_h):
        # so verlassen wir uns NICHT auf die eigene Solargeometrie -- ein
        # etwaiger tcart/UTC-Fehler kann den Cap nicht verfaelschen. Der Fehler
        # bleibt ganz bei ERA5, wird aber durch die Obergrenze begrenzt.
        sw_t = np.minimum(sw[t], SOL0)         # (band,1) auf Solarkonstante begrenzt

        if sw_t.max() <= 0.0:                  # nachts / kein ERA5-SW
            G_interp[t, :, :] = 0.0
            continue

        # Atmosphaerische Transmissivitaeten (Moelg-exakt)
        mopt  = 35.0 * (1224.0 * sin_h**2 + 1.0)**(-0.5)
        p_rel = P_interp[t] / 1013.25
        RH_safe = np.clip(RH_interp[t], 0.0, 100.0)

        # Dampfdruck via Sonntag (statt metpy/Magnus -- konsistent mit RH-Ableitung)
        ew = 611.2 * np.exp(17.67 * (T_interp[t] - 273.16) / (T_interp[t] - 29.66))
        ei = 611.2 * np.exp(22.46 * (T_interp[t] - 273.16) / (T_interp[t] - 0.55))
        es = np.where(T_interp[t] >= 273.16, ew, ei)
        vp = (RH_safe / 100.0) * es / 100.0     # Pa -> hPa

        TAUr = np.exp((-0.09030 * (p_rel * mopt)**0.84)
                      * (1.0 + p_rel * mopt - (p_rel * mopt)**1.01))
        TAUg = math.exp(-0.0127 * mopt**0.26)
        k_aes = np.clip(AESC2 * heights + AESC1, None, 1.0)
        TAUa = k_aes**mopt
        TAUaa = (1.0 - (1.0 - ALPHSS)
                 * (1.0 - p_rel * mopt + (p_rel * mopt)**1.06) * (1.0 - TAUa))
        _w = 46.5 * vp / T_interp[t]
        TAUw = 1.0 - (2.4959 * mopt * _w
                      / ((1.0 + 79.034 * mopt * _w)**0.6828 + 6.385 * mopt * _w))

        taucs = TAUr * TAUg * TAUa * TAUw
        sdir = SOL0 * eccorr * sin_h * taucs                      # Klarhimmel direkt
        Dcs = (DIFRA * SOL0 * eccorr * sin_h * TAUg * TAUw * TAUaa
               * (1.0 - TAUr * TAUa / TAUaa)
               / (1.0 - p_rel * mopt + (p_rel * mopt)**1.02))     # Klarhimmel diffus
        grcs = sdir + Dcs

        # ohne Wolken: theoretische = Klarhimmel
        G_dir_theory = sdir
        G_dif_theory = Dcs

        # Horayzon2022: Moelg-Verhaeltnis auf ERA5-SW anwenden
        G_theory = np.maximum(G_dir_theory + G_dif_theory, 1e-10)
        f_dif = np.clip(G_dif_theory / G_theory, 0.0, 1.0)
        G_dir_meas = sw_t * (1.0 - f_dif)
        G_dif_meas = sw_t * f_dif

        if svf_val is not None:
            g = sw_cor_val * G_dir_meas + svf_val * G_dif_meas
        else:
            g = sw_cor_val * sw_t
        # Der SW-Input ist bereits auf die TOA-Bestrahlung begrenzt (Cap 1),
        # sodass die ERA5-Artefakte weg sind. Hier ein Sicherheits-Cap bei
        # G_CAP = 1600 (COSIPYs eigene check-Grenze) gegen numerische
        # Extremwerte aus der sw_dir_cor-Projektion -- Backscattering ueber
        # hochalbedischem Schnee/Eis bleibt bis 1600 W/m2 erhalten.
        G_interp[t, :, :] = np.clip(g, 0.0, G_CAP)

    # Variiert SVF ueber die Umrisse? (fuer statisch vs. zeitabhaengig)
    svf_varies = False
    if first_svf is not None and len(svf_per_lut) > 1:
        uniq = np.unique(idx_used)
        if uniq.size > 1:
            means = [float(np.nanmean(svf_per_lut[i])) for i in uniq]
            svf_varies = not np.allclose(means, means[0], atol=1e-6)
    return G_interp, first_svf, svf_time, svf_varies


# ══════════════════════════════════════════════════════════════════════════
def build(a):
    cols, tvals, n_band = load_bands(a.glacier, a.start_date, a.end_date)
    n_time = len(tvals)

    ds_static = xr.open_dataset(a.static_file)
    if ds_static.sizes["lat"] != n_band:
        sys.exit(f"FEHLER: static-Baender {ds_static.sizes['lat']} != down_pt {n_band}")

    T2   = cols["t"]
    U2   = cols["ws"]
    PRES = cols["p"] / 100.0
    RRR  = cols["tp"]
    LWin = cols["LW"]
    SW   = cols["SW"]
    RH2  = rh_from_q(cols["q"], cols["t"], cols["p"])

    # Physikalische Untergrenzen erzwingen (Interpolations-/Rundungsartefakte
    # koennen z.B. RRR = -0.00 erzeugen; COSIPY erwartet >= 0).
    RRR  = np.clip(RRR,  0.0, None)          # Niederschlag >= 0
    U2   = np.clip(U2,   0.0, None)          # Windgeschwindigkeit >= 0
    RH2  = np.clip(RH2,  0.0, 100.0)         # relative Feuchte in [0,100]
    LWin = np.clip(LWin, 0.0, None)          # LWin >= 0
    SW   = np.clip(SW,   0.0, None)          # SW >= 0 (Roh; Cap oben im SW-Block)

    # COSIPY-check-Grenzen (identisch zu cosmo2cosipy check_data):
    #   T2 [223.16, 316.16] K, RH2 [0,100] %, U2 [0,50] m/s,
    #   G [0,1600] W/m2, PRES [200,1080] hPa, RRR [0,20] mm, LWin [0,400] W/m2
    print("Wertebereiche (n = Anzahl Werte ausserhalb der COSIPY-Grenzen):")
    for nm, ar, lo, hi in [("T2",T2,223.16,316.16),("RH2",RH2,0.0,100.0),
                           ("U2",U2,0.0,50.0),("PRES",PRES,200.0,1080.0),
                           ("RRR",RRR,0.0,20.0),("LWin",LWin,0.0,400.0),
                           ("SW",SW,0.0,1600.0)]:
        check_var(nm, ar, lo, hi)

    # N_Points pro Umriss aus der SRF-Datei extrahieren, zur Verifikation der
    # NaN-SVF-Bereinigung (jeder NaN-SVF muss mit N_Points=0 zusammenfallen).
    # Reihenfolge der SRF-time-Umrisse == Reihenfolge der --sw LUTs.
    npoints_per_lut = None
    if "N_Points" in ds_static:
        if "time" in ds_static["N_Points"].dims:
            npoints_per_lut = [ds_static["N_Points"].isel(time=k).values
                               for k in range(ds_static.sizes["time"])]
        else:
            npoints_per_lut = [ds_static["N_Points"].values]

    # SHORTWAVE
    heights = ds_static["HGT"].values                # (band,1)
    G_interp, first_svf, svf_time, svf_varies = compute_shortwave(
        sw=SW.reshape(n_time, n_band, 1),
        T_interp=T2.reshape(n_time, n_band, 1),
        RH_interp=RH2.reshape(n_time, n_band, 1),
        P_interp=PRES.reshape(n_time, n_band, 1),
        heights=heights,
        time_index_vals=tvals,
        corr_file=a.corr_file,
        station_lat=a.stationLat,
        tcart=a.tcart,
        forcing_utc_offset=a.forcing_utc_offset,
        sw_starts=a.sw_starts,
        npoints_per_lut=npoints_per_lut,
    )
    print("Nach SW-Block:")
    check_var("G", G_interp, 0.0, 1600.0)

    # ── Ausgabedatensatz aufbauen ──────────────────────────────────────────
    # Statische (lat,lon)-Geometrie uebernehmen. SRF/N_Points koennen eine
    # sparse time-Achse haben (mehrere Umrisse) -> per Forward-Fill auf die
    # volle Simulationszeit expandieren (juengster Umriss <= Simulationsdatum).
    dso = xr.Dataset()
    dso.coords["lat"] = ds_static["lat"]
    dso.coords["lon"] = ds_static["lon"]
    dso.coords["time"] = ("time", tvals)

    # statische Geometrie (immer lat,lon)
    for v in ["HGT", "ASPECT", "SLOPE", "MASK"]:
        if v in ds_static:
            dso[v] = (("lat", "lon"), ds_static[v].values)
            dso[v].attrs = dict(ds_static[v].attrs)

    # SRF / N_Points: Forward-Fill falls time-Dimension vorhanden
    sim_times = pd.to_datetime(tvals)
    for v in ["SRF", "N_Points"]:
        if v not in ds_static:
            continue
        if "time" in ds_static[v].dims:
            sparse = pd.to_datetime(ds_static["time"].values)
            idx_map = np.searchsorted(sparse, sim_times, side="right") - 1
            idx_map = np.clip(idx_map, 0, len(sparse) - 1)
            expanded = ds_static[v].values[idx_map]        # (time, lat, lon)
            dso[v] = (("time", "lat", "lon"), expanded)
            print(f"  {v}: Forward-Fill ueber {len(sparse)} Umrisse "
                  f"{[str(t)[:10] for t in sparse]}")
        else:
            dso[v] = (("lat", "lon"), ds_static[v].values)
        dso[v].attrs = dict(ds_static[v].attrs)

    def put3(name, arr, units, long_name):
        dso[name] = (("time", "lat", "lon"), arr.reshape(n_time, n_band, 1))
        dso[name].attrs = {"units": units, "long_name": long_name}

    put3("T2",   T2,   "K",     "Temperature at 2 m")
    put3("RH2",  RH2,  "%",     "Relative humidity at 2 m")
    put3("U2",   U2,   "m s-1", "Wind velocity at 2 m")
    put3("G",    G_interp[:, :, 0], "W m-2", "Incoming shortwave radiation")
    put3("PRES", PRES, "hPa",   "Atmospheric Pressure")
    put3("RRR",  RRR,  "mm",    "Total precipitation")
    put3("LWin", LWin, "W m-2", "Incoming longwave radiation")

    # SVF: zeitabhaengig, falls es ueber die Umrisse variiert (Multi-Outline),
    # sonst statisch. Analog zu SRF/N_Points, damit die diffuse SW-Korrektur
    # dem Gletscherrueckzug folgt.
    if first_svf is not None:
        if svf_varies:
            dso["SVF"] = (("time", "lat", "lon"), np.nan_to_num(svf_time, nan=0.0))
            dso["SVF"].attrs = {"units": "-",
                                "long_name": "Sky View Factor (HORAYZON), per outline"}
            print("  SVF: zeitabhaengig (variiert ueber Umrisse)")
        else:
            dso["SVF"] = (("lat", "lon"), np.nan_to_num(first_svf, nan=0.0).reshape(n_band, 1))
            dso["SVF"].attrs = {"units": "-", "long_name": "Sky View Factor (HORAYZON)"}
            print("  SVF: statisch")

    print(f"Schreibe {a.output}")
    enc = {v: {"zlib": True, "complevel": 4} for v in dso.data_vars}
    dso.to_netcdf(a.output, encoding=enc)
    print("Fertig.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glacier", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-s", "--static_file", required=True)
    p.add_argument("--sw", dest="corr_file", nargs="+", default=None,
                   help="HORAYZON-LUT(s) mit sw_dir_cor + svf")
    p.add_argument("--sw-starts", dest="sw_starts", type=int, nargs="+", default=None)
    p.add_argument("-b", "--start_date", default=None)
    p.add_argument("-e", "--end_date", default=None)
    p.add_argument("--stationLat", type=float, required=True,
                   help="Gletscherbreite [deg] fuer Moelg-Solargeometrie")
    p.add_argument("--tcart", type=float, required=True,
                   help="= -glacier_lon [deg] (Solarzeit-Offset)")
    p.add_argument("--forcing-utc-offset", dest="forcing_utc_offset",
                   type=int, default=0, help="ERA5 ist UTC -> 0")
    a = p.parse_args()
    build(a)


if __name__ == "__main__":
    main()
