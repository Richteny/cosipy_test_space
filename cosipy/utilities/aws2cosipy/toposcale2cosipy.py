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
CF_MOELG = 0.65      # Moelg-Wolkeneinflusskonstante (Cf in cosmo2cosipy)
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

    # HGT je Band aus pts_list.csv lesen — die Reihenfolge der down_pt-Baender
    # (nach Bandindex) entspricht der pts_list-Reihenfolge (nach Hoehe), NICHT
    # der SRF-Datei (nach lat/lon). Wir brauchen die Hoehe, um die Baender
    # spaeter korrekt an die SRF-Reihenfolge zu matchen.
    pts_path = PROJ / glacier / "pts_list.csv"
    if not pts_path.exists():
        sys.exit(f"FEHLER: pts_list.csv fehlt ({pts_path}) — noetig fuer "
                 f"Hoehen-Zuordnung der Baender.")
    pts = pd.read_csv(pts_path)
    # Reihenfolge nach Bandindex (band_000, band_001, ...) sicherstellen
    pts = pts.sort_values("name").reset_index(drop=True)
    if len(pts) != len(files):
        sys.exit(f"FEHLER: pts_list hat {len(pts)} Eintraege, aber {len(files)} "
                 f"down_pt-Dateien.")
    band_hgt = pts["elevation"].values.astype(float)   # (n_band,) in down_pt-Reihenfolge

    varmap = ["t", "q", "p", "ws", "tp", "LW", "SW", "vp"]
    # Optionale Felder aus dem gepatchten TopoPyScale: SW/LW VOR der
    # Gelaendekorrektur. Sind sie da, rechnet der SW-Block direkt damit und
    # HORAYZON korrigiert genau EINMAL. Fehlen sie, greift der alte Pfad
    # (Moelg-Verhaeltnis auf das bereits korrigierte SW) -- der doppelt
    # korrigiert und nur zur Reproduktion alter Laeufe taugt.
    varmap_opt = ["SW_direct_flat", "SW_diffuse_flat", "LW_flat"]
    cols, time_ref = {}, None
    for k, f in enumerate(files):
        d = xr.open_dataset(f)
        if start_date or end_date:
            d = d.sel(time=slice(start_date, end_date))
        if time_ref is None:
            time_ref = pd.to_datetime(d.time.values)
            have_opt = [v for v in varmap_opt if v in d]
            for v in varmap + have_opt:
                cols[v] = np.full((len(time_ref), len(files)), np.nan)
        for v in varmap:
            if v not in d:
                sys.exit(f"FEHLER: '{v}' fehlt in {Path(f).name}")
            cols[v][:, k] = d[v].values
        for v in have_opt:
            if v not in d:
                sys.exit(f"FEHLER: '{v}' fehlt in {Path(f).name}, war aber in "
                         f"{Path(files[0]).name} vorhanden -- gemischter "
                         f"Downscaling-Stand. Alle down_pt-Dateien neu bauen.")
            cols[v][:, k] = d[v].values
        d.close()
    print(f"  Zeit: {time_ref[0]} .. {time_ref[-1]}  ({len(time_ref)} Schritte)")
    if have_opt:
        print(f"  unkorrigierte Felder gefunden: {', '.join(have_opt)}")
    return cols, time_ref, len(files), band_hgt


# ---------------------------------------------------------------------------
#  Wolkenfelder (TCC, CBH) fuer die Liu-2020-LWin-Parametrisierung laden
# ---------------------------------------------------------------------------
def load_cloud(glacier, glat, glon, time_ref):
    """Laedt TCC (und CBH) aus den CLOUD_YYYY_MM.nc-Dateien am naechsten
    Gitterpunkt zum Gletscher und richtet sie auf die down_pt-Zeitachse aus.
    Rueckgabe: tcc (n_time,), cbh (n_time,) in Metern (NaN bei Klarhimmel).
    Wolken sind Gitterzellen-Werte -> hoehenkonstant, werden spaeter ueber
    alle Baender gebroadcastet."""
    cdir = PROJ / glacier / "inputs" / "climate" / "yearly"
    files = sorted(glob.glob(str(cdir / "CLOUD_*.nc")))
    if not files:
        sys.exit(f"FEHLER: keine CLOUD_*.nc in {cdir} — fuer --lw-method liu* "
                 f"noetig. Erst download_era5_arco_cloud.py laufen lassen.")
    ds = xr.open_mfdataset(files, combine="by_coords")
    di = ds.sel(latitude=glat, longitude=glon, method="nearest")
    di = di.sel(time=slice(time_ref[0], time_ref[-1]))
    # auf die exakte down_pt-Zeitachse reindexen (identische stuendliche Achse
    # erwartet; fehlende Stunden -> NaN, wird unten abgefangen)
    di = di.reindex(time=time_ref)
    tcc = di["tcc"].values.astype(float)
    cbh = di["cbh"].values.astype(float) if "cbh" in di else np.full(len(time_ref), np.nan)
    # TCC-Luecken (falls Reindex NaN erzeugt) auf 0 (klar) setzen — konservativ
    n_gap = int(np.isnan(tcc).sum())
    if n_gap:
        print(f"  WARNUNG: {n_gap} TCC-Zeitschritte ohne Wert -> 0 (klar) gesetzt")
        tcc = np.nan_to_num(tcc, nan=0.0)
    tcc = np.clip(tcc, 0.0, 1.0)
    print(f"  Wolken geladen: TCC mean={np.nanmean(tcc):.3f}, "
          f"CBH gueltig {100*np.mean(~np.isnan(cbh)):.0f}%")
    return tcc, cbh


def lwin_liu(t_band, vp_band, tcc):
    """Liu et al. (2020) LWin, lokal fuer das Tibetische Plateau kalibriert.
    Gl.5 (CF-basiert); Gl.6 (CBH-korrigiert) wurde verworfen, weil ERA5-CBH
    eine andere Groesse als Lius Lidar-CBH ist und die Koeffizienten sich nicht
    uebertragen (LWin wurde unphysikalisch, >600 W/m2 selbst im Kalibrierbereich).

    t_band  : (n_time, n_band) Band-Lufttemperatur [K]
    vp_band : (n_time, n_band) Band-Dampfdruck [Pa]  -> intern /100 = hPa
    tcc     : (n_time,)        cloud fraction 0..1   (hoehenkonstant)

    Klarhimmel (Gl.3): DLR_clr = -2.53 + 158.10*(T/273.16)^6
                                 + 106.40*sqrt(46.50*(e/T)/2.50)   [e in hPa]
    Bewoelkt  (Gl.5):  DLR_cld = (1 + 0.23*CF) * DLR_clr
    Bei CF=0 -> DLR_clr, bei CF=1 -> 1.23*DLR_clr. Tag und Nacht definiert
    (nutzt nur CF, kein tau_atm).
    """
    e_hPa = vp_band / 100.0                                  # Pa -> hPa
    w = np.sqrt(46.50 * e_hPa / t_band / 2.50)               # Prata precipitable water
    dlr_clr = -2.53 + 158.10 * (t_band / 273.16)**6 + 106.40 * w
    cf = tcc[:, None]                                        # (n_time,1) broadcast
    return (1.0 + 0.23 * cf) * dlr_clr                       # Gl.5


SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann [W m-2 K-4]

def apply_terrain(lw_sky, svf, t_band, G=None, method="off",
                  eps_terrain=0.98, solar_coeff=0.01):
    """Terrain-Emissionsterm auf das Sky-LWin addieren (Sicart 2006 Gl.6 /
    Prinz 2016 Gl.6):

        LWin = SVF * L_sky  +  (1 - SVF) * eps * sigma * T_terrain^4

    Der (1-SVF)-Anteil der Hemisphaere wird von den umgebenden Haengen
    gefuellt, die naeherungsweise mit Lufttemperatur (oder solar leicht
    aufgeheizt) emittieren. Physik ist rein geometrisch, nicht regionsspezifisch.

    lw_sky : (n_time, n_band) Sky-LWin (topopyscale oder liu-cf)
    svf    : (n_time, n_band) oder (n_band,) Sky-View-Factor 0..1
    t_band : (n_time, n_band) Band-Lufttemperatur [K]
    G      : (n_time, n_band) Globalstrahlung [W m-2], nur fuer method='prinz'
    method : 'off'    -> kein Terrain, LWin = lw_sky (unveraendert)
             'airT'   -> T_terrain = Band-Lufttemperatur
             'prinz'  -> T_terrain = Band-Lufttemperatur + solar_coeff*G
                         (Sicart/Prinz: solare Hangaufheizung, +0.01 K/(W/m2))
    eps_terrain : Terrain-Emissivitaet (0.97 Schnee/Fels, 0.99 quasi-Schwarzk.)
    """
    if method == "off":
        return lw_sky
    svf = np.asarray(svf, dtype=float)
    if svf.ndim == 1:
        svf = svf[None, :]                                   # (1,n_band) broadcast
    t_terr = t_band.copy()
    if method == "prinz":
        if G is None:
            sys.exit("FEHLER: method='prinz' braucht G (Globalstrahlung).")
        t_terr = t_band + solar_coeff * G                    # solare Hangaufheizung
    elif method != "airT":
        sys.exit(f"FEHLER: unbekannte Terrain-Methode '{method}'")
    lw_terrain = (1.0 - svf) * eps_terrain * SIGMA_SB * t_terr**4
    return svf * lw_sky + lw_terrain


# ══════════════════════════════════════════════════════════════════════════
#  SHORTWAVE -- Horayzon2022-Kern, bit-identisch aus cosmo2cosipy.py
#  (Debug-Prints entfernt; Konstanten als Modulkonstanten oben)
# ══════════════════════════════════════════════════════════════════════════
def compute_shortwave(sw, T_interp, RH_interp, P_interp, heights,
                      time_index_vals, corr_file, station_lat, tcart,
                      forcing_utc_offset, sw_starts, npoints_per_lut=None,
                      tcc=None, sw_dir_flat=None, sw_dif_flat=None):
    """
    sw, T_interp, RH_interp, P_interp: (time, band, 1)
    heights: (band, 1)  -- aus static HGT
    tcc: (time,) Wolkenfraktion 0..1 oder None

    Zur Aufteilung direkt/diffus wird das Moelg-2009-Verhaeltnis gebildet und
    auf das gemessene ERA5-SW angewandt. OHNE tcc ist dieses Verhaeltnis das
    KLARHIMMEL-Verhaeltnis (Dcs/grcs ~ 0.13) -- unter Bewoelkung wird der
    Diffusanteil dann massiv unterschaetzt und faelschlich durch sw_dir_cor
    (Abschattung) statt durch SVF geleitet. cosmo2cosipy warnt an dieser
    Stelle ebenfalls, wenn N fehlt.

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

    if sw_dir_flat is not None:
        print("SW: direkter Weg -- SW_direct_flat/SW_diffuse_flat aus "
              "TopoPyScale (Erbs-Split), HORAYZON korrigiert genau einmal.")
    else:
        print("=" * 72)
        print("WARNUNG: keine SW_direct_flat/SW_diffuse_flat in den down_pt-")
        print("         Dateien -> Altpfad. Das eingelesene SW ist von "
              "TopoPyScale")
        print("         bereits projiziert UND abgeschattet, sw_dir_cor "
              "korrigiert")
        print("         ein zweites Mal. Nur zur Reproduktion alter Laeufe.")
        print("=" * 72)

    if sw_dir_flat is not None:
        # Direkter Weg: f_dif wird gar nicht gebildet, die Wolkenfraktion
        # spielt im SW-Block keine Rolle mehr (TopoPyScale hat den Erbs-Split
        # schon gemacht). Also KEINE Wolkenwarnung -- die waere irrefuehrend.
        has_cloud = False
        cld_arr = None
    elif tcc is not None:
        has_cloud = True
        cld_arr = np.clip(np.asarray(tcc, dtype=float), 0.0, 1.0)
        print(f"SW: wolkenabhaengige Diffusaufteilung aktiv "
              f"(Moelg 2009, CF-Mittel {np.nanmean(cld_arr):.3f})")
    else:
        has_cloud = False
        cld_arr = None
        print("SW: WARNUNG -- keine Wolkenfraktion -> f_dif aus Klarhimmel "
              "(Dcs/grcs ~ 0.13). Unter Bewoelkung wird der Diffusanteil stark "
              "unterschaetzt und durch sw_dir_cor abgeschattet statt durch SVF "
              "geleitet. Mit --sw-cloud on aktivieren.")

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

        # Wolkenkorrigierte theoretische Direkt-/Diffusanteile.
        # Identisch zum Horayzon2022-Zweig in cosmo2cosipy.py:
        #   direkt : sdir * (1 - (1-DIROVC)*cld)     -> bei cld=1 und DIROVC=0: 0
        #   diffus : grcs * ((100-CF_M*100-DIF1)/100*cld + DIF1/100)
        # Bei cld=0 wird auf Dcs zurueckgefallen (np.where), damit der
        # Klarhimmelfall exakt der Iqbal/Hastenrath-Formel folgt.
        #
        # BEKANNTE UNSTETIGKEIT (aus cosmo2cosipy uebernommen, bewusst nicht
        # geglaettet, damit beide Skripte bit-identisch rechnen):
        # bei cld -> 0+ liefert die Wolkenformel nur DIF1/100 = 4.6 % Diffus,
        # waehrend Dcs/grcs ~ 13 % ergibt. f_dif springt also von 0.134 (cld=0)
        # auf ~0.05 (cld=0.001) und steigt erst ab cld ~ 0.2 wieder darueber.
        # Bei geringer Bewoelkung ist der Diffusanteil damit KLEINER als im
        # Klarhimmelfall. Wer das glaetten will: G_dif_theory =
        # np.maximum(<Wolkenformel>, Dcs) -- dann weicht das Ergebnis aber von
        # cosmo2cosipy ab.
        if has_cloud:
            cld = cld_arr[t]
            G_dir_theory = np.where(cld > 0, sdir * (1.0 - (1.0 - DIROVC) * cld),
                                    sdir)
            G_dif_theory = np.where(
                cld > 0,
                grcs * ((100.0 - CF_MOELG * 100.0 - DIF1) / 100.0 * cld
                        + DIF1 / 100.0),
                Dcs)
        else:
            G_dir_theory = sdir
            G_dif_theory = Dcs

        if sw_dir_flat is not None:
            # DIREKTER WEG. TopoPyScale liefert Direkt und Diffus getrennt und
            # UNKORRIGIERT (Erbs-Split, wolkenabhaengig). Keine Moelg-Schaetzung
            # noetig -- und vor allem keine Doppelkorrektur, weil beide Felder
            # das Gelaende noch nicht gesehen haben.
            # sw_t ist (band,1); die flat-Arrays sind (time,band) -> [t] gibt
            # (band,). Auf (band,1) bringen, sonst broadcastet (band,1)*(band,)
            # zu (band,band).
            G_dir_meas = sw_dir_flat[t][:, None]
            G_dif_meas = sw_dif_flat[t][:, None]
        else:
            # ALTPFAD: das Moelg-Verhaeltnis auf ein SW anwenden, das
            # TopoPyScale bereits projiziert und abgeschattet hat -> doppelte
            # Gelaendekorrektur. Nur zur Reproduktion alter Laeufe.
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
    # Sicherung gegen versehentliches Ueberschreiben: wird ein Terrain-Term
    # angefordert, der Ausgabename traegt aber kein '-terr...', dann landet das
    # Terrain-Forcing auf dem Namen der Sky-Datei und ueberschreibt sie. Das
    # ist fast immer ein nicht durchgereichtes LW_TERRAIN.
    if a.lw_terrain != "off" and "-terr" not in Path(a.output).name:
        print("=" * 72)
        print(f"WARNUNG: --lw-terrain {a.lw_terrain}, aber der Ausgabename")
        print(f"         '{Path(a.output).name}' enthaelt kein '-terr'.")
        print("         Das Terrain-Forcing ueberschreibt so die Sky-Datei!")
        print("         Im Slurm-Aufruf LW_TERRAIN pruefen (kam es durch?).")
        print("=" * 72)
    print(f"  Ausgabe: {a.output}  (LW={a.lw_method}, Terrain={a.lw_terrain})")

    cols, tvals, n_band, band_hgt = load_bands(a.glacier, a.start_date, a.end_date)
    n_time = len(tvals)

    ds_static = xr.open_dataset(a.static_file)
    if ds_static.sizes["lat"] != n_band:
        sys.exit(f"FEHLER: static-Baender {ds_static.sizes['lat']} != down_pt {n_band}")

    # ── Band-Reihenfolge angleichen ────────────────────────────────────────
    # Die down_pt-Baender (cols) stehen in pts_list-Reihenfolge (nach Hoehe),
    # die SRF-Statikfelder (HGT/MASK/N_Points/SRF/SVF) in lat/lon-Reihenfolge.
    # Ohne Angleich wird das Forcing von Band k neben die HGT einer ANDEREN
    # Hoehe geklebt -> Druck/Temperatur der falschen Hoehe zugeordnet (Druck
    # stieg faelschlich mit der Hoehe). Fix: jede SRF-Position bekommt das
    # down_pt-Band mit passender Hoehe. SRF-Reihenfolge bleibt Referenz (COSIPY
    # erwartet sie so); nur die Forcing-Spalten werden umsortiert.
    hgt_srf = ds_static["HGT"].values.flatten().astype(float)   # (n_band,) SRF-Reihenfolge
    # Fuer jede SRF-Position die passende down_pt-Bandposition finden (per Hoehe)
    order = np.full(n_band, -1, dtype=int)
    used = np.zeros(n_band, dtype=bool)
    for i in range(n_band):
        # naechstes noch unbenutztes down_pt-Band zur SRF-Hoehe hgt_srf[i]
        d = np.abs(band_hgt - hgt_srf[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > 1.0:      # Toleranz 1 m (Baender sind 20 m)
            sys.exit(f"FEHLER: keine Hoehen-Zuordnung fuer SRF-Band {i} "
                     f"(HGT={hgt_srf[i]:.0f} m). Naechste down_pt-Hoehe "
                     f"{band_hgt[j]:.0f} m, Abstand {d[j]:.0f} m > 1 m.")
        order[i] = j
        used[j] = True
    # order[i] = down_pt-Bandindex, der zu SRF-Position i gehoert.
    # Alle Forcing-Spalten in SRF-Reihenfolge umsortieren:
    for v in cols:
        cols[v] = cols[v][:, order]
    band_hgt = band_hgt[order]   # jetzt konsistent mit hgt_srf
    # Verifikation: Hoehen muessen jetzt uebereinstimmen
    if not np.allclose(band_hgt, hgt_srf, atol=1.0):
        sys.exit("FEHLER: Band-Hoehen nach Umsortierung != SRF-HGT — "
                 "Zuordnung fehlgeschlagen.")
    print(f"  Baender an SRF-Reihenfolge angeglichen (Hoehen-Lookup, "
          f"max Abweichung {np.abs(band_hgt-hgt_srf).max():.1f} m)")

    T2   = cols["t"]
    U2   = cols["ws"]
    PRES = cols["p"] / 100.0
    RRR  = cols["tp"]
    SW   = cols["SW"]
    RH2  = rh_from_q(cols["q"], cols["t"], cols["p"])

    # wind speed rises with elevation in topopyscale
    if a.wind_profile != "keep":
        if a.wind_profile == "hypsometric" and "N_Points" in ds_static:
            npv = ds_static["N_Points"].values
            if "time" in ds_static["N_Points"].dims:
                npv = np.nanmean(npv, axis=0)
            w_ref = npv.flatten().astype(float)
        else:
            w_ref = np.ones(n_band, dtype=float)

        ok = np.isfinite(w_ref) & (w_ref > 0) & np.isfinite(hgt_srf)
        if not ok.any():
            sys.exit("Error. No valid bands for wind reference")
        h_ref = float(np.sum(hgt_srf[ok] * w_ref[ok]) / np.sum(w_ref[ok]))

        d = np.abs(hgt_srf - h_ref)
        d[~ok] = np.inf
        i_ref = int(np.argmin(d))

        #gradient check
        u_mean_band = np.nanmean(U2, axis=0)
        slope = (np.polyfit(hgt_srf[ok], u_mean_band[ok], 1)[0] * 100.0
                 if ok.sum() >= 2 else np.nan)

        u_ref = U2[:, i_ref].copy() #n_time, 
        U2 = np.repeat(u_ref[:, None], n_band, axis=1) #ntime, band

        print(f"Wind profile: '{a.wind_profile}' - gradient deleted")
        print(f"Reference height {h_ref:.0f}m -> Band {i_ref} (HGT {hgt_srf[i_ref]:.0f} m,"
              f"{int(w_ref[i_ref])} points)")
        print(f"before dU2/dz {slope:+.3f} m/s per 100m,"
              f"band mean {u_mean_band[ok].min():.2f}-{u_mean_band[ok].max():.2f} m/s")
        print(f"after average {u_ref.mean():.2f} m/s"
              f"Span {u_ref.min():.2f}-{u_ref.max():.2f} m/s over time)")
    else:
        print("Wind profile 'keep' - gradient unchanged")

    # Gletscher-Koordinaten: lat = stationLat, lon = -tcart (tcart = -lon).
    # Zentral gesetzt, weil sowohl der LW- als auch der SW-Block sie brauchen —
    # frueher standen sie nur im liu-cf-Zweig, was --lw-method topopyscale
    # zusammen mit --sw-cloud on mit UnboundLocalError abbrechen liess.
    glat, glon = a.stationLat, -a.tcart

    # ── LWin nach gewaehlter Methode ───────────────────────────────────────
    tcc_lw = None
    if a.lw_method == "topopyscale":
        # TopoPyScale-downscaled Sky-LW. LW_flat ist die Himmelsemission OHNE
        # SVF-Skalierung und damit die einzige Variante, die als Sky-Methode
        # taugt: cols["LW"] ist bereits mit svf multipliziert, ohne dass je ein
        # Terrain-Term addiert wurde, und wuerde von apply_lw_terrain ein
        # zweites Mal skaliert (svf^2 * L).
        if "LW_flat" in cols:
            LWin = cols["LW_flat"]
            print("  LWin-Methode: topopyscale (LW_flat, ohne SVF-Skalierung)")
        else:
            LWin = cols["LW"]
            print("  WARNUNG: kein LW_flat -> nutze cols['LW'], das bereits "
                  "SVF-skaliert ist.\n"
                  "           Struktureller Minderbetrag ~(1-svf)*eps*sigma*T^4 "
                  "(~20 W/m2),\n"
                  "           und --lw-terrain wuerde svf ein zweites Mal "
                  "anwenden.")
            if a.lw_terrain != "off":
                sys.exit("FEHLER: --lw-method topopyscale ohne LW_flat "
                         "zusammen mit --lw-terrain ergibt svf^2 * L_sky. "
                         "Entweder --lw-terrain off, oder TopoPyScale mit "
                         "LW_flat neu laufen lassen.")
    elif a.lw_method == "liu-cf":
        # Liu et al. (2020) Gl.5, TP-kalibriert. Wolken am naechsten Gitterpunkt.
        tcc, _cbh = load_cloud(a.glacier, glat, glon, tvals)
        tcc_lw = tcc
        LWin = lwin_liu(cols["t"], cols["vp"], tcc)
        print(f"  LWin-Methode: {a.lw_method}  (mean {np.nanmean(LWin):.1f} W/m2)")
    else:
        sys.exit(f"FEHLER: unbekannte --lw-method '{a.lw_method}'")

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
    # Wolkenfraktion fuer den SW-Block. Wird ggf. schon fuer liu-cf geladen --
    # dann wiederverwenden statt erneut oeffnen.
    tcc_sw = None
    if a.sw_cloud == "on":
        try:
            tcc_sw = tcc_lw if tcc_lw is not None else load_cloud(
                a.glacier, glat, glon, tvals)[0]
        except SystemExit:
            sys.exit(
                "FEHLER: --sw-cloud on braucht die Wolkenfraktion, aber es "
                "wurden keine CLOUD_*.nc gefunden.\n"
                "  Die HORAYZON/Moelg-Kopplung ist ohne N nicht gueltig: f_dif "
                "faellt auf das Klarhimmel-Verhaeltnis (~0.13) zurueck und die "
                "Terrainabschattung trifft dann die Diffusstrahlung.\n"
                "  Entweder CLOUD_*.nc bereitstellen, oder bewusst "
                "--sw-cloud off setzen (nur zur Reproduktion alter Laeufe).")
    else:
        print("=" * 72)
        print("WARNUNG: --sw-cloud off -- f_dif ist auf das Klarhimmel-"
              "Verhaeltnis (~0.13)")
        print("         eingefroren. Unter Bewoelkung laufen ~85 % des "
              "gemessenen SW durch")
        print("         sw_dir_cor statt durch SVF -> systematisch zu "
              "niedriges SWin.")
        print("         Dieses Forcing dient NUR dem Vergleich, nicht der "
              "Produktion.")
        print("=" * 72)

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
        tcc=tcc_sw,
        sw_dir_flat=cols.get("SW_direct_flat"),
        sw_dif_flat=cols.get("SW_diffuse_flat"),
        forcing_utc_offset=a.forcing_utc_offset,
        sw_starts=a.sw_starts,
        npoints_per_lut=npoints_per_lut,
    )
    print("Nach SW-Block:")
    check_var("G", G_interp, 0.0, 1600.0)

    # ── LWin-Terrain-Term (nach SW-Block, weil svf_time hier vorliegt) ──────
    # LWin = SVF*L_sky + (1-SVF)*eps*sigma*T_terrain^4. svf_time ist (t,band,1),
    # LWin/T2 sind (t,band). svf_time[...,0] auf (t,band) bringen.
    if a.lw_terrain != "off":
        svf_tb = svf_time[:, :, 0]                           # (n_time, n_band)
        LWin = apply_terrain(
            lw_sky=LWin, svf=svf_tb, t_band=T2,
            G=G_interp[:, :, 0] if a.lw_terrain == "prinz" else None,
            method=a.lw_terrain,
            eps_terrain=a.lw_eps, solar_coeff=0.01,
        )
        LWin = np.clip(LWin, 0.0, None)
        print(f"  LWin-Terrain: {a.lw_terrain} (eps={a.lw_eps}) "
              f"-> mean {np.nanmean(LWin):.1f} W/m2")
        check_var("LWin (mit Terrain)", LWin, 0.0, 500.0)

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
    p.add_argument("--lw-method", dest="lw_method", default="topopyscale",
                   choices=["topopyscale", "liu-cf"],
                   help="LWin-Sky-Quelle: topopyscale (down_pt LW, default) oder "
                        "liu-cf (Liu 2020 Gl.5, CF aus TCC).")
    p.add_argument("--lw-terrain", dest="lw_terrain", default="off",
                   choices=["off", "airT", "prinz"],
                   help="LWin-Terrain-Term: off (nur Sky, default), "
                        "airT (T_terrain=Band-Lufttemp), "
                        "prinz (T_terrain=Band-Lufttemp+0.01*G, solare Hangaufheizung).")
    p.add_argument("--sw-cloud", dest="sw_cloud", default="on",
                   choices=["on", "off"],
                   help="Diffusaufteilung im SW-Block. 'on' (DEFAULT) nutzt die "
                        "Moelg-Wolkenkorrektur mit CF aus TCC -- das ist die "
                        "einzige physikalisch korrekte Variante und entspricht "
                        "dem Original-Moelg2009, das N zwingend verlangt. "
                        "'off' friert f_dif auf das KLARHIMMEL-Verhaeltnis (~0.13) "
                        "ein; Terrainabschattung wird dann auf Diffusstrahlung "
                        "angewandt und SWin systematisch unterschaetzt. 'off' "
                        "existiert nur, um aeltere Laeufe fuer den Vergleich zu "
                        "reproduzieren -- NICHT fuer Produktionslaeufe.")
    p.add_argument("--lw-eps", dest="lw_eps", type=float, default=0.98,
                   help="Terrain-Emissivitaet (0.98 = Prinz 2016 / natuerliche "
                        "Oberflaechen; 0.97 Schnee/Fels, 0.99 quasi-Schwarzkoerper).")
    p.add_argument("--wind-profile", dest="wind_profile", default="hypsometric",
                   choices=["hypsometric", "mean", "keep"],
                   help="U2-gradient: 'hypsometric' = Value at N_Points weighted mean of all bands"
                        "'mean' = unweighted mean,"
                        "'keep' = topoypscale profile unchanged.")
    a = p.parse_args()
    build(a)


if __name__ == "__main__":
    main()
