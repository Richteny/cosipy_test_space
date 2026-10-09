#!/bin/bash -l
#SBATCH --job-name="tps2cosipy"
# Tipp: sbatch --export=ALL,LW_TERRAIN=prinz,PRECIP_LR=on slurm_toposcale2cosipy.sh
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --partition=compute
#SBATCH --output=tps2cosipy-%j.out
#SBATCH --error=tps2cosipy-%j.err

conda activate downscaling
export GIT_PYTHON_REFRESH=quiet

BASE=/data/scratch/richteny/thesis/cosipy_test_space
SCRIPT=$BASE/cosipy/utilities/aws2cosipy/toposcale2cosipy.py
STATIC=$BASE/data/static
INPUT=$BASE/data/input
CONFIG=$BASE/cosipy/utilities/aws2cosipy/glacier_config.txt
PROJ=/data/scratch/richteny/projects
START="1987-10-01"
END="2024-12-31"

# LWin-Methode: topopyscale (default) | liu-cf | liu-cbh
LW_METHOD="${LW_METHOD:-topopyscale}"

# SW-Diffusaufteilung:
#   on  = Moelg-Wolkenkorrektur mit CF aus TCC  (korrekt, Default im Skript)
#   off = Klarhimmel-f_dif ~0.13                (NUR zur Reproduktion alter Laeufe)
SW_CLOUD="${SW_CLOUD:-on}"

# LWin-Terrain-Term: off (nur Sky) | airT | prinz
# Dateisuffix -terrairT bzw. -terrprinz, wie bei den bisherigen Terrain-Laeufen.
LW_TERRAIN="${LW_TERRAIN:-off}"
LW_EPS="${LW_EPS:-0.98}"
WIND_PROFILE="${WIND_PROFILE:-hypsometric}"

# Niederschlags-Lapse-Rate im TopoPyScale-Downscaling (config.yml: climate.precip_lapse_rate)
#   on  = down_pt wurde MIT Lapse-Rate gebaut  -> Dateisuffix -lrpr
#   off = down_pt ohne Lapse-Rate              -> kein Suffix (alte Namen)
# Das Skript rechnet die Lapse-Rate NICHT selbst -- sie steckt bereits in tp.
# Die Variable steuert nur den Dateinamen und die Vorpruefung weiter unten.
PRECIP_LR="${PRECIP_LR:-on}"

# Nur die Testgletscher bauen: GLACIERS="mera parlung"
GLACIERS="${GLACIERS:-}"

# Ausgabenamen tragen die Suffixe -terr<...> und -lrpr. Ein Lauf mit
# PRECIP_LR=on kann die Dateien eines frueheren Laufs ohne Lapse-Rate
# also NICHT ueberschreiben -- beide Varianten liegen nebeneinander.
# Innerhalb derselben Einstellungen wird weiterhin ueberschrieben.

if [ ! -f "$CONFIG" ]; then
    echo "FEHLER: Konfigurationsdatei fehlt: $CONFIG"; exit 1
fi

echo "############################################################"
echo "  LW_METHOD = $LW_METHOD"
echo "  LW_TERRAIN= $LW_TERRAIN   (eps=$LW_EPS)"
echo "  SW_CLOUD  = $SW_CLOUD"
echo "  WIND_PROF = $WIND_PROFILE"
echo "  PRECIP_LR = $PRECIP_LR   (Suffix: $([ "$PRECIP_LR" = off ] && echo "(keines)" || echo "-lrpr"))"
echo "  ACHTUNG: gleichnamige Forcings werden ueberschrieben"
[ -n "$GLACIERS" ] && echo "  nur: $GLACIERS"
echo "############################################################"
echo ""

n_ok=0; n_skip=0

while read -r g cap lat lon baseline outlines rest; do
    [[ -z "$g" || "$g" == \#* ]] && continue

    # optionale Beschraenkung auf einzelne Gletscher
    if [ -n "$GLACIERS" ] && [[ " $GLACIERS " != *" $g "* ]]; then
        continue
    fi

    sdir=$STATIC/$cap
    srf=$sdir/${cap}_combined_SRF_1D20m.nc
    if [ "$LW_TERRAIN" = "off" ]; then
        TERRTAG=""
    else
        TERRTAG="-terr${LW_TERRAIN}"
    fi
    if [ "$PRECIP_LR" = "off" ]; then
        PRTAG=""
    else
        PRTAG="-lrpr"
    fi
    if [ "$LW_METHOD" = "topopyscale" ]; then
        out=$INPUT/$cap/${cap}_ERA5_1D20m_HORAYZON_1987_2024${TERRTAG}${PRTAG}.nc
    else
        out=$INPUT/$cap/${cap}_ERA5_1D20m_HORAYZON_1987_2024_LW-${LW_METHOD}${TERRTAG}${PRTAG}.nc
    fi
    baseline_lut=$sdir/${cap}_${baseline}_HORAYZON-LUT_1D20m.nc

    if [ "$outlines" = "-" ]; then
        outlines_sp=""
    else
        outlines_sp=$(echo "$outlines" | tr ',' ' ')
    fi

    sw=("$baseline_lut")
    for yr in $outlines_sp; do
        sw+=($sdir/${cap}_${yr}_HORAYZON-LUT_1D20m.nc)
    done
    starts=""
    [ -n "$outlines_sp" ] && starts="--sw-starts $outlines_sp"

    # ── Vorpruefung ────────────────────────────────────────────────────────
    ok=1
    [ ! -f "$srf" ] && { echo "!!! $cap: SRF fehlt ($srf)"; ok=0; }
    for f in "${sw[@]}"; do
        [ ! -f "$f" ] && { echo "!!! $cap: LUT fehlt ($f)"; ok=0; }
    done

    # Wolkendateien: bei SW_CLOUD=on zwingend, sonst bricht das Skript ohnehin ab.
    # Hier VOR dem Start pruefen, damit nicht erst nach Minuten Ladezeit klar wird,
    # dass die Dateien fehlen.
    cdir=$PROJ/$g/inputs/climate/yearly
    n_cloud=$(ls "$cdir"/CLOUD_*.nc 2>/dev/null | wc -l)
    if [ "$SW_CLOUD" = "on" ] && [ "$n_cloud" -eq 0 ]; then
        echo "!!! $cap: keine CLOUD_*.nc in $cdir -- SW_CLOUD=on nicht moeglich"
        ok=0
    fi

    # down_pt-Vorpruefung: steckt die Niederschlags-Lapse-Rate wirklich drin?
    # Verhindert, dass ein alter Downscaling-Stand als "-lrpr" gelabelt wird.
    if [ "$PRECIP_LR" != "off" ] && [ $ok -eq 1 ]; then
        chk=$(python - "$PROJ/$g/outputs/downscaled" <<'PYEOF'
import glob, sys
import numpy as np, xarray as xr
files = sorted(glob.glob(sys.argv[1] + "/down_pt*.nc"))
if not files:
    print("NOFILE"); sys.exit()
probe = [files[0], files[len(files)//2], files[-1]]
dev, seen = 0.0, False
for f in probe:
    d = xr.open_dataset(f)
    if "precip_lapse_rate" not in d:
        continue
    seen = True
    v = d["precip_lapse_rate"].values
    dev = max(dev, float(np.nanmax(np.abs(v - 1.0))))
    d.close()
if not seen:
    print("NOVAR")
elif dev < 1e-6:
    print("FLAT")
else:
    print("OK %.2f" % (1.0 + dev))
PYEOF
)
        case "$chk" in
            OK*)    echo "    Lapse-Rate: aktiv, max. Faktor ${chk#OK } (Stichprobe 3 Baender)" ;;
            FLAT)   echo "!!! $cap: precip_lapse_rate == 1 in allen geprueften Baendern --"
                    echo "!!!       das down_pt stammt noch aus dem Lauf OHNE Lapse-Rate."
                    ok=0 ;;
            NOVAR)  echo "    (Hinweis) precip_lapse_rate nicht im down_pt gespeichert -- Pruefung uebersprungen" ;;
            NOFILE) echo "!!! $cap: keine down_pt-Dateien in $PROJ/$g/outputs/downscaled"; ok=0 ;;
            *)      echo "    (Hinweis) down_pt-Pruefung ergab: $chk" ;;
        esac
    fi

    if [ $ok -eq 0 ]; then
        echo "!!! $cap: uebersprungen"; echo ""; n_skip=$((n_skip+1)); continue
    fi

    mkdir -p "$INPUT/$cap"
    echo "=== $cap  $(date) ==="
    echo "    Baseline: ${cap}_${baseline}_HORAYZON-LUT_1D20m.nc"
    echo "    Umrisse:  ${outlines_sp:-(nur Baseline)}"
    echo "    Wolken:   $n_cloud CLOUD-Dateien in $cdir"
    echo "    lat=$lat  tcart=-$lon  LW=$LW_METHOD  Terrain=$LW_TERRAIN (eps=$LW_EPS)  SW-Wolken=$SW_CLOUD"
    echo "    -> $out"
    python "$SCRIPT" --glacier "$g" \
        -o "$out" \
        -s "$srf" \
        --sw "${sw[@]}" $starts \
        -b "$START" -e "$END" \
        --stationLat "$lat" --tcart "-$lon" \
        --lw-method "$LW_METHOD" \
        --lw-terrain "$LW_TERRAIN" \
        --lw-eps "$LW_EPS" \
        --sw-cloud "$SW_CLOUD" \
        --wind-profile "$WIND_PROFILE"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "!!! $cap: Skript endete mit Code $rc"
        n_skip=$((n_skip+1))
    else
        n_ok=$((n_ok+1))
    fi
    echo "=== $cap fertig $(date) ==="
    echo ""
done < "$CONFIG"

echo "ALLE FERTIG $(date):  $n_ok gebaut, $n_skip uebersprungen/fehlerhaft"
