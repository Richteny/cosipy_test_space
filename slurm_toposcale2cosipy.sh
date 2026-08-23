#!/bin/bash -l
#SBATCH --job-name="tps2cosipy"
# Tipp: sbatch --export=ALL,LW_METHOD=liu-cf,LW_TERRAIN=airT slurm_toposcale2cosipy.sh
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --partition=compute
#SBATCH --output=tps2cosipy.out
#SBATCH --error=tps2cosipy.err

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

# Nur die Testgletscher bauen: GLACIERS="mera parlung"
GLACIERS="${GLACIERS:-}"

# Ausgabenamen bleiben unveraendert -- alte Forcings werden UEBERSCHRIEBEN.
# (Bewusste Entscheidung: die alte Variante war fehlerhaft und wird nicht
# aufbewahrt. Fuer einen Vorher-Nachher-Vergleich vorher wegkopieren.)

if [ ! -f "$CONFIG" ]; then
    echo "FEHLER: Konfigurationsdatei fehlt: $CONFIG"; exit 1
fi

echo "############################################################"
echo "  LW_METHOD = $LW_METHOD"
echo "  LW_TERRAIN= $LW_TERRAIN   (eps=$LW_EPS)"
echo "  SW_CLOUD  = $SW_CLOUD"
echo "  ACHTUNG: bestehende Forcings werden ueberschrieben"
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
    if [ "$LW_METHOD" = "topopyscale" ]; then
        out=$INPUT/$cap/${cap}_ERA5_1D20m_HORAYZON_1987_2024${TERRTAG}.nc
    else
        out=$INPUT/$cap/${cap}_ERA5_1D20m_HORAYZON_1987_2024_LW-${LW_METHOD}${TERRTAG}.nc
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
        --sw-cloud "$SW_CLOUD"
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
