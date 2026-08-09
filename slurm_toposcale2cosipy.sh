#!/bin/bash -l
#SBATCH --job-name="tps2cosipy"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --partition=computehm
#SBATCH --output=tps2cosipy.out
#SBATCH --error=tps2cosipy.err

conda activate downscaling
export GIT_PYTHON_REFRESH=quiet

BASE=/data/scratch/richteny/thesis/cosipy_test_space
SCRIPT=$BASE/cosipy/utilities/aws2cosipy/toposcale2cosipy.py
STATIC=$BASE/data/static
INPUT=$BASE/data/input
CONFIG=$BASE/cosipy/utilities/aws2cosipy/glacier_config.txt
START="1987-10-01"
END="2024-12-31"

if [ ! -f "$CONFIG" ]; then
    echo "FEHLER: Konfigurationsdatei fehlt: $CONFIG"; exit 1
fi

# Tabelle zeilenweise lesen. Leerzeichen trennen die Spalten automatisch.
while read -r g cap lat lon baseline outlines rest; do
    # Kommentar- oder Leerzeilen ueberspringen
    [[ -z "$g" || "$g" == \#* ]] && continue

    sdir=$STATIC/$cap
    srf=$sdir/${cap}_combined_SRF_1D20m.nc
    out=$INPUT/$cap/${cap}_ERA5_1D20m_HORAYZON_1987_2024.nc
    baseline_lut=$sdir/${cap}_${baseline}_HORAYZON-LUT_1D20m.nc

    # outlines: Komma -> Leerzeichen; "-" bedeutet keine datierten Umrisse
    if [ "$outlines" = "-" ]; then
        outlines_sp=""
    else
        outlines_sp=$(echo "$outlines" | tr ',' ' ')
    fi

    # SW-Liste: Baseline zuerst, dann datierte Umrisse
    sw=("$baseline_lut")
    for yr in $outlines_sp; do
        sw+=($sdir/${cap}_${yr}_HORAYZON-LUT_1D20m.nc)
    done
    starts=""
    [ -n "$outlines_sp" ] && starts="--sw-starts $outlines_sp"

    # Vorpruefung: alle Dateien vorhanden?
    ok=1
    [ ! -f "$srf" ] && { echo "!!! $cap: SRF fehlt ($srf)"; ok=0; }
    for f in "${sw[@]}"; do
        [ ! -f "$f" ] && { echo "!!! $cap: LUT fehlt ($f)"; ok=0; }
    done
    if [ $ok -eq 0 ]; then echo "!!! $cap: uebersprungen"; echo ""; continue; fi

    mkdir -p "$INPUT/$cap"
    echo "=== $cap  $(date) ==="
    echo "    Baseline: ${cap}_${baseline}_HORAYZON-LUT_1D20m.nc"
    echo "    Umrisse:  ${outlines_sp:-(nur Baseline)}"
    echo "    lat=$lat  tcart=-$lon"
    python "$SCRIPT" --glacier "$g" \
        -o "$out" \
        -s "$srf" \
        --sw "${sw[@]}" $starts \
        -b "$START" -e "$END" \
        --stationLat "$lat" --tcart "-$lon"
    echo "=== $cap fertig $(date) ==="
    echo ""
done < "$CONFIG"

echo "ALLE FERTIG $(date)"
