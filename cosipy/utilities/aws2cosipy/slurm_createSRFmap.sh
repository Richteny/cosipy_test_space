#!/bin/bash -l
#SBATCH --job-name="createSRF"
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --chdir=/data/scratch/richteny/thesis/cosipy_test_space/
#SBATCH --account=morsanat
#SBATCH --partition=compute
#SBATCH --output=createSRF-%j.out
#SBATCH --error=createSRF-%j.err
#
# Baut <Cap>_combined_SRF_1D20m.nc (SRF, SRF_50, K_AVA, K_AVA_50, N_Points)
# fuer alle Gletscher aus glacier_config.txt -- dieselbe Datei und dieselben
# Umriss-Jahre wie slurm_toposcale2cosipy.sh, damit die SRF-time-Achse und die
# --sw/--sw-starts-LUTs garantiert in derselben Reihenfolge liegen.
#
#   DRYRUN=1 bash slurm_createSRFmap.sh          -> nur Aufrufe ausgeben
#   sbatch --export=ALL,GLACIERS="mera parlung" slurm_createSRFmap.sh

conda activate downscaling

BASE=/data/scratch/richteny/thesis/cosipy_test_space
SCRIPT=$BASE/cosipy/utilities/aws2cosipy/createSRFmap.py
STATIC=$BASE/data/static
CONFIG=$BASE/cosipy/utilities/aws2cosipy/glacier_config.txt
# k_ava-Tifs in Unterordnern RGI13/ RGI14/ RGI60/ (createSRFmap globbt RGI*/**).
# Beim ersten Lauf wird kneib_index.csv hier angelegt -> Schreibrecht noetig.
KNEIB_DIR="${KNEIB_DIR:-/data/scratch/richteny/Kneib_HMA}"

# Datum der Baseline-Geometrie: muss <= Simulationsstart (1987-10-01) liegen,
# weil toposcale2cosipy die SRF-time-Achse per Forward-Fill auffaellt.
BASELINE_DATE="${BASELINE_DATE:-1980-01-01}"

# Dateinamen je Umriss (<tag> = Baseline-Name RGI6/RGI7 oder Umriss-Jahr).
#   2D: 30-m-Statik mit Puffer, <Cap>_<tag>_static_30m_new.nc
#       Ausnahme Halji-Umrissjahre: Halji2014_static_30m_new.nc (ohne "_")
#   1D: <Cap>_<tag>_1D20m.nc, sonst Rueckfall auf <Cap>_<tag>_HORAYZON-LUT_1D20m.nc
static2d() {
    local f=$STATIC/$1/${1}_${2}_static_30m_new.nc
    [ ! -f "$f" ] && [ -f "$STATIC/$1/${1}${2}_static_30m_new.nc" ] && f=$STATIC/$1/${1}${2}_static_30m_new.nc
    echo "$f"
}
target1d() {
    local f=$STATIC/$1/${1}_${2}_1D20m.nc
    [ ! -f "$f" ] && [ -f "$STATIC/$1/${1}_${2}_HORAYZON-LUT_1D20m.nc" ] && f=$STATIC/$1/${1}_${2}_HORAYZON-LUT_1D20m.nc
    echo "$f"
}

GLACIERS="${GLACIERS:-}"
DRYRUN="${DRYRUN:-0}"

n_ok=0; n_skip=0
while read -r g cap lat lon baseline outlines rest; do
    [[ -z "$g" || "$g" == \#* ]] && continue
    if [ -n "$GLACIERS" ] && [[ " $GLACIERS " != *" $g "* ]]; then continue; fi

    tags=("$baseline");  dates=("$BASELINE_DATE")
    if [ "$outlines" != "-" ]; then
        for yr in $(echo "$outlines" | tr ',' ' '); do
            tags+=("$yr"); dates+=("${yr}-01-01")
        done
    fi

    s2d=(); t1d=(); ok=1
    for tg in "${tags[@]}"; do
        s2d+=("$(static2d "$cap" "$tg")"); t1d+=("$(target1d "$cap" "$tg")")
    done
    if [ "$DRYRUN" != "1" ]; then
        for f in "${s2d[@]}" "${t1d[@]}"; do
            [ ! -f "$f" ] && { echo "!!! $cap: fehlt $f"; ok=0; }
        done
    fi

    out=$STATIC/$cap/${cap}_combined_SRF_1D20m.nc
    out2d=$STATIC/$cap/${cap}_combined_SRF_2D.nc
    kn=(); [ -n "$KNEIB_DIR" ] && kn=(--kneib-dir "$KNEIB_DIR")

    cmd=(python "$SCRIPT" -s "${s2d[@]}" -t "${t1d[@]}" --dates "${dates[@]}"
         --merge-output "$out" -o2 "$out2d" "${kn[@]}")

    echo "=== $cap (${#tags[@]} Umriss/e: ${tags[*]})"
    echo "${cmd[@]}"
    [ "$DRYRUN" = "1" ] && { echo ""; continue; }
    [ $ok -eq 0 ] && { echo "!!! $cap: uebersprungen"; n_skip=$((n_skip+1)); echo ""; continue; }

    "${cmd[@]}" && n_ok=$((n_ok+1)) || { echo "!!! $cap: Fehler"; n_skip=$((n_skip+1)); }
    echo ""
done < "$CONFIG"

echo "FERTIG $(date): $n_ok gebaut, $n_skip uebersprungen/fehlerhaft"
