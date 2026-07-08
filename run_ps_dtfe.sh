#!/usr/bin/env bash
# Phase-Space DTFE counterpart of run_dtfe.sh: tessellates in Lagrangian space, so each
# particle needs Eulerian Coordinates plus a Lagrangian position. HDF5-only (input 105).
# Shared defaults (DATA_ROOT, SIMULATION, SNAPSHOTS, GRID_SIZE, PADDING) live in config.sh.
#
# Usage:
#   ./run_ps_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n AVG_SUBSAMPLES] [-m] [snapshot ...]
#     -m   run the deposit on the Apple GPU (same as PS_METAL=1; needs 'make PS-DTFE METAL=1')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

PARTITION="${PARTITION:-}"        # Lagrangian-partition grid. EMPTY (default) = the binary AUTO-TUNES the split
                           # from the particle count, grid, fields and available RAM/cores (see the AUTO-TUNE
                           # line it prints). Set to override, e.g. PARTITION="5 5 5"; manual reference points:
                           # 5 5 5 fits TNG300-3-Dark (2.4e8 particles) at MAX_CONCURRENT=2 in ~40 GB,
                           # 2 2 2 suits small sims like TNG50-4 (fewer partition overheads = faster).
MAX_CONCURRENT="${MAX_CONCURRENT:-}"  # cap on concurrent partitions. EMPTY (default) = auto-tuned together with the
                           # partition split; set to override (0 = all cores). Peak RAM ~ fixed + cap x
                           # per-triangulation, verify via the "Peak memory (RSS)" line after each run.
AVG_SUBSAMPLES="${AVG_SUBSAMPLES:-3}"   # nSub^3 sub-points for the '_a' fields; dominant runtime cost (~nSub^3), 1 = no averaging. Override: -n 1 or AVG_SUBSAMPLES=1
MPC_UNIT=1000              # length of 1 Mpc in the input's units (1000 for ckpc/h)
THREADS="${THREADS:-}"     # cap OpenMP threads globally; empty = all cores
PS_METAL="${PS_METAL:-0}"  # 1 = run the deposit on the Apple GPU (--ps-metal; needs 'make PS-DTFE METAL=1')

DATA_DIR=""                # default: $DATA_ROOT/$SIMULATION (config.sh); override with -d
OUTPUT_PREFIX="ps_output"   # -> <snapdir>/ps_output_nsubN.* so different nSub runs sit side by side instead of clobbering

usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n AVG_SUBSAMPLES] [-m] [snapshot ...]"; }

while getopts "d:s:g:n:mh" opt; do
    case "$opt" in
        d) DATA_DIR="$OPTARG" ;;
        s) SIMULATION="$OPTARG" ;;
        g) GRID_SIZE="$OPTARG" ;;
        n) AVG_SUBSAMPLES="$OPTARG" ;;
        m) PS_METAL=1 ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

# Any remaining positional arguments override the default snapshot list.
if [ "$#" -gt 0 ]; then
    SNAPSHOTS=("$@")
fi

[ -z "${DATA_DIR}" ] && DATA_DIR="${DATA_ROOT}/${SIMULATION}"

# Lagrangian positions, matched to present-day Coordinates by ParticleID. Must be in the
# SAME units as the snapshots (combined_*.hdf5 are h-removed ckpc, so the IC was rescaled
# by 1/h via tools/convert_ic_units). Do NOT reconstruct the grid from ParticleID: TNG IDs
# are Peano-Hilbert ordered, so id->(ix,iy,iz) is wrong.
LAGRANGIAN_INPUT="${DATA_DIR}/combined_ics.hdf5"

# Phase-space fields; '.streams' is always written, each field also gets a '_a' (averaged) form.
# Both forms are mass-conserving (each tetrahedron deposits its full mass onto the grid). The '_a'
# fields resolve each grid cell with an nSub^3 sub-sample grid (AVG_SUBSAMPLES, default 3), so they
# are smoother / better resolved at caustics -- prefer them for science plots. The plain fields use a
# single sample per cell (coarser, but the same conserved quantity).
# divergence/shear/vorticity are rigorous only where streams==1; for multi-stream kinematics
# use 'dispersion_a' + the stream count. vweb classifies the cosmic web from the velocity
# gradient (density-weighted across streams), so it inherits the same single-stream caveat.
# tweb = T-web from the TIDAL tensor (FFT Poisson solve of the RAW density grid, in C++); vweb =
# V-web from the multi-stream velocity shear. LAMBDA_TH is the eigenvalue threshold for BOTH webs
# (dimensionless; literature ~0.2-0.4 -- NOTE: runs before 2026-07-03 used the old default 0.0).
# The binary applies NO smoothing anywhere; smoothing is a plot-time choice in plot_PS_DTFE.py.
LAMBDA_TH="${LAMBDA_TH:-0.3}"
FIELDS="density_a velocity_a dispersion_a"

cd "$SCRIPT_DIR" || exit 1

if [ ! -x "./PS-DTFE" ]; then
    echo "Error: ./PS-DTFE not found or not executable. Build it with 'make PS-DTFE'." >&2
    exit 1
fi
[ -n "$THREADS" ] && export OMP_NUM_THREADS="$THREADS"

# Keep the Mac awake for the whole batch; caffeinate follows this PID and lifts on exit.
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -m -w $$ &
fi

echo "Starting PS-DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}   Partition: [${PARTITION:-auto}]   Fields: ${FIELDS}"
echo ""

for i in "${SNAPSHOTS[@]}"; do
    n_str=$(printf "%03d" "$i")

    input_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
    input_file="${input_dir}/combined_${n_str}.hdf5"
    output_root="${input_dir}/${OUTPUT_PREFIX}"

    echo "Processing snapshot ${n_str}..."

    if [ ! -f "${input_file}" ]; then
        echo "  Warning: Input file not found: ${input_file}"
        echo "  Skipping snapshot ${n_str}"
        continue
    fi

    # GPU deposit toggle (PS_METAL=1 ./run_ps_dtfe.sh); ignored with a warning on non-METAL builds.
    metal_args=()
    [ "${PS_METAL}" = "1" ] && metal_args=(--ps-metal)

    # Separate Lagrangian input, unless InitialCoordinates is in the snapshot (empty here).
    lag_args=()
    if [ -n "${LAGRANGIAN_INPUT}" ]; then
        if [ ! -f "${LAGRANGIAN_INPUT}" ]; then
            echo "  Warning: Lagrangian input not found: ${LAGRANGIAN_INPUT}"
            echo "  Skipping snapshot ${n_str}"
            continue
        fi
        lag_args=(--lagrangianInput "${LAGRANGIAN_INPUT}")
    fi

    # /usr/bin/time -l captures peak memory (verifies MAX_CONCURRENT fits RAM) and wall time.
    # The tee pipe hides the terminal from the binary's isatty() check, so force colours through
    # it (CLICOLOR_FORCE, honoured by message.h) -- but only when this script itself runs on a
    # terminal, keeping cron/CI output clean. The runlog is de-ANSI'd after the run.
    [ -t 1 ] && export CLICOLOR_FORCE=1
    echo "  Running PS-DTFE on ${input_file}..."
    run_log="${input_dir}/${OUTPUT_PREFIX}.runlog"
    SECONDS=0
    # pass --partition/--max-concurrent only when set; otherwise the binary auto-tunes them
    part_args=()
    [ -n "${PARTITION}" ] && part_args+=(--partition ${PARTITION})
    [ -n "${MAX_CONCURRENT}" ] && part_args+=(--max-concurrent "${MAX_CONCURRENT}")

    /usr/bin/time -l ./PS-DTFE "${input_file}" "${output_root}" \
        --grid ${GRID_SIZE} \
        --padding ${PADDING} \
        --periodic \
        ${part_args[@]+"${part_args[@]}"} \
        --avg-subsamples ${AVG_SUBSAMPLES} \
        --input 105 \
        --MpcUnit ${MPC_UNIT} \
        --field ${FIELDS} \
        --lambda_th ${LAMBDA_TH} \
        "${metal_args[@]}" \
        "${lag_args[@]}" 2>&1 | tee "${run_log}"
    rc=${PIPESTATUS[0]}

    # strip ANSI colour codes from the saved log so it stays grep-able
    sed -E -i '' $'s/\033\\[[0-9;]*m//g' "${run_log}"

    # macOS /usr/bin/time -l prints "<bytes>  maximum resident set size".
    peak_bytes=$(awk '/maximum resident set size/{print $1; exit}' "${run_log}")
    if [ -n "${peak_bytes}" ]; then
        peak_gb=$(awk -v b="${peak_bytes}" 'BEGIN{printf "%.1f", b/1073741824}')
        echo "  Peak memory (RSS): ${peak_gb} GB   (target < ~56 GB on 64 GB; if higher, set MAX_CONCURRENT=1 to override the auto choice)"
    fi
    echo "  Wall time: $((SECONDS/60)) min $((SECONDS%60)) s"

    if [ "${rc}" -eq 0 ]; then
        echo "  Snapshot ${n_str} processed successfully"
    else
        echo "  Error processing snapshot ${n_str}"
    fi
    echo ""
done

echo "PS-DTFE processing complete!"
