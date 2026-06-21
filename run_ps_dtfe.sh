#!/usr/bin/env bash
# Phase-Space DTFE counterpart of run_dtfe.sh: tessellates in Lagrangian space, so each
# particle needs Eulerian Coordinates plus a Lagrangian position. HDF5-only (input 105).

SNAPSHOTS=(99)

GRID_SIZE=512
PADDING=25
PARTITION="2 2 2"          # Lagrangian-partition grid; coarser = fewer seams but larger per-partition triangulation
MAX_CONCURRENT=1           # cap on concurrent triangulations (0 = all cores); peak RAM ~ fixed + cap x per-triangulation, verify via "Peak memory (RSS)" line
AVG_SUBSAMPLES=3   # nSub^3 sub-points for the '_a' fields; dominant runtime cost (~nSub^3), 1 = no averaging. Override per run: AVG_SUBSAMPLES=1 ./run_ps_dtfe.sh
MPC_UNIT=1000              # length of 1 Mpc in the input's units (1000 for ckpc/h)
THREADS=""                 # cap OpenMP threads globally; empty = all cores

DATA_DIR="/Users/luukw/output/TNG50-4-Dark"
INPUT_SUBDIR="snapdir"
OUTPUT_PREFIX="ps_output"   # -> <snapdir>/ps_output_nsubN.* so different nSub runs sit side by side instead of clobbering

# Lagrangian positions, matched to present-day Coordinates by ParticleID. Must be in the
# SAME units as the snapshots (combined_*.hdf5 are h-removed ckpc, so the IC was rescaled
# by 1/h via tools/convert_ic_units). Do NOT reconstruct the grid from ParticleID: TNG IDs
# are Peano-Hilbert ordered, so id->(ix,iy,iz) is wrong.
LAGRANGIAN_INPUT="${DATA_DIR}/combined_ics.hdf5"

# Phase-space fields; '.streams' is always written, each field also gets a '_a' (averaged) form.
# divergence/shear/vorticity are rigorous only where streams==1; for multi-stream kinematics
# use 'dispersion' + the stream count. tweb/vweb classify the cosmic web from the velocity
# gradient (density-weighted across streams), so they inherit the same single-stream caveat.
FIELDS="density velocity dispersion tweb vweb density_a velocity_a dispersion_a tweb_a vweb_a"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
echo "Grid size: ${GRID_SIZE}   Partition: [${PARTITION}]   Fields: ${FIELDS}"
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
    echo "  Running PS-DTFE on ${input_file}..."
    run_log="${input_dir}/${OUTPUT_PREFIX}.runlog"
    SECONDS=0
    /usr/bin/time -l ./PS-DTFE "${input_file}" "${output_root}" \
        --grid ${GRID_SIZE} \
        --padding ${PADDING} \
        --periodic \
        --partition ${PARTITION} \
        --max-concurrent ${MAX_CONCURRENT} \
        --avg-subsamples ${AVG_SUBSAMPLES} \
        --input 105 \
        --MpcUnit ${MPC_UNIT} \
        --field ${FIELDS} \
        "${lag_args[@]}" 2>&1 | tee "${run_log}"
    rc=${PIPESTATUS[0]}

    # macOS /usr/bin/time -l prints "<bytes>  maximum resident set size".
    peak_bytes=$(awk '/maximum resident set size/{print $1; exit}' "${run_log}")
    if [ -n "${peak_bytes}" ]; then
        peak_gb=$(awk -v b="${peak_bytes}" 'BEGIN{printf "%.1f", b/1073741824}')
        echo "  Peak memory (RSS): ${peak_gb} GB   (target < ~56 GB on 64 GB; if higher, lower MAX_CONCURRENT)"
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
