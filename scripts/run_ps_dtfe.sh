#!/usr/bin/env bash
# Phase-Space DTFE counterpart of run_dtfe.sh: tessellates in Lagrangian space, so each
# particle needs Eulerian Coordinates plus a Lagrangian position. HDF5-only (input 105).
# Shared defaults (DATA_ROOT, SIMULATION, SNAPSHOTS, GRID_SIZE, PADDING) live in config.sh.
#
# Usage:
#   ./run_ps_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n AVG_SUBSAMPLES] [-m] [-e] [snapshot ...]
#     -m   run the deposit on the Apple GPU (same as PS_METAL=1; needs 'make PS-DTFE METAL=1')
#     -e   exact conservative deposit (same as PS_EXACT=1; --ps-exact-deposit, CPU-only, slower)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"   # the binaries and Makefile live one level up
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
SCRATCH_DIR="${SCRATCH_DIR:-}"  # out-of-core mode (--scratch-dir): back the full-resolution grid
                           # accumulators (>= 1 GB allocations) with mmap'ed files in this LOCAL
                           # directory instead of RAM. Required for GRID_SIZE=1024 with the full
                           # FIELDS list (~146 GB of accumulators vs 55 GB budget); bit-identical
                           # results, RSS stays bounded, scratch files self-delete on any exit.
                           # MUST be a local non-synced path, e.g. /private/tmp/dtfe-scratch
                           # (mkdir it first) -- iCloud paths are rejected by the binary.
SAMPLE_POINTS="${SAMPLE_POINTS:-}"  # path to a --sample-points file (e.g. from
                           # python/tools/make_image_plane.py): evaluate the continuous field at
                           # those points IN ADDITION to the grid deposit, writing
                           # <snapdir>/<prefix>.pts_* (CPU, double precision). Shares the run's
                           # triangulation, so the marginal cost is just the evaluation --
                           # this is how the high-resolution figure slices piggyback on a
                           # production grid run (see run_ps_pipeline.sh).
PTS_VEL_GRAD="${PTS_VEL_GRAD:-0}"   # 1 = with SAMPLE_POINTS, also write '.pts_velGrad'
                           # (--pts-vel-grad): the density-weighted velocity gradient at each
                           # sample point (float64 x9). dtfelib.PointPlane derives the
                           # divergence / shear / vorticity maps from it, so this is what makes
                           # the velocity-derivative fields available to the hi-res figures.
PS_METAL="${PS_METAL:-0}"  # 1 = run the deposit on the Apple GPU (--ps-gpu; needs 'make PS-DTFE METAL=1')
PS_VERTEX_MASS="${PS_VERTEX_MASS:-1}"  # 1 (default) = chart-independent tet masses (--ps-vertex-mass):
                           # each particle's mass splits equally among its incident tetrahedra. REQUIRED for
                           # TNG runs: combined_ics.hdf5 holds the z=127 IC positions, whose configuration
                           # already carries delta_ic = D(127)/D(z)*delta -- the default rho_bar*V_lag masses
                           # then filter every density mode by 1-D(127)/D(z) (-16.5% at z=20, -3% at z=2;
                           # verified against regular DTFE on TNG50-3). Velocities are unaffected either way.
                           # Set PS_VERTEX_MASS=0 only for true-lattice Lagrangian inputs or A/B comparisons.
PS_VOLUME_WEIGHTED="${PS_VOLUME_WEIGHTED:-0}"  # 1 = volume-weighted velocity moments (--ps-volume-weighted):
                           # velocity/gradient/div/shear/vort/dispersion become VOLUME averages per cell
                           # (the standard-DTFE '_a' convention that -aHf*delta refers to) instead of the
                           # default mass-weighted (momentum-like) means. The DISPERSION is excluded and stays
                           # mass-weighted (sigma_ij is an f-weighted CBE moment by definition) -- it comes out
                           # bit-identical to a default run, so this ONE config is the literature-standard
                           # estimator for every field at once. Works with the CPU and GPU (-m) deposits.
PS_EXACT="${PS_EXACT:-0}"  # 1 = exact conservative deposit (--ps-exact-deposit): analytic r3d tet-cell
                           # clipping instead of the nSub^3 sub-sampled deposit -- no sampling noise, mass
                           # conservation to the arithmetic's precision. Runs on the CPU (double, the
                           # reference) and on the GPU with -m (float32 r3d port) -- USE -m HERE, the
                           # clipping is the most expensive deposit by far. Still slower than the sampled
                           # deposit: an accuracy option. AVG_SUBSAMPLES is ignored (the exact deposit is
                           # the nSub->infinity limit, so '.den' == '.a_den'). Override: -e

DATA_DIR=""                # default: $DATA_ROOT/$SIMULATION (config.sh); override with -d
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ps_output}"   # -> <snapdir>/<prefix>.*  Override to keep incompatible runs side by
                           # side instead of clobbering, e.g. OUTPUT_PREFIX=ps_mw with PS_VOLUME_WEIGHTED=0 for the
                           # mass-weighted (physical) dispersion next to the default volume-weighted shear/divergence
                           # set. dtfelib.FieldSet reads the 'ps_output.' prefix, so keep the primary set there.

usage() { echo "Usage: $0 [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n AVG_SUBSAMPLES] [-m] [-e] [snapshot ...]"; }

while getopts "d:s:g:n:meh" opt; do
    case "$opt" in
        d) DATA_DIR="$OPTARG" ;;
        s) SIMULATION="$OPTARG" ;;
        g) GRID_SIZE="$OPTARG" ;;
        n) AVG_SUBSAMPLES="$OPTARG" ;;
        m) PS_METAL=1 ;;
        e) PS_EXACT=1 ;;
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
# by 1/h via python/tools/convert_ic_units.py). Do NOT reconstruct the grid from ParticleID: TNG IDs
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
# Same velocity-derivative set as run_dtfe.sh (gradient_a divergence_a shear_a vorticity_a, all
# derived from the density-weighted multi-stream velocity gradient -- single-stream caveat above)
# plus the PS-only dispersion; add tweb_a/vweb_a here to also classify the cosmic web.
# Env-overridable, e.g. FIELDS="density_a velocity_a" for a lighter batch.
FIELDS="${FIELDS:-density_a velocity_a gradient_a divergence_a shear_a vorticity_a dispersion_a}"

cd "$REPO_ROOT" || exit 1

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
    [ "${PS_METAL}" = "1" ] && metal_args=(--ps-gpu)

    # Exact conservative deposit toggle (-e / PS_EXACT=1); CPU and GPU deposits.
    exact_args=()
    [ "${PS_EXACT}" = "1" ] && exact_args=(--ps-exact-deposit)

    # Chart-independent tet masses (PS_VERTEX_MASS=1, default -- see the comment at the top).
    vmass_args=()
    [ "${PS_VERTEX_MASS}" = "1" ] && vmass_args=(--ps-vertex-mass)

    # Volume-weighted velocity moments (PS_VOLUME_WEIGHTED=1); CPU and GPU deposits.
    vw_args=()
    [ "${PS_VOLUME_WEIGHTED}" = "1" ] && vw_args=(--ps-volume-weighted)

    # Point evaluation on top of the grid run (SAMPLE_POINTS=<file>, see the comment at the top).
    sp_args=()
    [ -n "${SAMPLE_POINTS}" ] && sp_args=(--sample-points "${SAMPLE_POINTS}")
    [ -n "${SAMPLE_POINTS}" ] && [ "${PTS_VEL_GRAD}" = "1" ] && sp_args+=(--pts-vel-grad)

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
    [ -n "${SCRATCH_DIR}" ] && part_args+=(--scratch-dir "${SCRATCH_DIR}")

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
        ${metal_args[@]+"${metal_args[@]}"} \
        ${exact_args[@]+"${exact_args[@]}"} \
        ${vmass_args[@]+"${vmass_args[@]}"} \
        ${vw_args[@]+"${vw_args[@]}"} \
        ${sp_args[@]+"${sp_args[@]}"} \
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
