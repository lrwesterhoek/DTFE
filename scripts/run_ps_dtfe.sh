#!/usr/bin/env bash
# Phase-Space DTFE counterpart of run_dtfe.sh: tessellates in Lagrangian space, so each
# particle needs Eulerian Coordinates plus a Lagrangian position. HDF5-only (input 105).
# Shared defaults (DATA_ROOT, SIMULATION, SNAPSHOTS, GRID_SIZE, PADDING) live in config.sh.
#
# Usage:
#   ./run_ps_dtfe.sh [-d DATA_DIR] [-s SIMULATION] [-g GRID_SIZE] [-n AVG_SUBSAMPLES] [-m] [-e] [snapshot ...]
#     -m   run the deposit on the Apple GPU (same as PS_METAL=1; needs 'make PS-DTFE METAL=1')
#     -e   exact conservative deposit (same as PS_EXACT=1; --ps-exact-deposit, GPU-capable, slower)

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
AUTO_TUNE_REPORT="${AUTO_TUNE_REPORT:-0}"  # 1 = only ASK the binary's auto-tuner what this run would need
                           # (--auto-tune-report): per snapshot one line
                           #   AUTO-TUNE-REPORT snap=NNN partition=.. mc=.. predicted_gb=.. budget_gb=.. over_budget=0|1 ...
                           # after reading the input, and nothing is computed or written (the
                           # snapshot's .runlog is left alone). The GUI's memory check uses it;
                           # DTFE_AUTO_PTS_N / DTFE_AUTO_PTS_VELGRAD=1 size a plane not generated yet.
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
PS_CAUSTICS="${PS_CAUSTICS:-0}"      # 1 = flag fold-caustic cells (--ps-caustics): writes '.caustic'
                           # (0/1 fold flag) and, on the CPU deposit, '.causticClass' -- the caustic
                           # stratification bitmask (fold parity, how many principal axes have
                           # collapsed, and the umbilic-degeneracy indicator).
PS_CAUSTIC_CUSPS="${PS_CAUSTIC_CUSPS:-0}"  # 1 = also estimate the A3 CUSP (bit 7) and A4
                           # SWALLOWTAIL (bit 8) indicators of '.causticClass'; requires
                           # PS_CAUSTICS=1, CPU deposit. Opt-in: both are finite differences over
                           # the deduplicated 64-nearest ring of tetrahedra, ~+22% on the caustic
                           # pass. Both RANK candidates rather than certify them.
TESS_CACHE="${TESS_CACHE:-}"  # directory for the reusable on-disk tessellation
                           # (--tessellation-cache). Re-running the same snapshot with a different
                           # grid, field set or point set then skips the triangulation AND the
                           # vertex-density pass (measured 10.9 s -> 4.6 s at 884736 particles).
                           # Must be a LOCAL, non-synced dir; costs ~0.15 KB per tessellation
                           # vertex (the file is deflated: 225 MB per 1.5e6 vertices).
PS_EXACT="${PS_EXACT:-0}"  # 1 = exact conservative deposit (--ps-exact-deposit): analytic r3d tet-cell
                           # clipping instead of the nSub^3 sub-sampled deposit -- no sampling noise, mass
                           # conservation to the arithmetic's precision. Runs on the CPU (double, the
                           # reference) and on the GPU with -m (float32 r3d port) -- USE -m HERE, the
                           # clipping is the most expensive deposit by far. Still slower than the sampled
                           # deposit: an accuracy option. AVG_SUBSAMPLES is ignored (the exact deposit is
                           # the nSub->infinity limit, so '.den' == '.a_den'). Override: -e

DATA_DIR=""                # default: sim_dir $SIMULATION (config.sh); override with -d
PS_PARALLEL_TRI="${PS_PARALLEL_TRI:-0}"  # 1 = build each tessellation's Delaunay triangulation in parallel
                           # (--parallel-triangulation; needs the TBB build, 'make' detects the library).
                           # Faster on small sets; OFF by default because two identical runs then
                           # differ at float rounding (stream counts do not). See the binary's help.
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
# Snapshot numbers as decimal: bash's printf reads a leading zero as OCTAL, so '010' became 008 and '099'
# an error (then 000) -- the folders are named snapdir_0NN, so typing them that way is natural (2026-10-05).
for _k in "${!SNAPSHOTS[@]}"; do
    _s="${SNAPSHOTS[$_k]}"
    if ! [[ "${_s}" =~ ^[0-9]+$ ]]; then
        echo "Error: snapshot '${_s}' is not a number." >&2
        exit 2
    fi
    SNAPSHOTS[$_k]=$((10#${_s}))
done

[ -z "${DATA_DIR}" ] && DATA_DIR="$(sim_dir "${SIMULATION}")"

# Lagrangian positions, matched to present-day Coordinates by ParticleID. Must be in the
# SAME units as the snapshots (combined_*.hdf5 are h-removed ckpc, so the IC was rescaled
# by 1/h via python/tools/convert_ic_units.py). Do NOT reconstruct the grid from ParticleID: TNG IDs
# are Peano-Hilbert ordered, so id->(ix,iy,iz) is wrong.
#
# Used when present but NOT required: PS-DTFE reads Lagrangian positions from the snapshot's own
# 'InitialCoordinates' when '--lagrangianInput' is omitted. ${VAR-default} not ${VAR:-default}, so
# an explicitly empty LAGRANGIAN_INPUT= forces the snapshot path even if an IC file sits beside it.
LAGRANGIAN_INPUT="${LAGRANGIAN_INPUT-${DATA_DIR}/combined_ics.hdf5}"

# python with a WORKING h5py, used only to look for the 'InitialCoordinates' dataset below (plain
# python3 resolves to a broken x86_64 h5py on the dev machine, same override the test scripts use)
PY_H5="${PYTHON:-python3}"
command -v /opt/homebrew/bin/python3.14 >/dev/null 2>&1 && PY_H5=/opt/homebrew/bin/python3.14

# Echoes yes/no/unknown for whether $1 carries its own Lagrangian positions. On 'unknown' (no usable
# h5py) we neither guess nor skip: the run is attempted and the binary decides, since it errors
# precisely when the dataset is absent. Guessing "no" would silently drop usable snapshots.
snapshot_has_initial_coords() {
    "${PY_H5}" - "$1" 2>/dev/null <<'PYEOF' || echo unknown
import sys
try:
    import h5py
except ImportError:
    sys.exit(1)
with h5py.File(sys.argv[1], "r") as f:
    for key in f:
        if key.startswith("PartType") and "InitialCoordinates" in f[key]:
            print("yes")
            sys.exit(0)
    print("no")
PYEOF
}

# Phase-space fields; '.streams' and '.unresolved' are always written, each field also gets a '_a'
# (averaged) form. Single-stream cells are streams==1 AND unresolved==0 (see demo/COMMANDS.md).
# Both forms are mass-conserving (each tetrahedron deposits its full mass onto the grid). The '_a'
# fields resolve each grid cell with an nSub^3 sub-sample grid (AVG_SUBSAMPLES, default 3), so they
# are smoother / better resolved at caustics -- prefer them for science plots. The plain fields use a
# single sample per cell (coarser, but the same conserved quantity).
# divergence/shear/vorticity are rigorous only in single-stream cells; for multi-stream kinematics
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

# DTFE_PRECISION=double runs the double-precision pair (./PS-DTFE-double, 'make PS-DTFE DOUBLE=1'):
# every position, density and field in double from the input read onward, float64 outputs, about
# twice the memory. Default single (./PS-DTFE). Nothing falls back silently: a missing binary stops.
DTFE_PRECISION="${DTFE_PRECISION:-single}"
case "${DTFE_PRECISION}" in
    single) PS_BIN="./PS-DTFE" ;;
    double) PS_BIN="./PS-DTFE-double" ;;
    *) echo "Error: DTFE_PRECISION must be 'single' or 'double' (got '${DTFE_PRECISION}')." >&2; exit 1 ;;
esac
if [ ! -x "${PS_BIN}" ]; then
    if [ "${DTFE_PRECISION}" = "double" ]; then
        echo "Error: ${PS_BIN} not found. Build the double-precision pair with 'make DTFE PS-DTFE DOUBLE=1' (add METAL=1 on Apple Silicon), or let the launcher's Setup do it." >&2
    else
        echo "Error: ${PS_BIN} not found or not executable. Build it with 'make PS-DTFE'." >&2
    fi
    exit 1
fi

n_ok=0; n_skipped=0; n_failed=0   # summarised at the end; see the exit-status note there

# Peak-RSS wrapper, resolved once and OPTIONAL: macOS spells the flag -l, GNU -v, and a bare Ubuntu
# container has no /usr/bin/time at all. Without one the run is identical, just with no RSS line.
TIME_WRAP=()
if [ -x /usr/bin/time ]; then
    if /usr/bin/time -l true >/dev/null 2>&1; then
        TIME_WRAP=(/usr/bin/time -l)          # BSD/macOS: "<bytes>  maximum resident set size"
    elif /usr/bin/time -v true >/dev/null 2>&1; then
        TIME_WRAP=(/usr/bin/time -v)          # GNU: "Maximum resident set size (kbytes): <n>"
    fi
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
        n_skipped=$((n_skipped+1))
        continue
    fi

    # GPU deposit toggle (PS_METAL=1 ./run_ps_dtfe.sh); ignored with a warning on non-METAL builds.
    metal_args=()
    [ "${PS_METAL}" = "1" ] && metal_args=(--ps-gpu)

    # Caustic flagging (PS_CAUSTICS=1) and the opt-in A3 cusp indicator (PS_CAUSTIC_CUSPS=1).
    caustic_args=()
    [ "${PS_CAUSTICS}" = "1" ] && caustic_args=(--ps-caustics)
    [ "${PS_CAUSTICS}" = "1" ] && [ "${PS_CAUSTIC_CUSPS}" = "1" ] && caustic_args+=(--ps-caustic-cusps)

    # Reusable on-disk tessellation (TESS_CACHE=<dir>); created if missing, since the binary
    # requires an EXISTING directory and refuses iCloud paths.
    tess_args=()
    if [ -n "${TESS_CACHE}" ]; then
        mkdir -p "${TESS_CACHE}"
        tess_args=(--tessellation-cache "${TESS_CACHE}")
    fi

    # Exact conservative deposit toggle (-e / PS_EXACT=1); CPU and GPU deposits.
    exact_args=()
    [ "${PS_EXACT}" = "1" ] && exact_args=(--ps-exact-deposit)

    # Chart-independent tet masses (PS_VERTEX_MASS=1, default -- see the comment at the top).
    vmass_args=()
    [ "${PS_VERTEX_MASS}" = "1" ] && vmass_args=(--ps-vertex-mass)

    # Volume-weighted velocity moments (PS_VOLUME_WEIGHTED=1); CPU and GPU deposits.
    vw_args=()
    [ "${PS_VOLUME_WEIGHTED}" = "1" ] && vw_args=(--ps-volume-weighted)

    # Parallel Delaunay insertion (PS_PARALLEL_TRI=1, opt-in -- see the comment at the top).
    ptri_args=()
    [ "${PS_PARALLEL_TRI}" = "1" ] && ptri_args=(--parallel-triangulation)

    # Point evaluation on top of the grid run (SAMPLE_POINTS=<file>, see the comment at the top).
    sp_args=()
    [ -n "${SAMPLE_POINTS}" ] && sp_args=(--sample-points "${SAMPLE_POINTS}")
    [ -n "${SAMPLE_POINTS}" ] && [ "${PTS_VEL_GRAD}" = "1" ] && sp_args+=(--pts-vel-grad)

    # Lagrangian positions: the separate IC file when it exists, else the snapshot's own
    # 'InitialCoordinates'. Skip only when NEITHER is available.
    lag_args=()
    if [ -n "${LAGRANGIAN_INPUT}" ] && [ -f "${LAGRANGIAN_INPUT}" ]; then
        lag_args=(--lagrangianInput "${LAGRANGIAN_INPUT}")
    else
        case "$(snapshot_has_initial_coords "${input_file}")" in
            yes)
                echo "  Lagrangian positions: the snapshot's own 'InitialCoordinates'"
                [ -n "${LAGRANGIAN_INPUT}" ] && echo "    (no ${LAGRANGIAN_INPUT##*/} needed)"
                ;;
            no)
                echo "  Warning: snapshot ${n_str} has NO Lagrangian positions:"
                [ -n "${LAGRANGIAN_INPUT}" ] && \
                    echo "           '${LAGRANGIAN_INPUT}' does not exist, and"
                echo "           the snapshot carries no 'InitialCoordinates' dataset either."
                echo "           Convert an IC snapshot with python/tools/convert_ic_units.py (it must"
                echo "           be in the snapshot's units: h-removed ckpc for combined_*.hdf5)."
                echo "  Skipping snapshot ${n_str}"
                n_skipped=$((n_skipped+1))
                continue
                ;;
            *)
                echo "  Note: cannot check for 'InitialCoordinates' (no usable h5py); running without"
                echo "        --lagrangianInput. PS-DTFE reports precisely if the dataset is missing."
                ;;
        esac
    fi

    # TIME_WRAP (resolved above) captures peak memory, verifying MAX_CONCURRENT fits RAM.
    # The tee pipe hides the terminal from the binary's isatty() check, so force colours through
    # it (CLICOLOR_FORCE, honoured by message.h) -- but only when this script itself runs on a
    # terminal, keeping cron/CI output clean. The runlog is de-ANSI'd after the run.
    [ -t 1 ] && export CLICOLOR_FORCE=1
    echo "  Running PS-DTFE on ${input_file}..."
    run_log="${input_dir}/${OUTPUT_PREFIX}.runlog"
    # a report writes nothing next to the data: not even over the snapshot's real run log
    [ "${AUTO_TUNE_REPORT}" = "1" ] && run_log="$(mktemp -t dtfe-auto-tune-report.XXXXXX)"
    SECONDS=0
    # pass --partition/--max-concurrent only when set; otherwise the binary auto-tunes them
    part_args=()
    report_args=()
    [ "${AUTO_TUNE_REPORT}" = "1" ] && report_args=(--auto-tune-report)
    [ -n "${PARTITION}" ] && part_args+=(--partition ${PARTITION})
    [ -n "${MAX_CONCURRENT}" ] && part_args+=(--max-concurrent "${MAX_CONCURRENT}")
    [ -n "${SCRATCH_DIR}" ] && part_args+=(--scratch-dir "${SCRATCH_DIR}")

    ${TIME_WRAP[@]+"${TIME_WRAP[@]}"} "${PS_BIN}" "${input_file}" "${output_root}" \
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
        ${ptri_args[@]+"${ptri_args[@]}"} \
        ${vw_args[@]+"${vw_args[@]}"} \
        ${sp_args[@]+"${sp_args[@]}"} \
        ${caustic_args[@]+"${caustic_args[@]}"} \
        ${tess_args[@]+"${tess_args[@]}"} \
        ${report_args[@]+"${report_args[@]}"} \
        "${lag_args[@]}" 2>&1 | tee "${run_log}"
    rc=${PIPESTATUS[0]}

    if [ "${AUTO_TUNE_REPORT}" = "1" ]; then
        report=$(grep -a '^AUTO-TUNE-REPORT' "${run_log}" | head -1)
        rm -f "${run_log}"
        if [ "${rc}" -eq 0 ] && [ -n "${report}" ]; then
            echo "  AUTO-TUNE-REPORT snap=${n_str} ${report#AUTO-TUNE-REPORT }"
            n_ok=$((n_ok+1))
        else
            echo "  Error: no auto-tune report for snapshot ${n_str} (exit ${rc})"
            n_failed=$((n_failed+1))
        fi
        continue
    fi

    # Strip ANSI colour codes from the saved log so it stays grep-able. Written through a temp file
    # rather than 'sed -i': BSD sed needs a suffix ARGUMENT after -i and GNU sed does not, so the
    # portable spelling of in-place editing is not to use it.
    if sed -E $'s/\033\\[[0-9;]*m//g' "${run_log}" > "${run_log}.tmp" 2>/dev/null; then
        mv "${run_log}.tmp" "${run_log}"
    else
        rm -f "${run_log}.tmp"
    fi

    # Peak RSS, in whichever dialect TIME_WRAP produced (absent when neither exists -- then the
    # awk finds nothing and the block below is skipped).
    #   BSD/macOS : "  1234567  maximum resident set size"            -- BYTES in $1
    #   GNU       : "  Maximum resident set size (kbytes): 1234"      -- KBYTES in $NF
    peak_bytes=$(awk 'tolower($0) ~ /maximum resident set size/ {
                          if (tolower($0) ~ /kbytes/) { gsub(/[^0-9]/, "", $NF); print $NF * 1024 }
                          else                        { print $1 }
                          exit }' "${run_log}")
    if [ -n "${peak_bytes}" ]; then
        peak_gb=$(awk -v b="${peak_bytes}" 'BEGIN{printf "%.1f", b/1073741824}')
        # macOS also reports the peak FOOTPRINT: what the run really needed, including the memory
        # the system compressed or swapped out -- RSS alone hid 90 GB behind "47 GB" at TNG100-3 z=0
        fp_bytes=$(awk 'tolower($0) ~ /peak memory footprint/ { print $1; exit }' "${run_log}")
        if [ -n "${fp_bytes}" ]; then
            peak_gb="$(awk -v b="${fp_bytes}" 'BEGIN{printf "%.1f", b/1073741824}') GB footprint, ${peak_gb} GB resident;"
        else
            peak_gb="${peak_gb} GB resident;"
        fi
        # the budget the binary's auto-tuner planned against (60% of RAM, less what other programs
        # use; see src/auto_tune.h) -- absent when --partition and --max-concurrent were both given
        budget_gb=$(sed -n 's/^.*memory budget: \([0-9.]*\) GB.*$/\1/p' "${run_log}" | head -1)
        if [ -n "${budget_gb}" ]; then
            echo "  Peak memory: ${peak_gb} auto-tune budget ${budget_gb} GB (DTFE_MEM_BUDGET_GB or MAX_CONCURRENT=1 caps it further)"
        else
            echo "  Peak memory: ${peak_gb} partition and concurrency set by hand, so no auto-tune budget applied"
        fi
    fi
    echo "  Wall time: $((SECONDS/60)) min $((SECONDS%60)) s"

    if [ "${rc}" -eq 0 ]; then
        echo "  Snapshot ${n_str} processed successfully"
        n_ok=$((n_ok+1))
    else
        echo "  Error processing snapshot ${n_str}"
        n_failed=$((n_failed+1))
    fi
    echo ""
done

# Honest exit status: processing NOTHING, or failing a snapshot that was attempted, exits 1. A
# partial skip stays exit 0 -- a partly-downloaded snapshot ladder is normal.
echo "PS-DTFE processing complete: ${n_ok} processed, ${n_skipped} skipped, ${n_failed} failed."
if [ "${n_failed}" -gt 0 ]; then
    echo "ERROR: ${n_failed} snapshot(s) failed -- see the messages above and the .runlog files." >&2
    exit 1
fi
if [ "${n_ok}" -eq 0 ]; then
    echo "ERROR: no snapshot was processed. ${n_skipped} were skipped for missing inputs -- check" >&2
    echo "       DATA_ROOT/SIMULATION (looked under '${DATA_DIR}'), and that each snapshot has" >&2
    echo "       Lagrangian positions: either a combined_ics.hdf5 beside it or its own" >&2
    echo "       'InitialCoordinates' dataset (the per-snapshot warnings above say which)." >&2
    exit 1
fi
