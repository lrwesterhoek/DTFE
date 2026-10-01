# Shared configuration for run_dtfe.sh, run_ps_dtfe.sh and download_snapshots.sh.
# Sourced, not executed. Every value can be overridden per run:
#   - environment: DTFE_DATA_ROOT=/scratch/tng DTFE_SIM=TNG50-3-Dark ./run_ps_dtfe.sh
#                  GRID_SIZE=256 ./run_ps_dtfe.sh          (or DTFE_GRID_SIZE / DTFE_PADDING)
#   - script flags: ./run_ps_dtfe.sh -d /path/to/sim -g 256 99
# The Python side (python/dtfelib/cli.py) honours the same DTFE_DATA_ROOT / DTFE_SIM variables.

# Root of the simulation data. Two layouts are recognised (see sim_dir below):
#   per family  $DATA_ROOT/<family>/<sim>/snapdir_NNN/   the Samsung T7 (default):
#               Illustris TNG/TNG100/TNG100-3-Dark, Illustris TNG/TNG300/TNG300-3-Dark, ...
#   flat        $DATA_ROOT/<sim>/snapdir_NNN/            e.g. DTFE_DATA_ROOT=~/output (the old copy)
# where <family> is the simulation name up to its first '-' (TNG100-3-Dark -> TNG100). With the
# T7 unplugged the run scripts stop at "no snapshots under ..." rather than writing elsewhere.
DATA_ROOT="${DTFE_DATA_ROOT:-/Volumes/Samsung T7/Illustris TNG}"

# The directory of simulation $1 under DATA_ROOT, in whichever layout holds it: an existing
# <sim> or <family>/<sim> directory wins; a simulation not on disk yet goes into its family
# folder when that exists (so downloads land beside their siblings), else flat.
# Python: dtfelib.cli.sim_dir -- keep the two in step.
sim_dir() {
    local sim="$1" fam="${1%%-*}"
    if [ -d "${DATA_ROOT}/${sim}" ]; then
        echo "${DATA_ROOT}/${sim}"
    elif [ "${fam}" != "${sim}" ] && [ -d "${DATA_ROOT}/${fam}" ]; then
        echo "${DATA_ROOT}/${fam}/${sim}"
    else
        echo "${DATA_ROOT}/${sim}"
    fi
}

# Simulation to process; snapshots live in $(sim_dir $SIMULATION)/snapdir_NNN/.
SIMULATION="${DTFE_SIM:-TNG50-3-Dark}"

# Snapshot numbers (each corresponds to a redshift) the run scripts process by default;
# positional arguments to the scripts override this list.
SNAPSHOTS=(99)

# Interpolation grid cells per axis and padding (mean interparticle spacings).
# Environment-overridable like DATA_ROOT above. The run scripts source this file BEFORE parsing
# their flags, so precedence is:  -g flag  >  environment  >  the default here.
GRID_SIZE="${GRID_SIZE:-${DTFE_GRID_SIZE:-1024}}"
PADDING="${PADDING:-${DTFE_PADDING:-50}}"

# Subdirectory prefix of the snapshot chunks inside the simulation directory.
INPUT_SUBDIR="snapdir"
