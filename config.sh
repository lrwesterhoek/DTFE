# Shared configuration for run_dtfe.sh, run_ps_dtfe.sh and download_snapshots.sh.
# Sourced, not executed. Every value can be overridden per run:
#   - environment: DTFE_DATA_ROOT=/scratch/tng DTFE_SIM=TNG50-3-Dark ./run_ps_dtfe.sh
#   - script flags: ./run_ps_dtfe.sh -d /path/to/sim -g 256 99
# The Python side (python/dtfelib/cli.py) honours the same DTFE_DATA_ROOT / DTFE_SIM variables.

# Root directory holding one subdirectory per simulation (e.g. $DATA_ROOT/TNG300-3-Dark).
DATA_ROOT="${DTFE_DATA_ROOT:-$HOME/output}"

# Simulation to process; snapshots live in $DATA_ROOT/$SIMULATION/snapdir_NNN/.
SIMULATION="${DTFE_SIM:-TNG50-3-Dark}"

# Snapshot numbers (each corresponds to a redshift) the run scripts process by default;
# positional arguments to the scripts override this list.
SNAPSHOTS=(99)

# Interpolation grid cells per axis and padding (mean interparticle spacings).
GRID_SIZE=512
PADDING=25

# Subdirectory prefix of the snapshot chunks inside the simulation directory.
INPUT_SUBDIR="snapdir"
