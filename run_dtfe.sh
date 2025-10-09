# ============================================
# DTFE Processing Script Configuration
# ============================================

# Simulation snapshots to process
SNAPSHOTS=(0 1 2 3 4 5 8 13 17 21 25 33 40 50 67 72 78 84 91 99)

# Grid parameters
GRID_SIZE=512
PADDING=20
PARTITION="2 2 1"

# Data locations (relative to script directory)
DATA_DIR="output/TNG50-3-Dark"
INPUT_SUBDIR="snapdir"
OUTPUT_SUBDIR="output"

# DTFE fields to compute
FIELDS="density_a velocity_a gradient_a divergence_a shear_a vorticity_a"

# ============================================
# Script Execution
# ============================================

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "Starting DTFE processing..."
echo "Data directory: ${DATA_DIR}"
echo "Grid size: ${GRID_SIZE}"
echo ""

for i in "${SNAPSHOTS[@]}"; do
    # Format snapshot number with leading zeros
    n_str=$(printf "%03d" "$i")
    
    # Define paths
    input_dir="${DATA_DIR}/${INPUT_SUBDIR}_${n_str}"
    input_file="${input_dir}/combined_${n_str}.hdf5"
    output_dir="${input_dir}/${OUTPUT_SUBDIR}"
    
    echo "Processing snapshot ${n_str}..."
    
    # Create output directory if it doesn't exist
    mkdir -p "${output_dir}"
    
    # Check if input file exists
    if [ ! -f "${input_file}" ]; then
        echo "  Warning: Input file not found: ${input_file}"
        echo "  Skipping snapshot ${n_str}"
        continue
    fi
    
    # Run DTFE
    echo "  Running DTFE on ${input_file}..."
    ./DTFE "${input_file}" "${output_dir}" \
        --grid ${GRID_SIZE} \
        --padding ${PADDING} \
        --periodic \
        --partition ${PARTITION} \
        --field ${FIELDS}
    
    if [ $? -eq 0 ]; then
        echo "  Snapshot ${n_str} processed successfully"
    else
        echo "  Error processing snapshot ${n_str}"
    fi
    echo ""
done

echo "DTFE processing complete!"