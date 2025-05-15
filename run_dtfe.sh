#!/bin/bash

# Set the simulation number and grid size
n=0
grid=512

# Format the simulation number with three digits (e.g., 50 -> 050)
n_str=$(printf "%03d" "$n")

# Loop over snapshot indices 0 to 3 (adjust the range if needed)
for i in {0..10}; do
    echo "Processing snapshot $i..."
    ./DTFE test/snapdir_${n_str}/snap_${n_str}."$i".hdf5 test/snapdir_${n_str}/output_${n_str}."$i" --grid "$grid" -f density velocity
done