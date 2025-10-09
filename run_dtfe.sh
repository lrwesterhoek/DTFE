# Set the simulation number and grid size
n=(0 1 2 3 4 5 8 13 17 21 25 33 40 50 67 72 78 84 91 99)
grid=512

# Base directory path
base_dir="/Users/luukw/Documents/Rijksuniversiteit Groningen/Physics/Year 3/Bachelor Research Project PH/Code/DTFE/"

for i in "${n[@]}"; do
    n_str=$(printf "%03d" "$i")
    
    echo "Processing snapshot $n_str to ${dest_dir}..."
    
    # Download to the specific directory
    #wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition \
    #     -P "${dest_dir}" \
    #     --header="API-Key: 544f44ca2820e2600e6fd85e1b553564" \
    #     "http://www.tng-project.org/api/TNG50-3/files/snapshot-${i}/?format=api"
         
         ./DTFE output/TNG50-3-Dark/snapdir_${n_str}/combined_${n_str}.hdf5 output/TNG50-3-Dark/snapdir_${n_str}/output --grid ${grid} --padding 20 --periodic --partition 2 2 1 --field density_a velocity_a gradient_a divergence_a shear_a vorticity_a
done
