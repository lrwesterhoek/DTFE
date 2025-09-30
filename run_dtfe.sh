# Set the simulation number and grid size
n=(0 1 2 3 4 5 8 13 17 21 25 33 40 50 67 72 78 84 91 99)
grid=512

# Base directory path
base_dir="/Users/luukw/Documents/Rijksuniversiteit Groningen/Physics/Year 3/Bachelor Research Project PH/Code/DTFE/output/TNG50-3"

for i in "${n[@]}"; do
    n_str=$(printf "%03d" "$i")
    
    # Create the destination directory
    dest_dir="${base_dir}/snapdir_${n_str}"
    mkdir -p "${dest_dir}"
    
    echo "Processing snapshot $n_str to ${dest_dir}..."
    
    # Download to the specific directory
    wget -nd -nc -nv -e robots=off -l 1 -r -A hdf5 --content-disposition \
         -P "${dest_dir}" \
         --header="API-Key: 544f44ca2820e2600e6fd85e1b553564" \
         "http://www.tng-project.org/api/TNG50-3/files/snapshot-${i}/?format=api"
done