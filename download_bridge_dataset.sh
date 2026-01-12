#!/bin/bash

#SBATCH --job-name=download_bridge
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# load eth proxy module to access external sites
module purge
module load stack/2024-06 eth_proxy

# create logs directory
mkdir -p logs

# set the base datasets directory (use your own scratch)
BASE_DATASETS_DIR="${SCRATCH}/embodied-cot-aria"

echo "downloading bridge dataset to ${BASE_DATASETS_DIR}"
echo "this will download ~124 GB of data"
echo "using proxy: ${https_proxy}"

# create the directory if it doesn't exist
mkdir -p ${BASE_DATASETS_DIR}
cd ${BASE_DATASETS_DIR}

# download the full bridge dataset (124 GB)
echo "starting download..."
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# check if download succeeded
if [ $? -ne 0 ]; then
    echo "error: wget failed"
    exit 1
fi

# rename to bridge_orig (required by the training script)
if [ -d "bridge_dataset" ]; then
    echo "renaming bridge_dataset to bridge_orig"
    mv bridge_dataset bridge_orig
    echo "download complete! dataset is at ${BASE_DATASETS_DIR}/bridge_orig"
    ls -lh ${BASE_DATASETS_DIR}/bridge_orig
else
    echo "error: bridge_dataset directory not found after download"
    exit 1
fi

echo "done!"
