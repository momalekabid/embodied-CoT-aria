#!/bin/bash

#SBATCH --job-name=download_bridge
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# create logs directory
mkdir -p logs

# set the base datasets directory (change this to your scratch dir)
BASE_DATASETS_DIR="/cluster/scratch/jterrassier/embodied-CoT-aria"

echo "downloading bridge dataset to ${BASE_DATASETS_DIR}"
echo "this will download ~124 GB of data"

cd ${BASE_DATASETS_DIR}

# download the full bridge dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# rename to bridge_orig (required by the training script)
if [ -d "bridge_dataset" ]; then
    echo "renaming bridge_dataset to bridge_orig"
    mv bridge_dataset bridge_orig
    echo "download complete! dataset is at ${BASE_DATASETS_DIR}/bridge_orig"
else
    echo "error: bridge_dataset directory not found after download"
    exit 1
fi

echo "verifying dataset structure..."
ls -lh ${BASE_DATASETS_DIR}/bridge_orig

echo "done!"
