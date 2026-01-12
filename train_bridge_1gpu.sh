#!/bin/bash

#SBATCH --job-name=bridge_1gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpumem:80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bridge_1gpu_%j.out
#SBATCH --error=logs/bridge_1gpu_%j.err

# bridge dataset training pipeline - single 80gb a100

# load modules
module purge
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

# change to repo directory
cd ${SCRATCH}/embodied-cot-aria

# environment setup
export HF_HOME=${SCRATCH}/embodied-cot-aria/huggingface
export TORCH_HOME=${SCRATCH}/embodied-cot-aria/torch
export PIP_CACHE_DIR=${SCRATCH}/embodied-cot-aria/.pip_cache
export TMPDIR=${SCRATCH}/embodied-cot-aria/.tmp

mkdir -p $PIP_CACHE_DIR $TMPDIR logs
nvidia-smi

# tokens
export HF_TOKEN=
echo "" >.hf_token
export WANDB_API_KEY=

# venv setup
deactivate 2>/dev/null || true
unset VIRTUAL_ENV

python -m venv ${SCRATCH}/embodied-cot-aria/.venv_train
source ${SCRATCH}/embodied-cot-aria/.venv_train/bin/activate

# install dependencies
echo "=== installing dependencies ==="
pip install --upgrade pip setuptools wheel ninja packaging
pip install --no-cache-dir -e .
pip install --no-cache-dir zmq
pip install --no-cache-dir tensorrt==10.13.2.6
pip install --no-cache-dir nvidia-cuda-runtime-cu12
pip install --no-cache-dir huggingface_hub
pip install --no-cache-dir wandb

# install flash-attn
pip uninstall -y flash-attn flash_attn
pip install flash-attn==2.7.3 --no-build-isolation

# login to wandb
wandb login

# paths
DATA_ROOT="${SCRATCH}/embodied-cot-aria"
RUN_ROOT="${SCRATCH}/embodied-cot-aria/checkpoints/bridge_1gpu"
ADAPTER_TMP="${SCRATCH}/embodied-cot-aria/adapter_tmp"
EVAL_DIR="${SCRATCH}/embodied-cot-aria/evaluation/bridge_1gpu"

mkdir -p $RUN_ROOT $ADAPTER_TMP $EVAL_DIR

echo "=== starting training run ==="

# single gpu training - optimized for 80gb a100
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir $DATA_ROOT \
  --dataset_name bridge_orig \
  --run_root_dir $RUN_ROOT \
  --adapter_tmp_dir $ADAPTER_TMP \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 100 \
  --max_steps 500 \
  --shuffle_buffer_size 1000 \
  --wandb_project "bridge-1gpu-train" \
  --wandb_entity "mabid-university-of-zurich"

echo "=== extracting test samples ==="

TEST_JSON="${EVAL_DIR}/test_samples.json"
TEST_IMAGES="${EVAL_DIR}/test_images"

python scripts/extract_test_samples.py \
  --rlds_dir ${DATA_ROOT}/bridge_orig \
  --output_json $TEST_JSON \
  --num_samples 20 \
  --output_images_dir $TEST_IMAGES

echo "=== running inference visualization ==="

VIZ_DIR="${EVAL_DIR}/visualizations"
CHECKPOINT=$(find $RUN_ROOT -maxdepth 1 -type d -name "openvla-7b*" | head -n 1)

python scripts/visualize_inference.py \
  --test_json $TEST_JSON \
  --base_model "openvla/openvla-7b" \
  --finetuned_a $CHECKPOINT \
  --output_dir $VIZ_DIR \
  --device cuda:0

echo "=== pipeline complete ==="
echo "check logs/bridge_1gpu_*.out for full output"
echo "visualizations saved to: $VIZ_DIR"
