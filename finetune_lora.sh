#!/bin/bash
#SBATCH --job-name=vla_lora_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# load modules if needed (uncomment and adjust as needed)
# module load cuda/12.1
# module load python/3.11

# activate venv
source .venv_train/bin/activate

# create logs and checkpoint directories
mkdir -p logs
mkdir -p checkpoint/lora_adapters
mkdir -p checkpoint/lora_runs

# set environment variables
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME="${PWD}/.cache"

# number of gpus
NGPUS=2

# batch size settings for A100s
# for A100 80GB: batch_size=16, grad_accum=1 (uses ~72GB per GPU)
# for A100 40GB: batch_size=8, grad_accum=1 (uses ~36GB per GPU)
BATCH_SIZE=16  # use 16 for A100 80GB, or 8 for A100 40GB
GRAD_ACCUM=1

echo "starting lora fine-tuning with ${NGPUS} GPUs"
echo "batch_size=${BATCH_SIZE}, grad_accumulation_steps=${GRAD_ACCUM}"

# run lora fine-tuning
# note: wandb args are omitted to disable wandb logging
torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node ${NGPUS} \
  vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "${PWD}/bridge_orig" \
  --dataset_name bridge_orig \
  --run_root_dir "${PWD}/checkpoint/lora_runs" \
  --adapter_tmp_dir "${PWD}/checkpoint/lora_adapters" \
  --lora_rank 32 \
  --batch_size ${BATCH_SIZE} \
  --grad_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate 5e-4 \
  --image_aug False \
  --save_steps 2500

echo "fine-tuning completed"
