#!/bin/bash

#SBATCH --job-name=train_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=100G
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpumem:60G
#SBATCH --time=24:00:00

module purge
module load stack/2024-06 python_cuda/3.11.6 eth_proxy

export HF_HOME=${SCRATCH}/embodied-CoT-aria/huggingface
export TORCH_HOME=${SCRATCH}/embodied-CoT-aria/torch
export PIP_CACHE_DIR=${SCRATCH}/embodied-CoT-aria/.pip_cache
export TMPDIR=${SCRATCH}/embodied-CoT-aria/.tmp

mkdir -p $PIP_CACHE_DIR $TMPDIR

export HF_TOKEN=
echo "" >.hf_token

deactivate 2>/dev/null || true
unset VIRTUAL_ENV

export WANDB_API_KEY=

python -m venv ${SCRATCH}/embodied-CoT-aria/.venv_train
source ${SCRATCH}/embodied-CoT-aria/.venv_train/bin/activate

pip install --upgrade pip setuptools wheel ninja packaging
pip install --no-cache-dir -e .
pip install --no-cache-dir zmq
pip install --no-cache-dir tensorrt==10.13.2.6
pip install --no-cache-dir nvidia-cuda-runtime-cu12
pip install --no-cache-dir huggingface_hub
pip install --no-cache-dir wandb

pip uninstall -y flash-attn flash_attn
pip install flash-attn==2.7.3 --no-build-isolation

wandb login

# test run with limited data - batch size 2, only 50 max_steps for testing
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "${SCRATCH}/embodied-cot-aria/" \
  --dataset_name bridge_orig \
  --run_root_dir "${SCRATCH}/embodied-cot-aria/checkpoint/" \
  --adapter_tmp_dir "${SCRATCH}/embodied-cot-aria/weights/" \
  --lora_rank 32 \
  --batch_size 2 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --save_steps 25 \
  --max_steps 50 \
  --shuffle_buffer_size 1000 \
  --wandb_project "embodied-cot-test" \
  --wandb_entity "mabid-university-of-zurich"
