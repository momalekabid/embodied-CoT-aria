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

export HF_TOKEN=<hugging face token>
echo "hugging face token" > .hf_token

deactivate 2>/dev/null || true
unset VIRTUAL_ENV

python -m venv ${SCRATCH}/embodied-CoT-aria/.venv_train
source ${SCRATCH}/embodied-CoT-aria/.venv_train/bin/activate

pip install --upgrade pip setuptools wheel ninja packaging
pip install --no-cache-dir -e .
pip install --no-cache-dir zmq
pip install --no-cache-dir tensorrt==10.13.2.6
pip install --no-cache-dir nvidia-cuda-runtime-cu12
pip install --no-cache-dir huggingface_hub

pip uninstall -y flash-attn flash_attn
pip install flash-attn==2.7.3 --no-build-isolation

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/train.py \
	 --vla.type "prism-dinosiglip-224px+mx-bridge" \
	 --vla.expected_world_size 2 --vla.global_batch_size 64 \
	 --vla.per_device_batch_size 32 \ 
	 --data_root_dir "/cluster/scratch/jterrassier/embodied-CoT-aria/" \
	 --run_root_dir "/cluster/scratch/jterrassier/embodied-CoT-aria/checkpoint/" \
	 --trackers jsonl
