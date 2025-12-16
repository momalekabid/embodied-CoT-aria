#!/bin/bash

# aria training - uv version (self-contained)

set -e
cd "$(dirname "$0")"

# install uv (idempotent - skips if already installed)
echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

# use full path to uv
UV="/root/.local/bin/uv"

# environment setup
export HF_HOME=$(pwd)/huggingface
export TORCH_HOME=$(pwd)/torch
export UV_CACHE_DIR=$(pwd)/.uv_cache
export TMPDIR=$(pwd)/.tmp

# suppress tf warnings
export TF_CPP_MIN_LOG_LEVEL=2

# tensorrt library path (for system tensorrt)
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/tensorrt_libs:$LD_LIBRARY_PATH

mkdir -p $UV_CACHE_DIR $TMPDIR logs
nvidia-smi

# tokens
export HF_TOKEN=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
echo "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" >.hf_token
export WANDB_API_KEY=469490f5d68b8546bcdddf2b4a2127dcc2e379d2
export GEMINI_API_KEY=AIzaSyAO2VkR-fpiwWc4wGAlOupRZW2AKPYKvd4

# clean old venv
#rm -rf .venv_train

# install python 3.11.6
echo "=== installing python 3.11.6 ==="
#$UV python install 3.11.6

echo "=== creating venv with uv (python 3.11.6, system-site-packages) ==="
#$UV venv --python 3.11.6 --system-site-packages .venv_train
source .venv_train/bin/activate

# install dependencies
echo "=== installing base dependencies ==="
$UV pip install --no-cache --upgrade pip setuptools wheel ninja packaging

# install aria_dataset builder FIRST (so tfds can find it)
echo "=== installing aria_dataset builder ==="
cd aria_rlds_builder-main && $UV pip install --no-cache -e . && cd ..

# install main package
echo "=== installing main package ==="
$UV pip install --no-cache -e .

# install extra deps
echo "=== installing extra dependencies ==="
$UV pip install --no-cache zmq huggingface_hub wandb google-generativeai

# flash-attn with optimized build
echo "=== installing flash-attn (optimized build) ==="
export MAX_JOBS=4
export FLASH_ATTN_CUDA_ARCHS="80"  # ampere (a100)
export TORCH_CUDA_ARCH_LIST="8.0"
$UV pip install --no-cache flash-attn==2.7.3 --no-build-isolation

echo "=== verifying installation ==="
python -c "import torch; import flash_attn; print(f'torch: {torch.__version__}, flash_attn: {flash_attn.__version__}')"
python -c "import aria_dataset; print('AriaDataset class exists:', hasattr(aria_dataset, 'AriaDataset'))"
python -c "import tensorflow_datasets as tfds; print('tfds:', tfds.__version__)"

wandb login

# paths
DATA_ROOT="$(pwd)"
RUN_ROOT="$(pwd)/checkpoints/aria_1gpu"
ADAPTER_TMP="$(pwd)/adapter_tmp"
BBOX_LOG_DIR="$(pwd)/training_logs/bboxes/aria_1gpu"
EVAL_DIR="$(pwd)/evaluation/aria_1gpu"

mkdir -p $RUN_ROOT $ADAPTER_TMP $BBOX_LOG_DIR $EVAL_DIR

echo "=== preprocessing aria dataset (generating reasoning) ==="

# delete old reasonings to regenerate with improved hand-filtered bbox classification
rm -f aria_reasonings_*.json
echo "  deleted old reasoning files to regenerate with hand-filtered bboxes"

if [ -z "$GEMINI_API_KEY" ]; then
    echo "error: GEMINI_API_KEY not set. export GEMINI_API_KEY=your_key"
    exit 1
fi

python3 scripts/preprocess_aria_ecot.py \
    --tfrecord_dir ./aria_dataset \
    --api gemini \
    --api_key $GEMINI_API_KEY \
    --output_prefix aria_reasonings \
    --gpu 0 \
    --max_episodes 20 \
    --no_gaze_classification

# set paths for visualization to find checkpoints
RUN_DIR_WITH_GAZE="${RUN_ROOT}/with_gaze"
BBOX_LOG_WITH_GAZE="${BBOX_LOG_DIR}/with_gaze"

echo "=== training condition a: with gaze ==="
export REASONING_DATASET_PATH="${DATA_ROOT}/aria_reasonings_with_gaze.json"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "Embodied-CoT/ecot-openvla-7b-bridge" \
  --data_root_dir $DATA_ROOT \
  --dataset_name aria_dataset \
  --run_root_dir $RUN_DIR_WITH_GAZE \
  --adapter_tmp_dir $ADAPTER_TMP \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --image_aug False \
  --save_steps 25 \
  --max_steps 50 \
  --shuffle_buffer_size 1000 \
  --bbox_log_dir $BBOX_LOG_WITH_GAZE \
  --bbox_log_frequency 50 \
  --wandb_project "aria-ecot-train" \
  --wandb_entity "mabid-university-of-zurich"

# echo "=== training condition b: no gaze ==="
#
# RUN_DIR_NO_GAZE="${RUN_ROOT}/no_gaze"
# BBOX_LOG_NO_GAZE="${BBOX_LOG_DIR}/no_gaze"
# export REASONING_DATASET_PATH="${DATA_ROOT}/aria_reasonings_no_gaze.json"
#
# torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
#   --vla_path "Embodied-CoT/ecot-openvla-7b-bridge" \
#   --data_root_dir $DATA_ROOT \
#   --dataset_name aria_dataset \
#   --run_root_dir $RUN_DIR_NO_GAZE \
#   --adapter_tmp_dir $ADAPTER_TMP \
#   --lora_rank 32 \
#   --batch_size 8 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 1e-3 \
#   --image_aug False \
#   --save_steps 100 \
#   --max_steps 500 \
#   --shuffle_buffer_size 1000 \
#   --bbox_log_dir $BBOX_LOG_NO_GAZE \
#   --bbox_log_frequency 50 \
#   --wandb_project "aria-ecot-train" \
#   --wandb_entity "mabid-university-of-zurich"
#
# echo "=== training baseline: speech only (no llm) ==="
#
# RUN_DIR_SPEECH_ONLY="${RUN_ROOT}/speech_only"
# BBOX_LOG_SPEECH_ONLY="${BBOX_LOG_DIR}/speech_only"
# export REASONING_DATASET_PATH="${DATA_ROOT}/aria_reasonings_speech_only.json"
#
# torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
#   --vla_path "Embodied-CoT/ecot-openvla-7b-bridge" \
#   --data_root_dir $DATA_ROOT \
#   --dataset_name aria_dataset \
#   --run_root_dir $RUN_DIR_SPEECH_ONLY \
#   --adapter_tmp_dir $ADAPTER_TMP \
#   --lora_rank 32 \
#   --batch_size 8 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 1e-3 \
#   --image_aug False \
#   --save_steps 100 \
#   --max_steps 500 \
#   --shuffle_buffer_size 1000 \
#   --bbox_log_dir $BBOX_LOG_SPEECH_ONLY \
#   --bbox_log_frequency 50 \
#   --wandb_project "aria-ecot-train" \
#   --wandb_entity "mabid-university-of-zurich"

echo "=== extracting test samples ==="

TEST_JSON="${EVAL_DIR}/test_samples.json"
TEST_IMAGES="${EVAL_DIR}/test_images"

python3 scripts/extract_test_samples.py \
    --rlds_dir ${DATA_ROOT}/aria_dataset \
    --output_json $TEST_JSON \
    --num_samples 20 \
    --output_images_dir $TEST_IMAGES

echo "=== adding bboxes to test samples with hierarchical gaze classification ==="

python3 scripts/add_bboxes_to_test_samples.py \
    --test_json $TEST_JSON \
    --gaze_dir ./gaze_pixels \
    --gpu 0

echo "=== testing inference on sample 0 ==="

python3 scripts/test_inference.py \
    --model_path "${RUN_DIR_WITH_GAZE}/$(ls ${RUN_DIR_WITH_GAZE} | grep ecot-openvla | head -n 1)" \
    --image "${TEST_IMAGES}/sample_0.png" \
    --instruction "pick up the object" \
    --device cuda:0

echo "=== running inference visualization ==="

VIZ_DIR="${EVAL_DIR}/visualizations"
CHECKPOINT_WITH_GAZE=$(find $RUN_DIR_WITH_GAZE -maxdepth 1 -type d -name "ecot-openvla-7b-bridge*" | head -n 1)
# CHECKPOINT_NO_GAZE=$(find $RUN_DIR_NO_GAZE -maxdepth 1 -type d -name "ecot-openvla-7b-bridge*" | head -n 1)
# CHECKPOINT_SPEECH_ONLY=$(find $RUN_DIR_SPEECH_ONLY -maxdepth 1 -type d -name "ecot-openvla-7b-bridge*" | head -n 1)

echo "checkpoint:"
echo "  with_gaze: $CHECKPOINT_WITH_GAZE"
# echo "  no_gaze: $CHECKPOINT_NO_GAZE"
# echo "  speech_only: $CHECKPOINT_SPEECH_ONLY"

# visualize with_gaze only
python3 scripts/visualize_inference.py \
    --test_json $TEST_JSON \
    --base_model "Embodied-CoT/ecot-openvla-7b-bridge" \
    --finetuned_with_gaze $CHECKPOINT_WITH_GAZE \
    --output_dir $VIZ_DIR \
    --device cuda:0
#    --finetuned_no_gaze $CHECKPOINT_NO_GAZE \
#    --finetuned_speech_only $CHECKPOINT_SPEECH_ONLY \

echo "=== pipeline complete ==="
echo "outputs:"
echo "  checkpoints: $RUN_ROOT"
echo "  visualizations: $VIZ_DIR"
echo "  wandb: https://wandb.ai/mabid-university-of-zurich/aria-ecot-train"
