# Aria Training & Evaluation Guide



## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Training the Model](#training-the-model)
4. [Understanding Training Logs](#understanding-training-logs)
5. [Extracting Test Samples](#extracting-test-samples)
6. [Running Inference & Visualization](#running-inference--visualization)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisite
### Dataset Requirements
your aria dataset should be in rlds format with the following structure:
```
datasets/open-x-embodiment/aria_dataset/
├── 1.0.0/
│   ├── dataset_info.json
│   └── aria_dataset-train.tfrecord-*
```

the dataset must include:
- `image_primary`: rgb images from aria glasses
- `classified_bboxes`: detected objects with gaze classification
- `reasoning`: embodied-cot reasoning chain
- `action`: 7-dof actions (position, rotation, gripper)
- `language_instruction`: task instruction

---

## Dataset Setup

### Option 1: Use Existing RLDS Dataset
if you already have an rlds-formatted dataset:
```bash
# set the data directory
export DATA_DIR=/path/to/datasets/open-x-embodiment
```

### Option 2: Convert Your Data to RLDS
if you need to create an rlds dataset from scratch:
```bash
# see scripts/generate_embodied_data/ for conversion tools
python scripts/generate_embodied_data/bounding_boxes/generate_bboxes_with_gaze.py \
    --input_dir /path/to/aria/recordings \
    --output_dir datasets/open-x-embodiment/aria_dataset
```

### Training with Gaze Classification
to use eye gaze for classifying objects as PRIMARY/GAZE_FOCUS/AUXILIARY:

```bash
python vla-scripts/finetune.py \
    --dataset_name aria_dataset \
    --use_gaze_classification true \
    --batch_size 8 \
    --max_steps 50000
```
#### Model & Dataset
- `--vla_path`: huggingface model path (default: `openvla/openvla-7b`)
- `--data_root_dir`: root directory containing rlds datasets
- `--dataset_name`: name of your dataset folder (e.g., `aria_dataset`)

#### Training Configuration
- `--batch_size`: batch size per gpu
- `--max_steps`: total training steps (~50k usually)
- `--save_steps`: checkpoint frequency (default: 5000)
- `--learning_rate`: learning rate (default: 2e-5)
- `--grad_accumulation_steps`: gradient accumulation (use 2-4 if batch size is too small)

#### LoRA Settings
- `--use_lora`: whether to use lora finetuning
- `--lora_rank`: rank of lora matrices (default: 32)
- `--lora_dropout`: dropout for lora (default: 0.0)
- `--use_quantization`: 4-bit quantization (saves memory but hurts performance)

#### Aria-Specific
- `--use_gaze_classification`: use eye gaze to classify object importance
  - `true`: objects are classified as PRIMARY/GAZE_FOCUS/AUXILIARY
  - `false`: all objects treated equally

#### Logging
- `--bbox_log_dir`: where to save bbox visualizations (default: `training_logs/bboxes`)
- `--bbox_log_frequency`: log every N steps (default: 100)
- `--disable_bbox_logging`: set to true to disable logging (not recommended)

#### OOM error fixes
1. **reduce batch size**: start with `--batch_size 4` or `--batch_size 2`
2. **increase gradient accumulation**: `--grad_accumulation_steps 4`
3. **reduce lora rank**: `--lora_rank 16`
4. **enable quantization**: `--use_quantization true` (untested)

### Training Output Structure

after training starts, you'll see:
```
runs/
└── openvla-7b+aria_dataset+b8+lr-2e-05+lora-r32+dropout-0.0/
    ├── config.json
    ├── dataset_statistics.json  # critical! needed for inference
    ├── preprocessor_config.json
    ├── model.safetensors
    └── ...

training_logs/
└── bboxes/
    ├── step_100/
    │   ├── frame_0.png
    │   ├── frame_0.json
    │   └── ...
    ├── step_200/
    └── ...
```

---
every `bbox_log_frequency` steps (default: 100), the callback saves:

#### 1. Visualization Images (`frame_N.png`)
- rgb image with detected object bounding boxes
- gaze-focused objects highlighted in red
- other objects in gray

#### 2. Reasoning Chain Data (`frame_N.json`)
```json
{
  "step": 100,
  "instruction": "pick up the red block",
  "all_objects": ["red block", "blue cup", "table"],
  "gaze_target": "red block",
  "num_objects": 3,
  "reasoning": {
    "task": "pick up the red block from the table",
    "plan": "locate red block, approach it, grasp it, lift it up",
    "subtask_reasoning": "first need to reach the red block",
    "subtask": "move gripper towards red block",
    "move_reasoning": "red block is to the right and slightly forward",
    "move": "move right 0.15m, forward 0.08m",
    "gripper_position": "open gripper",
    "visible_objects": "red block at center, blue cup to left"
  }
}
```

### Embodied-CoT Reasoning Components

the full reasoning chain includes:

1. **TASK**: high-level task description
2. **PLAN**: overall strategy to accomplish the task
3. **SUBTASK_REASONING**: why this subtask is needed now
4. **SUBTASK**: current subtask being executed
5. **MOVE_REASONING**: spatial reasoning for the next action
6. **MOVE**: specific movement command
7. **GRIPPER_POSITION**: gripper state (open/closed)
8. **VISIBLE_OBJECTS**: what objects the robot can see

## Extracting Test Samples

before running inference visualization, extract test samples from your validation set:

```bash
python scripts/extract_test_samples.py \
    --rlds_dir datasets/open-x-embodiment/aria_dataset \
    --output_json test_samples.json \
    --num_samples 50 \
    --output_images_dir test_images
```

### Parameters
- `--rlds_dir`: path to your rlds dataset
- `--output_json`: where to save test sample metadata
- `--num_samples`: how many samples to extract (default: 5)
- `--output_images_dir`: where to save extracted images

### Output Format
creates `test_samples.json` with:
```json
[
  {
    "episode_idx": 0,
    "step_idx": 25,
    "image_path": "test_images/sample_0.png",
    "instruction": "pick up the red block",
    "state": [[x, y, z, roll, pitch, yaw, gripper], [...]],
    "action": [[dx, dy, dz, droll, dpitch, dyaw, dgripper], [...]],
    "bboxes_with_class": [...],
    "bboxes_no_class": [...],
    "has_classification": true
  },
  ...
]
```

---

## Running Inference & Visualization

### Basic Visualization

compare your finetuned model against base openvla:

```bash
python scripts/visualize_inference.py \
    --test_json test_samples.json \
    --finetuned_a runs/openvla-7b+aria_dataset+b8+lr-2e-05+lora-r32+dropout-0.0 \
    --output_dir eval_viz \
    --device cuda:0
```

### Compare Multiple Models

compare base model vs two finetuned variants:

```bash
python scripts/visualize_inference.py \
    --test_json test_samples.json \
    --base_model openvla/openvla-7b \
    --finetuned_a runs/model_with_gaze_classification \
    --finetuned_b runs/model_without_gaze_classification \
    --training_logs_dir training_logs/bboxes \
    --output_dir eval_viz \
    --device cuda:0
```

### Parameters Explained

- `--test_json`: test samples from `extract_test_samples.py`
- `--base_model`: pretrained model for comparison (optional)
- `--finetuned_a`: first finetuned model checkpoint
- `--finetuned_b`: second finetuned model checkpoint (optional)
- `--training_logs_dir`: training logs directory for reasoning data (optional)
- `--output_dir`: where to save visualizations
- `--device`: cuda device (default: cuda:0)

for each test sample and each model:

1. **Visualization Image** (`sample_N_<model>.png`)
   - 3-panel layout showing:
     - **left**: input image with detected objects
     - **middle**: full embodied-cot reasoning chain
     - **right**: predicted action values

2. **Action Data** (`sample_N_<model>_action.json`)
   ```json
   {
     "instruction": "pick up the red block",
     "action": [dx, dy, dz, droll, dpitch, dyaw, dgripper],
     "model": "finetuned_a",
     "reasoning": { ... }
   }
   ```
### Output Guide

#### Action Values
- **position (dx, dy, dz)**: cartesian velocity in m/s
- **rotation (droll, dpitch, dyaw)**: angular velocity in rad/s
- **gripper**: -1 = open, +1 = closed
- **action norm**: magnitude of motion (higher = more aggressive)
