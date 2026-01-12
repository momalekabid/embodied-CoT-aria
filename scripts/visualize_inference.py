#!/usr/bin/env python3
"""
visualize inference results for aria ablation study
shows detected objects, gaze classification, predicted actions, and embodied-cot reasoning
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import sys

# add prismatic to path for cot utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from prismatic.util.cot_utils import CotTag


def parse_reasoning(reasoning_str):
    """parse reasoning string into dict of components"""
    if reasoning_str is None or reasoning_str == "":
        return {}

    # decode if bytes
    if isinstance(reasoning_str, bytes):
        reasoning_str = reasoning_str.decode()

    reasoning_dict = {}
    # split on @ separator
    parts = reasoning_str.split('@')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # try to match each cot tag
        for tag in CotTag:
            if part.startswith(tag.value):
                # extract value after the tag
                value = part[len(tag.value):].strip()
                reasoning_dict[tag.name.lower()] = value
                break

    return reasoning_dict


def load_model(model_path, device="cuda:0"):
    """load openvla model"""
    print(f"loading {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    model.eval()
    return processor, model


def run_inference(processor, model, image, instruction, device="cuda:0"):
    """run inference and get predicted action"""
    inputs = processor(instruction, image).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="aria_dataset", do_sample=False)
    return action.cpu().numpy()


def draw_frame_with_info(image, bboxes, action, instruction, model_name, show_classification=False, reasoning=None):
    """
    draw single frame with:
    - detected objects (bboxes)
    - classification if enabled
    - predicted action overlay
    - embodied-cot reasoning chain
    """
    # create figure with 3 columns if reasoning is provided
    if reasoning:
        fig, (ax_img, ax_reasoning, ax_action) = plt.subplots(1, 3, figsize=(20, 6),
                                                gridspec_kw={'width_ratios': [2, 1.5, 1]})
    else:
        fig, (ax_img, ax_action) = plt.subplots(1, 2, figsize=(16, 6),
                                                gridspec_kw={'width_ratios': [2, 1]})

    # === left: image with bboxes ===
    img = image.copy()
    draw = ImageDraw.Draw(img, 'RGBA')

    # colors for categories
    colors = {
        'PRIMARY': (255, 50, 50),
        'GAZE_FOCUS': (50, 255, 50),
        'AUXILIARY': (150, 150, 150),
        'NONE': (100, 150, 255)
    }

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # draw bboxes
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(c) for c in bbox['bbox']]
        label = bbox['label']
        category = bbox.get('category', 'NONE') if show_classification else 'NONE'
        color = colors.get(category, (100, 150, 255))

        # thick bbox
        for offset in range(4):
            draw.rectangle([x1-offset, y1-offset, x2+offset, y2+offset],
                          outline=color, width=1)

        # label
        if show_classification and category != 'NONE':
            text = f"{label}\n[{category}]"
        else:
            text = label

        bbox_text = draw.textbbox((x1, y1-35), text, font=font_small)
        draw.rectangle([bbox_text[0]-3, bbox_text[1]-3, bbox_text[2]+3, bbox_text[3]+3],
                      fill=color + (220,))
        draw.text((x1, y1-35), text, fill=(255, 255, 255), font=font_small)

    ax_img.imshow(img)
    ax_img.set_title(f"{model_name}\n\"{instruction}\"", fontsize=14, fontweight='bold', pad=15)
    ax_img.axis('off')

    # add legend if classification enabled
    if show_classification:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(1.0, 0.2, 0.2), label='PRIMARY'),
            Patch(facecolor=(0.2, 1.0, 0.2), label='GAZE_FOCUS'),
            Patch(facecolor=(0.6, 0.6, 0.6), label='AUXILIARY')
        ]
        ax_img.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # === middle: embodied-cot reasoning (if provided) ===
    if reasoning:
        ax_reasoning.axis('off')

        reasoning_dict = parse_reasoning(reasoning) if isinstance(reasoning, str) else reasoning

        reasoning_text = (
            f"Embodied-CoT Reasoning\n"
            f"{'='*30}\n\n"
        )

        # add each reasoning component
        if "task" in reasoning_dict and reasoning_dict["task"]:
            reasoning_text += f"TASK:\n{reasoning_dict['task']}\n\n"

        if "plan" in reasoning_dict and reasoning_dict["plan"]:
            reasoning_text += f"PLAN:\n{reasoning_dict['plan']}\n\n"

        if "subtask_reasoning" in reasoning_dict and reasoning_dict["subtask_reasoning"]:
            reasoning_text += f"SUBTASK REASONING:\n{reasoning_dict['subtask_reasoning']}\n\n"

        if "subtask" in reasoning_dict and reasoning_dict["subtask"]:
            reasoning_text += f"SUBTASK:\n{reasoning_dict['subtask']}\n\n"

        if "move_reasoning" in reasoning_dict and reasoning_dict["move_reasoning"]:
            reasoning_text += f"MOVE REASONING:\n{reasoning_dict['move_reasoning']}\n\n"

        if "move" in reasoning_dict and reasoning_dict["move"]:
            reasoning_text += f"MOVE:\n{reasoning_dict['move']}\n\n"

        if "gripper_position" in reasoning_dict and reasoning_dict["gripper_position"]:
            reasoning_text += f"GRIPPER POSITION:\n{reasoning_dict['gripper_position']}\n\n"

        if "visible_objects" in reasoning_dict and reasoning_dict["visible_objects"]:
            reasoning_text += f"VISIBLE OBJECTS:\n{reasoning_dict['visible_objects']}"

        ax_reasoning.text(0.05, 0.95, reasoning_text, ha='left', va='top',
                         fontsize=10, family='monospace',
                         bbox=dict(boxstyle='round,pad=1.5', facecolor='lightyellow', alpha=0.8))

    # === right: predicted action ===
    ax_action.axis('off')

    action_text = (
        f"Predicted Action\n"
        f"{'='*25}\n\n"
        f"Position (m/s):\n"
        f"  dx: {action[0]:+.4f}\n"
        f"  dy: {action[1]:+.4f}\n"
        f"  dz: {action[2]:+.4f}\n\n"
        f"Rotation (rad/s):\n"
        f"  droll:  {action[3]:+.4f}\n"
        f"  dpitch: {action[4]:+.4f}\n"
        f"  dyaw:   {action[5]:+.4f}\n\n"
        f"Gripper:\n"
        f"  {action[6]:+.4f}\n\n"
        f"Action Norm: {np.linalg.norm(action[:6]):.4f}"
    )

    ax_action.text(0.5, 0.5, action_text, ha='center', va='center',
                  fontsize=12, family='monospace',
                  bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="visualize inference on aria test data - all 3 ablations")
    parser.add_argument("--test_json", type=str, required=True, help="test_samples.json from extract_test_samples.py")
    parser.add_argument("--base_model", type=str, default="Embodied-CoT/ecot-openvla-7b-bridge")
    parser.add_argument("--finetuned_with_gaze", type=str, help="condition a: with gaze classification")
    parser.add_argument("--finetuned_no_gaze", type=str, help="condition b: no gaze classification")
    parser.add_argument("--finetuned_speech_only", type=str, help="baseline: speech only, no llm")
    parser.add_argument("--output_dir", type=str, default="./inference_viz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--training_logs_dir", type=str, help="optional: path to training logs with reasoning")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load test data
    with open(args.test_json, 'r') as f:
        test_samples = json.load(f)

    print(f"loaded {len(test_samples)} test samples\n")

    # load models
    models = {}
    if args.base_model:
        models['base'] = {
            'path': args.base_model,
            'name': 'Base E-CoT',
            'show_class': False
        }
    if args.finetuned_with_gaze:
        models['with_gaze'] = {
            'path': args.finetuned_with_gaze,
            'name': 'Condition A (with gaze)',
            'show_class': True
        }
    if args.finetuned_no_gaze:
        models['no_gaze'] = {
            'path': args.finetuned_no_gaze,
            'name': 'Condition B (no gaze)',
            'show_class': False
        }
    if args.finetuned_speech_only:
        models['speech_only'] = {
            'path': args.finetuned_speech_only,
            'name': 'Baseline (speech only)',
            'show_class': False
        }

    # load all models
    loaded_models = {}
    for key, info in models.items():
        processor, model = load_model(info['path'], args.device)
        loaded_models[key] = {'processor': processor, 'model': model, **info}

    print("\ngenerating visualizations...")

    # process each test sample
    for i, sample in enumerate(test_samples):
        print(f"\nsample {i+1}/{len(test_samples)}: {sample['instruction'][:50]}...")

        image = Image.open(sample['image_path']).convert('RGB')
        instruction = sample['instruction']

        # choose bboxes based on classification availability
        bboxes_with_class = sample.get('bboxes_with_class', [])
        bboxes_no_class = sample.get('bboxes_no_class', [])

        # load reasoning if available (from sample or training logs)
        reasoning = sample.get('reasoning', None)
        if reasoning is None and args.training_logs_dir:
            # try to load from training logs
            training_log_path = Path(args.training_logs_dir) / f"sample_{i}" / "frame_0.json"
            if training_log_path.exists():
                with open(training_log_path, 'r') as f:
                    log_data = json.load(f)
                    reasoning = log_data.get('reasoning', None)

        # run inference for each model
        for model_key, model_data in loaded_models.items():
            processor = model_data['processor']
            model = model_data['model']
            model_name = model_data['name']
            show_class = model_data['show_class']

            # run inference
            action = run_inference(processor, model, image, instruction, args.device)

            # choose bboxes
            bboxes = bboxes_with_class if (show_class and len(bboxes_with_class) > 0) else bboxes_no_class

            # create visualization with reasoning
            fig = draw_frame_with_info(image, bboxes, action, instruction,
                                      model_name, show_classification=show_class,
                                      reasoning=reasoning)

            # save
            output_path = Path(args.output_dir) / f"sample_{i}_{model_key}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  saved {output_path}")

            # save action and reasoning
            action_path = Path(args.output_dir) / f"sample_{i}_{model_key}_action.json"
            with open(action_path, 'w') as f:
                json.dump({
                    'instruction': instruction,
                    'action': action.tolist(),
                    'model': model_key,
                    'reasoning': reasoning
                }, f, indent=2)

    print(f"\n✓ done! visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
