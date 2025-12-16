"""
full E-CoT preprocessing pipeline for aria dataset.

ONE RUN generates ALL ablation conditions:
- condition a: bboxes WITH gaze classification + LLM reasoning
- condition b: bboxes WITHOUT gaze classification + LLM reasoning
- baseline: bboxes WITH gaze classification + NO LLM (speech + move primitives only)

usage:
    python scripts/preprocess_aria_ecot.py \
        --ogvids_dir ./ogvids \
        --tfrecord_dir ./aria_dataset \
        --api gemini \
        --api_key YOUR_KEY \
        --output_prefix aria_reasonings

outputs (all in one run):
    - aria_reasonings_with_gaze.json (condition a)
    - aria_reasonings_no_gaze.json (condition b)
    - aria_reasonings_speech_only.json (baseline)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

# tensorflow
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

try:
    import tensorflow_datasets as tfds
    HAS_TFDS = True
except ImportError:
    HAS_TFDS = False

# apis
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# bbox detection
try:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    HAS_BBOX = True
except ImportError:
    HAS_BBOX = False

# aria tools
try:
    from projectaria_tools.core import data_provider, calibration
    from projectaria_tools.core.calibration import distort_by_calibration
    from projectaria_tools.core.stream_id import StreamId
    HAS_ARIA = True
except ImportError:
    HAS_ARIA = False


class GeminiClient:
    def __init__(self, api_key: str):
        if not HAS_GEMINI:
            raise ImportError("pip install google-generativeai")
        genai.configure(api_key=api_key)
        # gemini-3-pro-preview is the strongest model
        self.model = genai.GenerativeModel("gemini-3-pro-preview")

    def generate(self, prompt: str, verbose: bool = True) -> Optional[str]:
        if verbose:
            print(f"\n{'='*80}")
            print(f"GEMINI PROMPT:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")

        for attempt in range(5):
            try:
                # use low thinking for faster/cheaper responses
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": 1.0}
                )

                if verbose:
                    print(f"\n{'='*80}")
                    print(f"GEMINI RESPONSE:")
                    print(f"{'='*80}")
                    print(response.text)
                    print(f"{'='*80}\n")

                return response.text
            except Exception as e:
                if "ResourceExhausted" in str(type(e)):
                    print(f"  rate limited, waiting...")
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"  error: {e}")
                    time.sleep(2)
        return None


class OpenAIClient:
    def __init__(self, api_key: str):
        if not HAS_OPENAI:
            raise ImportError("pip install openai")
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, verbose: bool = True) -> Optional[str]:
        if verbose:
            print(f"\n{'='*80}")
            print(f"OPENAI PROMPT:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            response_text = response.choices[0].message.content

            if verbose:
                print(f"\n{'='*80}")
                print(f"OPENAI RESPONSE:")
                print(f"{'='*80}")
                print(response_text)
                print(f"{'='*80}\n")

            return response_text
        except Exception as e:
            print(f"  openai error: {e}")
            return None


class BboxGenerator:
    """generates bboxes using grounding dino."""

    def __init__(self, device: str = "cuda:0"):
        if not HAS_BBOX:
            raise ImportError("pip install torch transformers")

        print(f"loading grounding dino on {device}...")
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(
            model_id, size={"shortest_edge": 256, "longest_edge": 256}
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        print("done.")

    def detect(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.2,
    ) -> List[Dict]:
        """detect objects in image."""
        pil_image = Image.fromarray(image)
        inputs = self.processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        bboxes = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"],
            results["boxes"].cpu().numpy(),
        ):
            bboxes.append({
                "confidence": float(score),
                "label": label,
                "bbox": list(box.astype(int)),
            })

        return bboxes


def classify_move_from_velocity(action: np.ndarray, state: np.ndarray, threshold: float = 0.01) -> Tuple[str, Dict]:
    """classify movement primitive from velocity/action data."""
    if action.shape == (2, 7):
        vel = action[1, :3]
        rot_vel = action[1, 3:6]
        gripper_change = action[1, 6]
    else:
        vel = action[:3] if len(action) >= 3 else np.zeros(3)
        rot_vel = action[3:6] if len(action) >= 6 else np.zeros(3)
        gripper_change = action[6] if len(action) >= 7 else 0

    velocity_info = {
        "vel_x": float(vel[0]),
        "vel_y": float(vel[1]),
        "vel_z": float(vel[2]),
        "vel_magnitude": float(np.linalg.norm(vel)),
        "gripper_change": float(gripper_change),
    }

    if abs(gripper_change) > 0.5:
        return ("closing gripper" if gripper_change > 0 else "opening gripper"), velocity_info

    if np.linalg.norm(vel) < threshold:
        return "holding position", velocity_info

    abs_vel = np.abs(vel)
    dominant_axis = np.argmax(abs_vel)
    direction = np.sign(vel[dominant_axis])

    move_map = {
        (0, 1): "moving forward", (0, -1): "moving backward",
        (1, 1): "moving left", (1, -1): "moving right",
        (2, 1): "moving up", (2, -1): "moving down",
    }

    return f"right hand {move_map.get((dominant_axis, direction), 'moving')}", velocity_info


def build_reasoning_prompt(
    task_description: str,
    subtask: str,
    proposed_move: str,
    velocity_info: Dict,
    state: np.ndarray,
    bboxes: Optional[List[Dict]],
    gaze_point: Optional[List[int]],
    step_idx: int,
    total_steps: int,
) -> str:
    """build prompt for llm reasoning generation."""

    # state description
    if state.shape == (2, 7):
        left_pos, right_pos = state[0, :3], state[1, :3]
        left_grip = "open" if state[0, 6] > 0.5 else "closed"
        right_grip = "open" if state[1, 6] > 0.5 else "closed"
        state_desc = f"left hand: ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f}), gripper={left_grip}\nright hand: ({right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f}), gripper={right_grip}"
    else:
        state_desc = f"state: {state.tolist()}"

    # bbox description
    if bboxes:
        bbox_desc = "visible objects:\n"
        for i, bb in enumerate(bboxes[:5]):  # limit to 5
            gaze_marker = " [LOOKING AT]" if bb.get("is_gaze_target") else ""
            bbox_desc += f"  - {bb['label']} at {bb['bbox']}{gaze_marker}\n"
    else:
        bbox_desc = "visible objects: not available"

    # gaze description
    gaze_desc = f"gaze point: ({gaze_point[0]}, {gaze_point[1]})" if gaze_point else "gaze: not available"

    return f"""annotate this robot manipulation step with chain-of-thought reasoning.

## context
- task: {task_description}
- spoken instruction: "{subtask}"
- step {step_idx + 1}/{total_steps}

## state
{state_desc}

## {bbox_desc}
## {gaze_desc}

## velocity
magnitude: {velocity_info['vel_magnitude']:.4f}, gripper_change: {velocity_info['gripper_change']:.2f}

## proposed move (from velocity): "{proposed_move}"

respond with EXACTLY this format:

<task>{task_description} (what remains)</task>
<plan>2-3 remaining high-level steps</plan>
<subtask_reason>why is "{subtask}" the focus now?</subtask_reason>
<subtask>{subtask}</subtask>
<move_reason>why this movement?</move_reason>
<move>corrected move if needed, or "{proposed_move}"</move>

be concise. 1-2 sentences per field."""


def parse_reasoning_response(response: str, defaults: Dict[str, str]) -> Dict[str, str]:
    """extract reasoning fields from llm response."""
    tags = ["task", "plan", "subtask_reason", "subtask", "move_reason", "move"]
    result = {}
    for tag in tags:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL | re.IGNORECASE)
        result[tag] = match.group(1).strip() if match else defaults.get(tag, "")
    return result


def convert_numpy_types(obj):
    """convert numpy types to python native types for json serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def load_tfrecord_episodes(tfrecord_dir: str, split: str = "train") -> List[Dict]:
    """load episodes from tfrecords."""
    if not HAS_TFDS:
        raise ImportError("pip install tensorflow-datasets")

    print(f"loading dataset from {tfrecord_dir}")
    # tfds expects data_dir to be parent of dataset folder (which contains version subdir)
    # if tfrecord_dir is ./aria_dataset, data_dir should be . so it finds ./aria_dataset/1.0.0/
    parent_dir = os.path.dirname(os.path.abspath(tfrecord_dir)) if os.path.basename(tfrecord_dir) == "aria_dataset" else "."
    builder = tfds.builder("aria_dataset", data_dir=parent_dir)
    ds = builder.as_dataset(split=split)

    episodes = []
    for idx, episode in enumerate(ds):
        steps = []
        file_path = episode["episode_metadata"]["file_path"].numpy().decode() if "file_path" in episode["episode_metadata"] else f"episode_{idx}"
        episode_id = int(episode["episode_metadata"]["episode_id"].numpy()) if "episode_id" in episode["episode_metadata"] else idx
        task_desc = episode["episode_metadata"]["task_description"].numpy().decode() if "task_description" in episode["episode_metadata"] else ""

        for step in episode["steps"]:
            steps.append({
                "language_instruction": step["language_instruction"].numpy().decode() if "language_instruction" in step else "",
                "state": step["observation"]["state"].numpy() if "state" in step["observation"] else np.zeros((2, 7)),
                "action": step["action"].numpy() if "action" in step else np.zeros((2, 7)),
                "image": step["observation"]["image_0"].numpy() if "image_0" in step["observation"] else None,
            })

        episodes.append({
            "file_path": file_path,
            "episode_id": episode_id,
            "task_description": task_desc,
            "steps": steps,
        })

        if idx % 10 == 0:
            print(f"  loaded episode {idx}: {len(steps)} steps")

    print(f"loaded {len(episodes)} episodes")
    return episodes


def load_gaze_data(mps_path: str) -> Optional[Dict]:
    """load gaze data from MPS output."""
    gaze_file = os.path.join(mps_path, "eye_gaze", "general_eye_gaze.csv")
    if not os.path.exists(gaze_file):
        return None

    import csv
    gaze_data = {}
    with open(gaze_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row.get("tracking_timestamp_us", 0)) * 1000  # convert to ns
            if "gaze_point_x" in row and "gaze_point_y" in row:
                gaze_data[ts] = [float(row["gaze_point_x"]), float(row["gaze_point_y"])]
    return gaze_data


def process_episode_with_bboxes_both_versions(
    episode: Dict,
    bbox_generator: Optional[BboxGenerator],
    gaze_data: Optional[Dict],
    text_prompt: str,
    gaze_threshold: float = 100.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    process episode to extract bboxes per step.
    returns TWO versions: (with_gaze, no_gaze)
    runs detection ONCE per frame to save compute.
    """
    step_bboxes_with_gaze = []
    step_bboxes_no_gaze = []

    for step_idx, step in enumerate(episode["steps"]):
        gaze_point = None
        image = step.get("image")

        # detect objects once
        raw_bboxes = []
        if bbox_generator and image is not None:
            raw_bboxes = bbox_generator.detect(image, text_prompt)

        # get gaze for this timestamp
        if gaze_data:
            gaze_point = list(gaze_data.values())[min(step_idx, len(gaze_data) - 1)] if gaze_data else None

        # version 1: WITH gaze classification
        bboxes_with_gaze = []
        for bb in raw_bboxes:
            bb_copy = bb.copy()
            if gaze_point:
                box = bb_copy["bbox"]
                box_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                dist = np.sqrt((box_center[0] - gaze_point[0])**2 + (box_center[1] - gaze_point[1])**2)
                bb_copy["gaze_distance"] = float(dist)
            bboxes_with_gaze.append(bb_copy)

        # mark gaze target for version 1
        if gaze_point and bboxes_with_gaze:
            min_idx = min(range(len(bboxes_with_gaze)), key=lambda i: bboxes_with_gaze[i].get("gaze_distance", float("inf")))
            if bboxes_with_gaze[min_idx].get("gaze_distance", float("inf")) < gaze_threshold:
                bboxes_with_gaze[min_idx]["is_gaze_target"] = True

        # version 2: WITHOUT gaze classification (just raw detections)
        bboxes_no_gaze = [bb.copy() for bb in raw_bboxes]

        step_bboxes_with_gaze.append({
            "bboxes": bboxes_with_gaze,
            "gaze_point": [int(gaze_point[0]), int(gaze_point[1])] if gaze_point else None,
        })

        step_bboxes_no_gaze.append({
            "bboxes": bboxes_no_gaze,
            "gaze_point": None,  # no gaze for this version
        })

    return step_bboxes_with_gaze, step_bboxes_no_gaze


def generate_reasoning_for_episode(
    episode: Dict,
    step_bboxes: List[Dict],
    client,
    use_llm: bool = True,
) -> Dict:
    """generate reasoning for all steps in episode."""
    task_desc = episode["task_description"]
    steps = episode["steps"]
    total_steps = len(steps)

    reasonings = {}
    features = {
        "gripper_position": [],
        "state_3d": [],
        "move_primitive": [],
        "bboxes": [],
    }

    for step_idx, step in enumerate(steps):
        subtask = step.get("language_instruction", task_desc) or task_desc
        state = step.get("state", np.zeros((2, 7)))
        action = step.get("action", np.zeros((2, 7)))

        # classify move
        proposed_move, velocity_info = classify_move_from_velocity(action, state)

        # get bboxes for this step
        step_bbox_data = step_bboxes[step_idx] if step_idx < len(step_bboxes) else {}
        bboxes = step_bbox_data.get("bboxes", [])
        gaze_point = step_bbox_data.get("gaze_point")

        # defaults
        defaults = {
            "task": task_desc,
            "plan": "complete the task",
            "subtask_reason": "following instruction",
            "subtask": subtask,
            "move_reason": "executing trajectory",
            "move": proposed_move,
        }

        if use_llm and client:
            prompt = build_reasoning_prompt(
                task_description=task_desc,
                subtask=subtask,
                proposed_move=proposed_move,
                velocity_info=velocity_info,
                state=state,
                bboxes=bboxes,
                gaze_point=gaze_point,
                step_idx=step_idx,
                total_steps=total_steps,
            )

            if step_idx % 20 == 0:
                print(f"    step {step_idx}/{total_steps}...")

            response = client.generate(prompt)
            reasoning = parse_reasoning_response(response, defaults) if response else defaults
            time.sleep(0.3)
        else:
            reasoning = defaults

        reasonings[str(step_idx)] = reasoning

        # features
        if state.shape == (2, 7):
            features["state_3d"].append(state[1, :3].tolist())
            features["gripper_position"].append([int(state[1, 0] * 100 + 128), int(state[1, 1] * 100 + 128)])
        features["move_primitive"].append(reasoning["move"])

        # format bboxes for E-CoT: [(confidence, label, [x1,y1,x2,y2]), ...]
        bbox_tuples = [(bb["confidence"], bb["label"], bb["bbox"]) for bb in bboxes]
        features["bboxes"].append(bbox_tuples)

    return {
        "reasoning": reasonings,
        "features": features,
        "metadata": {
            "file_path": episode["file_path"],
            "episode_id": episode["episode_id"],
            "task_description": task_desc,
            "n_steps": total_steps,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="E-CoT preprocessing - generates all ablations in one run")
    parser.add_argument("--ogvids_dir", type=str, default=None, help="path to ogvids with .vrs files")
    parser.add_argument("--tfrecord_dir", type=str, required=True, help="path to aria_dataset tfrecords")
    parser.add_argument("--api", type=str, choices=["gemini", "openai"], default="gemini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--output_prefix", type=str, default="aria_reasonings", help="prefix for output files")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # output files
    output_with_gaze = f"{args.output_prefix}_with_gaze.json"
    output_no_gaze = f"{args.output_prefix}_no_gaze.json"
    output_speech_only = f"{args.output_prefix}_speech_only.json"

    # config summary
    print("=" * 60)
    print("E-CoT Preprocessing - ALL Ablations in ONE Run")
    print("=" * 60)
    print(f"tfrecord_dir: {args.tfrecord_dir}")
    print(f"ogvids_dir: {args.ogvids_dir}")
    print(f"outputs:")
    print(f"  1. {output_with_gaze} (bboxes + gaze + llm)")
    print(f"  2. {output_no_gaze} (bboxes + no gaze + llm)")
    print(f"  3. {output_speech_only} (bboxes + gaze + no llm)")
    print(f"llm: {args.api}")
    print("=" * 60)

    # init client
    client = None
    if not args.dry_run:
        if args.api_key is None:
            print("error: --api_key required (unless --dry_run)")
            return
        client = GeminiClient(args.api_key) if args.api == "gemini" else OpenAIClient(args.api_key)

    # init bbox generator
    bbox_generator = None
    if not args.dry_run:
        if HAS_BBOX:
            bbox_generator = BboxGenerator(device=f"cuda:{args.gpu}")
        else:
            print("warning: bbox dependencies not available, skipping bboxes")

    # load existing for all three outputs
    reasonings_with_gaze = {}
    reasonings_no_gaze = {}
    reasonings_speech_only = {}

    if args.resume:
        if os.path.exists(output_with_gaze):
            with open(output_with_gaze, "r") as f:
                reasonings_with_gaze = json.load(f)
            print(f"resumed {len(reasonings_with_gaze)} entries (with_gaze)")
        if os.path.exists(output_no_gaze):
            with open(output_no_gaze, "r") as f:
                reasonings_no_gaze = json.load(f)
            print(f"resumed {len(reasonings_no_gaze)} entries (no_gaze)")
        if os.path.exists(output_speech_only):
            with open(output_speech_only, "r") as f:
                reasonings_speech_only = json.load(f)
            print(f"resumed {len(reasonings_speech_only)} entries (speech_only)")

    # load episodes
    episodes = load_tfrecord_episodes(args.tfrecord_dir, args.split)

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    print(f"\nprocessing {len(episodes)} episodes...")

    for i, episode in enumerate(episodes):
        episode_key = episode["file_path"]

        # check if already processed in all three outputs
        if args.resume and (episode_key in reasonings_with_gaze and
                            episode_key in reasonings_no_gaze and
                            episode_key in reasonings_speech_only):
            print(f"[{i+1}/{len(episodes)}] skipping {episode_key} (exists in all outputs)")
            continue

        print(f"[{i+1}/{len(episodes)}] {episode_key}")

        try:
            # load gaze data if available
            gaze_data = None
            # extract recording name from path (e.g., Orange_v1, Banana_v1, etc.)
            # look for pattern like "Banana_v1", "Orange_v1", "Sponge_v1", "Bottle_v2" in the path
            import re
            match = re.search(r'(Banana|Orange|Sponge|Bottle)_v\d+', episode_key)
            recording_name = match.group(0) if match else None

            # try ogvids_dir first (legacy)
            if args.ogvids_dir and os.path.exists(args.ogvids_dir):
                for folder in os.listdir(args.ogvids_dir):
                    if recording_name in folder:
                        mps_path = os.path.join(args.ogvids_dir, folder, f"mps_{folder}_vrs")
                        if os.path.exists(mps_path):
                            gaze_data = load_gaze_data(mps_path)
                        break

            # fallback: look for csv directly in tfrecord_dir
            if not gaze_data and recording_name:
                gaze_csv = os.path.join(args.tfrecord_dir, f"{recording_name}_gaze.csv")
                if os.path.exists(gaze_csv):
                    import csv
                    gaze_data = {}
                    with open(gaze_csv, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            ts = int(row.get("tracking_timestamp_us", 0)) * 1000
                            if "gaze_point_x" in row and "gaze_point_y" in row:
                                gaze_data[ts] = [float(row["gaze_point_x"]), float(row["gaze_point_y"])]
                    print(f"  loaded gaze from {recording_name}_gaze.csv: {len(gaze_data)} points")

            if not gaze_data:
                print(f"  no gaze data found for {recording_name or 'unknown recording'}")

            text_prompt = episode["task_description"] + ". hand. table."

            # generate bboxes - runs detection once, returns both versions
            print("  generating bboxes (both versions)...")
            bboxes_with_gaze, bboxes_no_gaze = process_episode_with_bboxes_both_versions(
                episode,
                bbox_generator if not args.dry_run else None,
                gaze_data,
                text_prompt,
            )

            if args.dry_run:
                print(f"  {len(episode['steps'])} steps, task: {episode['task_description'][:50]}...")
                continue

            # generate reasoning for all three conditions
            print("  generating reasoning (with gaze + llm)...")
            result_with_gaze = generate_reasoning_for_episode(
                episode,
                bboxes_with_gaze,
                client,
                use_llm=True,
            )

            print("  generating reasoning (no gaze + llm)...")
            result_no_gaze = generate_reasoning_for_episode(
                episode,
                bboxes_no_gaze,
                client,
                use_llm=True,
            )

            print("  generating reasoning (speech only, no llm)...")
            result_speech_only = generate_reasoning_for_episode(
                episode,
                bboxes_with_gaze,  # use with_gaze bboxes for baseline
                client,
                use_llm=False,
            )

            # store in all three outputs
            episode_id = str(episode["episode_id"])
            reasonings_with_gaze[episode_key] = {episode_id: result_with_gaze}
            reasonings_no_gaze[episode_key] = {episode_id: result_no_gaze}
            reasonings_speech_only[episode_key] = {episode_id: result_speech_only}

            # save all three files (convert numpy types to native python)
            with open(output_with_gaze, "w") as f:
                json.dump(convert_numpy_types(reasonings_with_gaze), f, indent=2)
            with open(output_no_gaze, "w") as f:
                json.dump(convert_numpy_types(reasonings_no_gaze), f, indent=2)
            with open(output_speech_only, "w") as f:
                json.dump(convert_numpy_types(reasonings_speech_only), f, indent=2)

            print(f"  saved {len(result_with_gaze['reasoning'])} steps to all 3 outputs")

        except Exception as e:
            print(f"  error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\ndone!")
    print(f"outputs:")
    print(f"  1. {output_with_gaze} ({len(reasonings_with_gaze)} episodes)")
    print(f"  2. {output_no_gaze} ({len(reasonings_no_gaze)} episodes)")
    print(f"  3. {output_speech_only} ({len(reasonings_speech_only)} episodes)")
    print(f"\nfor training (condition a - with gaze):")
    print(f"  export REASONING_DATASET_PATH={os.path.abspath(output_with_gaze)}")
    print(f"\nfor training (condition b - no gaze):")
    print(f"  export REASONING_DATASET_PATH={os.path.abspath(output_no_gaze)}")


if __name__ == "__main__":
    main()
