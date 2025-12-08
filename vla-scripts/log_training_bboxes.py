#!/usr/bin/env python3
"""
callback to log bbox detections and gaze targets during training
saves visualizations showing what the model sees
includes full embodied-cot reasoning chain
"""

import os
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import sys

# add prismatic to path for cot utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from prismatic.util.cot_utils import CotTag


class BboxLoggingCallback:
    """logs bbox detections and full reasoning chain during training"""

    def __init__(self, log_dir="training_logs/bboxes", log_frequency=100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency
        self.step_count = 0

    def _parse_reasoning(self, reasoning_str):
        """parse reasoning string into dict of components"""
        if reasoning_str is None:
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

    def __call__(self, batch, trajectory_info=None):
        """called during training to log bbox and reasoning info"""
        self.step_count += 1

        # only log every N steps
        if self.step_count % self.log_frequency != 0:
            return

        # extract bbox info from batch
        if "classified_bboxes" not in batch.get("observation", {}):
            return

        step_dir = self.log_dir / f"step_{self.step_count}"
        step_dir.mkdir(exist_ok=True)

        # log each item in batch
        batch_size = len(batch["observation"]["image_primary"])
        for i in range(min(batch_size, 4)):  # log max 4 per batch
            # extract reasoning if available
            reasoning = None
            if "reasoning" in batch:
                reasoning = batch["reasoning"][i] if hasattr(batch["reasoning"], '__getitem__') else batch["reasoning"]

            self._log_single_frame(
                batch["observation"]["image_primary"][i],
                batch["observation"]["classified_bboxes"][i],
                batch.get("language_instruction", [None])[i],
                reasoning,
                step_dir / f"frame_{i}.png",
                step_dir / f"frame_{i}.json"
            )

    def _log_single_frame(self, image, bboxes_str, instruction, reasoning, img_path, json_path):
        """log single frame with bboxes and reasoning chain"""
        # parse bbox string
        import ast
        try:
            bboxes = ast.literal_eval(bboxes_str.decode() if isinstance(bboxes_str, bytes) else bboxes_str)
        except:
            return

        # create visualization
        img = Image.fromarray(image.numpy() if hasattr(image, 'numpy') else image)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()

        gaze_target = None
        all_objects = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['bbox']
            label = bbox['label']
            is_gaze = bbox.get('is_gaze_target', False)

            # color based on whether it's the gaze target
            color = (255, 0, 0) if is_gaze else (100, 100, 100)

            # draw bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3 if is_gaze else 2)

            # label
            text = f"{label}" + (" [GAZE]" if is_gaze else "")
            draw.text((x1, y1-20), text, fill=color, font=font)

            if is_gaze:
                gaze_target = label
            all_objects.append(label)

        # save image
        img.save(img_path)

        # parse reasoning
        reasoning_dict = self._parse_reasoning(reasoning)

        # save json
        log_data = {
            "step": self.step_count,
            "instruction": instruction.decode() if isinstance(instruction, bytes) else instruction,
            "all_objects": all_objects,
            "gaze_target": gaze_target,
            "num_objects": len(all_objects),
            # embodied-cot reasoning chain
            "reasoning": {
                "task": reasoning_dict.get("task", ""),
                "plan": reasoning_dict.get("plan", ""),
                "subtask_reasoning": reasoning_dict.get("subtask_reasoning", ""),
                "subtask": reasoning_dict.get("subtask", ""),
                "move_reasoning": reasoning_dict.get("move_reasoning", ""),
                "move": reasoning_dict.get("move", ""),
                "gripper_position": reasoning_dict.get("gripper_position", ""),
                "visible_objects": reasoning_dict.get("visible_objects", "")
            }
        }
        with open(json_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def add_bbox_logging_to_training(trainer, log_dir="training_logs/bboxes"):
    """
    add bbox logging callback to training loop

    usage in finetune.py:
        from log_training_bboxes import add_bbox_logging_to_training
        add_bbox_logging_to_training(trainer)
    """
    callback = BboxLoggingCallback(log_dir)
    # this would be integrated into the training loop
    # for now, document how to use it
    return callback
