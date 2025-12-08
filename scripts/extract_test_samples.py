#!/usr/bin/env python3
"""
extract test samples from rlds dataset for visualization
creates json with image paths, instructions, bboxes, and gaze data
"""

import argparse
import json
import numpy as np
from PIL import Image
import tensorflow_datasets as tfds
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dir", type=str, required=True, help="path to rlds dataset directory")
    parser.add_argument("--output_json", type=str, default="test_samples.json")
    parser.add_argument("--num_samples", type=int, default=5, help="number of test samples to extract")
    parser.add_argument("--output_images_dir", type=str, default="test_images")
    args = parser.parse_args()

    Path(args.output_images_dir).mkdir(parents=True, exist_ok=True)

    # load validation split
    print(f"loading dataset from {args.rlds_dir}...")
    builder = tfds.builder_from_directory(args.rlds_dir)
    ds = builder.as_dataset(split='val')

    test_samples = []
    sample_idx = 0

    for episode_idx, episode in enumerate(ds.take(args.num_samples)):
        steps = episode['steps']

        # pick middle frame from episode
        num_steps = len(list(steps))
        middle_idx = num_steps // 2

        for step_idx, step in enumerate(steps):
            if step_idx != middle_idx:
                continue

            # extract data
            image = step['observation']['image_0'].numpy()
            instruction = step['language_instruction'].numpy().decode('utf-8')
            state = step['observation']['state'].numpy()  # (2, 7) bimanual
            action = step['action'].numpy()  # (2, 7) bimanual

            # check if bbox data exists
            has_bboxes = 'classified_bboxes' in step['observation']
            bboxes_with_class = []
            bboxes_no_class = []

            if has_bboxes:
                # parse bbox string (stored as tf.string in transform)
                bbox_str = step['observation']['classified_bboxes'].numpy().decode('utf-8')
                try:
                    # bbox_str is string representation of list of dicts
                    import ast
                    bboxes_list = ast.literal_eval(bbox_str)
                    # bboxes_list is list of frame data, get current frame
                    if isinstance(bboxes_list, list) and len(bboxes_list) > step_idx:
                        frame_bboxes = bboxes_list[step_idx]
                        # frame_bboxes is dict with PRIMARY, GAZE_FOCUS, AUXILIARY keys
                        for category, bbox_list in frame_bboxes.items():
                            for bbox in bbox_list:
                                bbox_with_cat = bbox.copy()
                                bbox_with_cat['category'] = category
                                bboxes_with_class.append(bbox_with_cat)
                                # also add to no_class list (without category)
                                bboxes_no_class.append({
                                    'bbox': bbox['bbox'],
                                    'label': bbox['label'],
                                    'confidence': bbox['confidence']
                                })
                except Exception as e:
                    print(f"warning: could not parse bboxes: {e}")

            # save image
            image_path = f"{args.output_images_dir}/sample_{sample_idx}.png"
            Image.fromarray(image).save(image_path)

            # create sample entry
            sample = {
                'episode_idx': episode_idx,
                'step_idx': step_idx,
                'image_path': image_path,
                'instruction': instruction,
                'state': state.tolist(),  # (2, 7)
                'action': action.tolist(),  # (2, 7)
                'bboxes_with_class': bboxes_with_class,
                'bboxes_no_class': bboxes_no_class,
                'has_classification': len(bboxes_with_class) > 0
            }

            test_samples.append(sample)
            sample_idx += 1
            print(f"extracted sample {sample_idx}: {instruction[:50]}...")

    # save json
    with open(args.output_json, 'w') as f:
        json.dump(test_samples, f, indent=2)

    print(f"\nextracted {len(test_samples)} test samples")
    print(f"saved to: {args.output_json}")
    print(f"images saved to: {args.output_images_dir}/")


if __name__ == "__main__":
    main()
