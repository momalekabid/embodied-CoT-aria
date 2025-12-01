"""
generate bounding boxes with gaze-aware classification for aria recordings.

this script:
1. loads aria vrs files with hand tracking and gaze data
2. runs object detection (grounding dino) on each frame
3. classifies each bbox as PRIMARY (actively looking at) or SECONDARY (context)
4. stores gaze point (2d pixel coords) per frame
5. outputs json with categorized bboxes + gaze points
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))
from gaze_utils import GazeMPSLoader
from gaze_guided_bbox_filter import classify_bbox_by_gaze
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.calibration import distort_by_calibration
from projectaria_tools.core.stream_id import StreamId


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs", type=str, required=True, help="path to .vrs file")
    parser.add_argument("--mps", type=str, required=True, help="path to hand_tracking_results.csv")
    parser.add_argument("--mps_base", type=str, default=None, help="path to mps base folder (for gaze)")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt for object detection (e.g., 'bottle. hand. cup.')")
    parser.add_argument("--output", type=str, default="bboxes_with_gaze.json", help="output json path")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--frame_skip", type=int, default=1, help="process every nth frame")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    parser.add_argument("--primary_threshold", type=float, default=100.0, help="max pixel distance for primary classification")
    args = parser.parse_args()

    # load vrs data
    print(f"\nloading vrs file: {args.vrs}")
    vrs_data_provider = data_provider.create_vrs_data_provider(args.vrs)
    if not vrs_data_provider:
        print("error: couldn't create vrs data provider")
        return

    # setup camera
    rgb_stream_id = StreamId("214-1")
    rgb_camera_label = "camera-rgb"
    rgb_camera_calibration = vrs_data_provider.get_device_calibration().get_camera_calib(rgb_camera_label)
    focal_lengths = rgb_camera_calibration.get_focal_lengths()
    image_size = rgb_camera_calibration.get_image_size()

    print(f"rgb image size: {image_size}")

    # create pinhole (undistorted) calibration
    pinhole_calib = calibration.get_linear_camera_calibration(image_size[0], image_size[1], focal_lengths[0])

    # get device calibration
    device_calib = vrs_data_provider.get_device_calibration()

    # load gaze data
    if args.mps_base is None:
        mps_base = os.path.dirname(os.path.dirname(args.mps))
    else:
        mps_base = args.mps_base

    print(f"loading gaze from: {mps_base}")
    try:
        gaze_loader = GazeMPSLoader(mps_base, vrs_data_provider, use_general_gaze=True)
    except Exception as e:
        print(f"error loading gaze: {e}")
        return

    # load grounding dino
    print(f"loading grounding dino on gpu:{args.gpu}...")
    model_id = "IDEA-Research/grounding-dino-base"
    device = f"cuda:{args.gpu}"
    processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print("done.")

    # process frames
    num_frames = vrs_data_provider.get_num_data(rgb_stream_id)
    print(f"\nprocessing {num_frames} frames...")

    results = {
        "vrs_path": args.vrs,
        "text_prompt": args.text_prompt,
        "frames": []
    }

    for frame_idx in tqdm(range(0, num_frames, args.frame_skip)):
        image_data_and_record = vrs_data_provider.get_image_data_by_index(rgb_stream_id, frame_idx)

        if image_data_and_record is None:
            continue

        image = image_data_and_record[0].to_numpy_array()
        timestamp_ns = image_data_and_record[1].capture_timestamp_ns

        # undistort frame
        undistorted_image = distort_by_calibration(image, pinhole_calib, rgb_camera_calibration)
        undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)

        # rotate image (match process_fisheye output)
        undistorted_image = cv2.rotate(undistorted_image, cv2.ROTATE_90_CLOCKWISE)

        # get gaze projection
        gaze_point = None
        try:
            gaze_projection = gaze_loader.get_gaze_projection(
                timestamp_ns,
                rgb_camera_label,
                device_calib,
                pinhole_calib,
                depth_m=1.0
            )
            if gaze_projection is not None:
                gaze_point = gaze_projection  # (gx, gy)
        except:
            pass  # skip if gaze unavailable

        # run object detection
        pil_image = Image.fromarray(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
        inputs = processor(
            images=pil_image,
            text=args.text_prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        detection_results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]

        logits = detection_results["scores"].cpu().numpy()
        labels = detection_results["labels"]
        boxes = detection_results["boxes"].cpu().numpy()

        # classify each bbox as PRIMARY (actively looking at) or SECONDARY (context)
        categorized_bboxes = {
            "PRIMARY": [],
            "SECONDARY": []
        }

        for lg, label, box in zip(logits, labels, boxes):
            box_int = list(box.astype(int))
            confidence = round(float(lg), 5)

            bbox_dict = {
                "confidence": confidence,
                "label": label,
                "bbox": box_int
            }

            # classify using gaze proximity
            if gaze_point is not None:
                category, gaze_dist = classify_bbox_by_gaze(
                    bbox_dict,
                    gaze_point,
                    primary_threshold=args.primary_threshold
                )
            else:
                # no gaze data, all objects are secondary
                category = "SECONDARY"
                gaze_dist = None

            bbox_entry = {
                "confidence": confidence,
                "label": label,
                "box": box_int,
                "gaze_distance": float(gaze_dist) if gaze_dist is not None else None
            }

            categorized_bboxes[category].append(bbox_entry)

        # store frame results
        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_ns": int(timestamp_ns),
            "gaze_point": [int(gaze_point[0]), int(gaze_point[1])] if gaze_point is not None else None,
            "bboxes": categorized_bboxes
        }

        results["frames"].append(frame_data)

    # save results
    print(f"\nsaving results to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"done! processed {len(results['frames'])} frames")


if __name__ == "__main__":
    main()
