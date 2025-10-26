"""
process hand tracking on rgb camera with undistortion (VLA-ready).
outputs undistorted rgb video with hand landmarks + velocity overlay, plus velocity json data.

usage:
    python process_rgb_hands.py --vrs path/to/recording.vrs --mps path/to/hand_tracking_results.csv --output output_dir
"""

import argparse
import os
import numpy as np
import cv2
import pandas as pd
import json
from tqdm import tqdm

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.calibration import distort_by_calibration
from projectaria_tools.core.stream_id import StreamId

from hand_tracking_utils import (
    get_camera_calibration,
    project_3d_to_2d,
    draw_hand_skeleton,
    draw_velocity_axes,
    compute_velocity,
)


def main():
    parser = argparse.ArgumentParser(description="process hand tracking on undistorted rgb camera")
    parser.add_argument("--vrs", type=str, required=True, help="path to .vrs file")
    parser.add_argument("--mps", type=str, required=True, help="path to hand_tracking_results.csv")
    parser.add_argument("--output", type=str, default="output_rgb_hands", help="output directory")
    parser.add_argument("--frame_skip", type=int, default=1, help="process every nth frame")
    args = parser.parse_args()

    if not os.path.exists(args.vrs):
        print(f"error: vrs file not found: {args.vrs}")
        return

    if not os.path.exists(args.mps):
        print(f"error: mps file not found: {args.mps}")
        return

    print(f"\nprocessing vrs file: {args.vrs}")
    print(f"frame skip: {args.frame_skip}")

    # create data provider
    vrs_data_provider = data_provider.create_vrs_data_provider(args.vrs)
    if not vrs_data_provider:
        print("error: couldn't create vrs data provider")
        return

    # use rgb camera (214-1)
    rgb_stream_id = StreamId("214-1")
    rgb_camera_label = "camera-rgb"

    # get rgb camera calibration
    rgb_camera_calibration = get_camera_calibration(vrs_data_provider, rgb_stream_id)
    focal_lengths = rgb_camera_calibration.get_focal_lengths()
    image_size = rgb_camera_calibration.get_image_size()

    print(f"using camera: {rgb_camera_label}")
    print(f"rgb image size: {image_size}")
    print(f"focal length: {focal_lengths[0]}")

    # create pinhole (undistorted) calibration
    pinhole_calib = calibration.get_linear_camera_calibration(
        image_size[0], image_size[1], focal_lengths[0]
    )

    # get device to rgb camera transform
    device_calib = vrs_data_provider.get_device_calibration()
    T_device_camera = device_calib.get_transform_device_sensor(rgb_camera_label).inverse().to_matrix()

    # load hand tracking
    print(f"loading hand tracking from: {args.mps}")
    hand_tracking_df = pd.read_csv(args.mps)
    print(f"loaded {len(hand_tracking_df)} tracking samples")

    # setup output
    os.makedirs(args.output, exist_ok=True)

    # setup video writer
    output_video_path = os.path.join(args.output, "undistorted_rgb_with_hands.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps, (image_size[0], image_size[1])
    )
    print(f"writing video to: {output_video_path}")

    # process frames
    num_frames = vrs_data_provider.get_num_data(rgb_stream_id)
    print(f"total frames in rgb stream: {num_frames}")

    # track palm positions for velocity computation
    right_palm_positions = []
    right_palm_timestamps = []
    left_palm_positions = []
    left_palm_timestamps = []

    all_frame_data = []

    print("\nprocessing frames...")
    for frame_idx in tqdm(range(0, num_frames, args.frame_skip)):
        # get rgb image data by index
        image_data_and_record = vrs_data_provider.get_image_data_by_index(
            rgb_stream_id, frame_idx
        )

        if image_data_and_record is None:
            continue

        image = image_data_and_record[0].to_numpy_array()
        timestamp_ns = image_data_and_record[1].capture_timestamp_ns

        # undistort the frame using projectaria calibration
        # this converts fisheye -> pinhole projection
        undistorted_image = distort_by_calibration(
            image, pinhole_calib, rgb_camera_calibration
        )

        # find nearest hand tracking timestamp
        timestamp_us = timestamp_ns // 1000
        time_diffs = np.abs(hand_tracking_df["tracking_timestamp_us"] - timestamp_us)
        nearest_idx = time_diffs.argmin()
        tracking_data = hand_tracking_df.iloc[nearest_idx].to_dict()

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp_ns": timestamp_ns,
            "timestamp_us": timestamp_us,
        }

        # process right hand
        if tracking_data.get("right_tracking_confidence", -1) > 0:
            landmarks_3d = []
            landmarks_2d = []

            # extract all 21 landmarks
            for i in range(21):
                x = tracking_data.get(f"tx_right_landmark_{i}_device")
                y = tracking_data.get(f"ty_right_landmark_{i}_device")
                z = tracking_data.get(f"tz_right_landmark_{i}_device")

                if x is not None and y is not None and z is not None:
                    point_3d = np.array([x, y, z])
                    landmarks_3d.append(point_3d)

                    # project to undistorted 2d coords using pinhole calibration
                    # note: we project using pinhole_calib because the undistorted image
                    # is in pinhole projection space
                    point_2d = project_3d_to_2d(point_3d, T_device_camera, pinhole_calib)
                    landmarks_2d.append(point_2d)
                else:
                    landmarks_3d.append(None)
                    landmarks_2d.append(None)

            # track palm position (index 20) for velocity
            if landmarks_3d[20] is not None:
                right_palm_positions.append(landmarks_3d[20])
                right_palm_timestamps.append(timestamp_us)

                frame_data["right_palm_3d"] = landmarks_3d[20].tolist()
                frame_data["right_palm_confidence"] = tracking_data["right_tracking_confidence"]

            # draw skeleton on undistorted frame
            undistorted_image = draw_hand_skeleton(undistorted_image, landmarks_2d, hand_label="right")

        # process left hand
        if tracking_data.get("left_tracking_confidence", -1) > 0:
            landmarks_3d = []
            landmarks_2d = []

            for i in range(21):
                x = tracking_data.get(f"tx_left_landmark_{i}_device")
                y = tracking_data.get(f"ty_left_landmark_{i}_device")
                z = tracking_data.get(f"tz_left_landmark_{i}_device")

                if x is not None and y is not None and z is not None:
                    point_3d = np.array([x, y, z])
                    landmarks_3d.append(point_3d)

                    # project using pinhole calibration
                    point_2d = project_3d_to_2d(point_3d, T_device_camera, pinhole_calib)
                    landmarks_2d.append(point_2d)
                else:
                    landmarks_3d.append(None)
                    landmarks_2d.append(None)

            # track palm position (index 20) for velocity
            if landmarks_3d[20] is not None:
                left_palm_positions.append(landmarks_3d[20])
                left_palm_timestamps.append(timestamp_us)

                frame_data["left_palm_3d"] = landmarks_3d[20].tolist()
                frame_data["left_palm_confidence"] = tracking_data["left_tracking_confidence"]

            undistorted_image = draw_hand_skeleton(undistorted_image, landmarks_2d, hand_label="left")

        # compute and overlay velocities
        right_vel = None
        left_vel = None

        if len(right_palm_positions) >= 4:
            right_vels = compute_velocity(
                np.array(right_palm_positions),
                np.array(right_palm_timestamps),
                window_size=3
            )
            right_vel = right_vels[-1]

        if len(left_palm_positions) >= 4:
            left_vels = compute_velocity(
                np.array(left_palm_positions),
                np.array(left_palm_timestamps),
                window_size=3
            )
            left_vel = left_vels[-1]

        # draw velocity overlay
        undistorted_image = draw_velocity_axes(undistorted_image, right_vel, left_vel)

        all_frame_data.append(frame_data)

        # write frame
        video_writer.write(undistorted_image)

    # cleanup
    video_writer.release()

    # compute final velocities and save
    print("\ncomputing velocities...")

    if len(right_palm_positions) > 0:
        right_positions = np.array(right_palm_positions)
        right_timestamps = np.array(right_palm_timestamps)
        right_velocities = compute_velocity(right_positions, right_timestamps)

        print(f"right hand:")
        print(f"  tracked frames: {len(right_positions)}")
        print(f"  mean velocity magnitude: {np.linalg.norm(right_velocities, axis=1).mean():.4f} m/s")
        print(f"  max velocity magnitude: {np.linalg.norm(right_velocities, axis=1).max():.4f} m/s")

        right_velocity_data = {
            "timestamps_us": right_timestamps.tolist(),
            "positions_3d_m": right_positions.tolist(),
            "velocities_3d_ms": right_velocities.tolist(),
            "velocity_magnitudes_ms": np.linalg.norm(right_velocities, axis=1).tolist(),
        }
        with open(os.path.join(args.output, "right_hand_velocity.json"), "w") as f:
            json.dump(right_velocity_data, f, indent=2)

    if len(left_palm_positions) > 0:
        left_positions = np.array(left_palm_positions)
        left_timestamps = np.array(left_palm_timestamps)
        left_velocities = compute_velocity(left_positions, left_timestamps)

        print(f"left hand:")
        print(f"  tracked frames: {len(left_positions)}")
        print(f"  mean velocity magnitude: {np.linalg.norm(left_velocities, axis=1).mean():.4f} m/s")
        print(f"  max velocity magnitude: {np.linalg.norm(left_velocities, axis=1).max():.4f} m/s")

        left_velocity_data = {
            "timestamps_us": left_timestamps.tolist(),
            "positions_3d_m": left_positions.tolist(),
            "velocities_3d_ms": left_velocities.tolist(),
            "velocity_magnitudes_ms": np.linalg.norm(left_velocities, axis=1).tolist(),
        }
        with open(os.path.join(args.output, "left_hand_velocity.json"), "w") as f:
            json.dump(left_velocity_data, f, indent=2)

    with open(os.path.join(args.output, "all_frames.json"), "w") as f:
        json.dump(all_frame_data, f, indent=2)

    print(f"\noutput saved to: {args.output}")
    print(f"  video: {output_video_path}")
    if len(right_palm_positions) > 0:
        print(f"  right hand velocity: right_hand_velocity.json")
    if len(left_palm_positions) > 0:
        print(f"  left hand velocity: left_hand_velocity.json")
    print(f"  all frames data: all_frames.json")
    print(f"\ndone!")


if __name__ == "__main__":
    main()
