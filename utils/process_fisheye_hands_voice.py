"""
process hand tracking and project onto the recording from aria rgb camera after undistorting fisheye
outputs undistorted rgb video with hand landmarks + velocity overlay, plus velocity json data.

usage:
    python process_fisheye_hands.py --vrs path/to/recording.vrs --mps path/to/hand_tracking_results.csv --output output_dir
"""

import argparse
import os
import numpy as np
import cv2
import pandas as pd
import json
from tqdm import tqdm
#import whisper
from faster_whisper import WhisperModel
import shutil
import textwrap

import tempfile
from typing import Optional
from projectaria_tools.core.vrs import extract_audio_track

from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.calibration import distort_by_calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeDomain
from hand_tracking_utils import (
    get_camera_calibration,
    project_3d_to_2d,
    draw_hand_skeleton,
    draw_velocity_axes,
    compute_velocity,
)

from gaze_utils import GazeMPSLoader

def extract_audio(vrs_file_path: str) -> Optional[str]:
    """Extract audio from a VRS file as a wav file in a temporary folder."""
    temp_folder = tempfile.mkdtemp()
    if not temp_folder:
        return None
    # else continue process vrs audio extraction
    json_output_string = extract_audio_track(
        vrs_file_path, os.path.join(temp_folder, "audio.wav")
    )
    json_output = json.loads(json_output_string)  # Convert string to Dict
    if json_output and json_output["status"] == "success":
        return json_output["output"]
    # Else we were not able to export a Wav file from the VRS file
    return None

def is_hand_closed_by_distance(landmarks, wrist_idx=0, threshold=0.12):
    """
    Check if hand is closed by measuring fingertip distances from wrist.
    Threshold should be around 0.12 for normalized coordinates.
    """
    if landmarks[wrist_idx] is None:
        return False
    
    wrist = landmarks[wrist_idx]
    tip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    dists = []
    
    for i in tip_indices:
        if landmarks[i] is not None:
            dists.append(np.linalg.norm(landmarks[i] - wrist))
    
    if len(dists) == 0:
        return False
    
    mean_dist = np.mean(dists)
    # Closed if mean distance is small
    # Typical values: closed ~0.08-0.12, open ~0.20-0.30
    return mean_dist < threshold


def is_finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx, threshold_deg=160):
    """
    Check if a finger is extended by measuring joint angles.
    Extended fingers have angles CLOSE TO 180° (straight).
    """
    if any(landmarks[i] is None for i in [mcp_idx, pip_idx, dip_idx, tip_idx]):
        return False
    
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    dip = landmarks[dip_idx]
    tip = landmarks[tip_idx]
    
    # Vectors for finger segments
    v1 = pip - mcp
    v2 = dip - pip
    v3 = tip - dip
    
    # Normalize
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v3 /= np.linalg.norm(v3)
    
    # Angles between segments
    ang1 = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    ang2 = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0)))
    
    # Extended if both joints are relatively straight (angles close to 180°)
    # Use GREATER THAN threshold (not less than)
    return (ang1 > threshold_deg) and (ang2 > threshold_deg)


def is_thumb_extended(landmarks, threshold_dist=0.08):
    """
    Special check for thumb since it moves differently.
    Compare thumb tip distance to index finger MCP.
    """
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    
    if any(x is None for x in [thumb_tip, thumb_mcp, index_mcp]):
        return False
    
    # Distance from thumb tip to its base
    thumb_length = np.linalg.norm(thumb_tip - thumb_mcp)
    
    # Distance from thumb tip to index MCP
    thumb_to_index = np.linalg.norm(thumb_tip - index_mcp)
    
    # Thumb is extended if it's far from the index finger
    return thumb_to_index > threshold_dist


def is_hand_open(landmarks):
    """
    Check if hand is open by counting extended fingers.
    """
    # Define finger joint indices (MCP, PIP, DIP, TIP)
    fingers = {
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20),
    }
    
    extended_count = 0
    
    # Check thumb separately (different anatomy)
    if is_thumb_extended(landmarks):
        extended_count += 1
    
    # Check other four fingers
    for finger_name, (mcp, pip, dip, tip) in fingers.items():
        if is_finger_extended(landmarks, mcp, pip, dip, tip, threshold_deg=160):
            extended_count += 1
    
    # Hand is open if 4 or more fingers are extended
    return extended_count >= 4


def is_hand_closed(landmarks):
    """
    Check if hand is closed (fist).
    """
    # Define finger joint indices
    fingers = {
        "index":  (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring":   (13, 14, 15, 16),
        "pinky":  (17, 18, 19, 20),
    }
    
    closed_count = 0
    
    # Check if thumb is tucked in
    if not is_thumb_extended(landmarks):
        closed_count += 1
    
    # Check other four fingers
    for finger_name, (mcp, pip, dip, tip) in fingers.items():
        # Closed = NOT extended
        if not is_finger_extended(landmarks, mcp, pip, dip, tip, threshold_deg=160):
            closed_count += 1
    
    # Hand is closed if 4 or more fingers are NOT extended
    return closed_count >= 4


from scipy.spatial.transform import Rotation as R

def compute_hand_rotation_matrix(landmarks_3d):
    wrist = landmarks_3d[0]
    index_mcp = landmarks_3d[5]
    middle_mcp = landmarks_3d[9]
    ring_mcp = landmarks_3d[13]
    
    if any(v is None for v in (wrist, index_mcp, middle_mcp, ring_mcp)):
        return None
    
    # Define axes more clearly:
    # X-axis: points from wrist toward middle finger (forward)
    x_axis = middle_mcp - wrist
    x_axis /= np.linalg.norm(x_axis)
    
    # Y-axis: points from middle to index (sideways across palm)
    side_vec = index_mcp - ring_mcp
    side_vec /= np.linalg.norm(side_vec)
    
    # Z-axis: palm normal (perpendicular to palm surface)
    z_axis = np.cross(x_axis, side_vec)
    z_axis /= np.linalg.norm(z_axis)
    
    # Recompute Y to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    # Build rotation matrix with proper column ordering
    R_hand = np.column_stack((x_axis, y_axis, z_axis))
    return R_hand


def draw_centered_rotating_rectangle(image, R_hand, size=100, color=(0,255,0)):
    """
    Draw a rotated rectangle representing hand orientation.
    The rectangle plane is aligned with the palm.
    """
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    
    # Use X and Y axes from the rotation matrix
    # These represent the palm plane directions
    x_axis = R_hand[:, 0]  # forward direction
    y_axis = R_hand[:, 1]  # side direction
    
    # Project 3D axes to 2D screen coordinates
    # Flip Y because screen coordinates have Y pointing down
    x_dir = np.array([x_axis[0], -x_axis[1]]) * size
    y_dir = np.array([y_axis[0], -y_axis[1]]) * size
    
    # Define rectangle corners
    corners = np.array([
        [cx, cy] + x_dir + y_dir,
        [cx, cy] - x_dir + y_dir,
        [cx, cy] - x_dir - y_dir,
        [cx, cy] + x_dir - y_dir
    ]).astype(int)
    
    # Draw with transparency
    overlay = image.copy()
    cv2.fillPoly(overlay, [corners], color)
    image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    cv2.polylines(image, [corners], isClosed=True, color=color, thickness=2)
    
    # Optional: Draw axis indicators to debug orientation
    cv2.arrowedLine(image, (cx, cy), 
                    (int(cx + x_dir[0]), int(cy + x_dir[1])),
                    (255, 0, 0), 2, tipLength=0.3)  # Red: X-axis
    cv2.arrowedLine(image, (cx, cy), 
                    (int(cx + y_dir[0]), int(cy + y_dir[1])),
                    (0, 255, 0), 2, tipLength=0.3)  # Green: Y-axis
    
    return image



def main():
    parser = argparse.ArgumentParser(description="process hand tracking on undistorted rgb camera")
    parser.add_argument("--vrs", type=str, required=True, help="path to .vrs file")
    parser.add_argument("--mps", type=str, required=True, help="path to hand_tracking_results.csv")
    parser.add_argument("--mps_base", type=str, default=None, help="path to mps base folder (optional, auto-derived if not provided)")
    parser.add_argument("--output", type=str, default="output_rgb_hands", help="output directory")
    parser.add_argument("--frame_skip", type=int, default=1, help="process every nth frame")
    parser.add_argument("--show_gaze", action="store_true", help="overlay eye gaze on video")
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
    




    #processing audio
    audio_path = None
    audio_path = extract_audio(args.vrs)
    if audio_path:
        print(f"audio extraction done: {audio_path}")
    else:
        print("Unable to extract audio from VRS file. Exiting...")
        exit(1)

    #init whisper model
    model_size = "base.en"
    whisper_model = WhisperModel(model_size_or_path= model_size, device="cpu", compute_type="int8")

    ##1
    # segments, info = whisper_model.transcribe(
    #     audio_path, beam_size=5, vad_filter=False,
    # )

    # print("Detected text segments:")
    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    
    # ##2
    # segments, _ = whisper_model.transcribe(
    #     audio_path, word_timestamps=True, vad_filter=True
    # )
    # print("Detected text WORD segments:")
    # for segment in segments:
    #     for word in segment.words:
    #         print(f"[{round(word.start,2)}s, -> {round(word.end,2)}s] {word.word}")

    ##3
    audio_stream_id = vrs_data_provider.get_stream_id_from_label("mic")
    audio_starting_timestamp = vrs_data_provider.get_first_time_ns(
        audio_stream_id, TimeDomain.DEVICE_TIME
    )
    segments, _ = whisper_model.transcribe(
        audio_path, word_timestamps=True, vad_filter=True
    )
    # save data to an array to log them to a CSV file
    speech_data = []#[["startTime_ns", "endTime_ns", "written"]]
    print("Detected text segments (time aligned to Aria time domain):")
    print(f"VRS audio stream starting timestamp(ns): {audio_starting_timestamp}")
    for segment in segments:
        current_phrase = ""
        cuurrent_start = None
        for word in segment.words:
            # move to aria TimeDomain
            s_to_ns = int(1e9)
            begin = int(word.start * s_to_ns + audio_starting_timestamp)
            if cuurrent_start == None:
                cuurrent_start = begin
            end = int(word.end * s_to_ns + audio_starting_timestamp)
            if word.word.endswith(","):
                current_phrase += word.word[:-1]
                speech_data.append([(cuurrent_start, end), current_phrase.strip()])
                cuurrent_start = None
                current_phrase = ""
            else:
                current_phrase += word.word
                current_phrase += " "
        
        if current_phrase != "":
            speech_data.append([(cuurrent_start, end), current_phrase.strip()])
            # print(f"[{begin}ns, -> {end}ns] {word.word}")
            # data.append([begin, end, word.word, word.probability])
    
    # print(speech_data)

    speech_json_name = os.path.join(args.output, "speech_data.json")

    print(f"saving speech data to json...")
    print(speech_json_name)
    print(f"cur_path: {os.getcwd()}")

    with open(speech_json_name, "w") as f:
        json.dump(speech_data, f, indent=2)
    print(f"speech data saved to: {speech_json_name}")
    
    if audio_path:
        shutil.rmtree(os.path.dirname(audio_path))

    # Show how to export this data to CSV
    # filename = "speech.csv"
    # with open(filename, mode="w") as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)


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
    pinhole_calib = calibration.get_linear_camera_calibration(image_size[0], image_size[1], focal_lengths[0])

    # get device to rgb camera transform
    device_calib = vrs_data_provider.get_device_calibration()
    T_device_camera = device_calib.get_transform_device_sensor(rgb_camera_label).inverse().to_matrix()

    # load hand tracking
    print(f"loading hand tracking from: {args.mps}")
    hand_tracking_df = pd.read_csv(args.mps)
    print(f"loaded {len(hand_tracking_df)} tracking samples")

    # load gaze if requested
    gaze_loader = None
    if args.show_gaze:
        # derive mps base path from hand tracking csv path if not provided
        if args.mps_base is None:
            # e.g., "path/to/mps_Orange_v1_vrs/hand_tracking/hand_tracking_results.csv" -> "path/to/mps_Orange_v1_vrs"
            mps_base = os.path.dirname(os.path.dirname(args.mps))
        else:
            mps_base = args.mps_base

        print(f"loading gaze from: {mps_base}")
        try:
            gaze_loader = GazeMPSLoader(mps_base, vrs_data_provider, use_general_gaze=True)
        except Exception as e:
            print(f"warning: could not load gaze data: {e}")
            print("continuing without gaze overlay...")
            args.show_gaze = False

    # setup output
    os.makedirs(args.output, exist_ok=True)

    # setup video writer
    video_filename = "undistorted_rgb_with_hands_and_gaze.mp4" if args.show_gaze else "undistorted_rgb_with_hands.mp4"
    output_video_path = os.path.join(args.output, video_filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # output_video_path = "undistorted_rgb_with_hands.avi"
    fps = 30
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (image_size[0], image_size[1]))
    print(f"writing video to: {output_video_path}")

    # process frames
    num_frames = vrs_data_provider.get_num_data(rgb_stream_id)
    print(f"total frames in rgb stream: {num_frames}")

    # track palm positions for velocity computation
    right_palm_positions = []
    right_palm_timestamps = []
    left_palm_positions = []
    left_palm_timestamps = []
    hand_rot = {} # timestamp_us -> (R_hand, L_hand) : (3x3 list, 3x3 list)
    hand_open_states =  {} # timestamp_us -> (l_open, r_open) : (bool, bool)

    all_frame_data = []

    audio_idx = 0
    current_audio_timestamp = speech_data[audio_idx][0]
    current_speech_text = None

    print("\nprocessing frames...")
    for frame_idx in tqdm(range(0, num_frames, args.frame_skip)):
        # get rgb image data by index
        image_data_and_record = vrs_data_provider.get_image_data_by_index(rgb_stream_id, frame_idx)

        if image_data_and_record is None:
            continue

        image = image_data_and_record[0].to_numpy_array()
        timestamp_ns = image_data_and_record[1].capture_timestamp_ns

        # undistort the frame using projectaria calibration
        # this converts fisheye -> pinhole projection
        undistorted_image = distort_by_calibration(image, pinhole_calib, rgb_camera_calibration)
        undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)
        
        if timestamp_ns >= current_audio_timestamp[0] and timestamp_ns <= current_audio_timestamp[1]:
            current_speech_text = speech_data[audio_idx][1]
        elif timestamp_ns > current_audio_timestamp[1]:
            audio_idx += 1
            if audio_idx < len(speech_data):
                current_audio_timestamp = speech_data[audio_idx][0]
                current_speech_text = None
                if timestamp_ns >= current_audio_timestamp[0] and timestamp_ns <= current_audio_timestamp[1]:
                    current_speech_text = speech_data[audio_idx][1]


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

            # calculates hand in form of (X-Axis, Y-Axis, Z-Axis)
            R_hand = compute_hand_rotation_matrix(landmarks_3d)
            if R_hand is not None:
                if timestamp_us not in hand_rot:
                    hand_rot[timestamp_us] = R_hand.tolist()
                else: 
                    raise ValueError("Duplicate timestamp in hand rotations!")
                undistorted_image = draw_centered_rotating_rectangle(undistorted_image, R_hand, size=100, color=(0,255,0))

            # hand_open = is_hand_open(landmarks_3d)
            hand_open = is_hand_open(landmarks_3d)
            # frame_data["hand_open"] = hand_open

            label_text = "OPEN" if hand_open else "CLOSED"
            
            cv2.putText(undistorted_image, f"Right hand: {label_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if hand_open else (0,0,255), 2)
    
            if timestamp_us not in hand_open_states:
                if label_text == "OPEN":
                    hand_open_states[timestamp_us] = True
                elif label_text == "CLOSED":
                    hand_open_states[timestamp_us] = False
                else:
                    raise ValueError("Unexpected hand state label!")                    
            else:
                raise ValueError("Duplicate timestamp in hand open states!")
            
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

                    # project w/ pinhole calibration
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
            right_vels = compute_velocity(np.array(right_palm_positions), np.array(right_palm_timestamps), window_size=3)
            right_vel = right_vels[-1]

        if len(left_palm_positions) >= 4:
            left_vels = compute_velocity(np.array(left_palm_positions), np.array(left_palm_timestamps), window_size=3)
            left_vel = left_vels[-1]

        undistorted_image = cv2.rotate(undistorted_image, cv2.ROTATE_90_CLOCKWISE)
        # get and draw gaze if available
        if gaze_loader is not None:
            try:
                # get gaze projection for undistorted (pinhole) frame
                gaze_projection = gaze_loader.get_gaze_projection(
                    timestamp_ns,
                    rgb_camera_label,
                    device_calib,
                    pinhole_calib,  # use pinhole calib for undistorted frame
                    depth_m=1.0
                )

                if gaze_projection is not None:
                    gaze_x, gaze_y = gaze_projection
                    height, width = undistorted_image.shape[:2]
                    # draw red circle at gaze point
                    if 0 <= gaze_x < width and 0 <= gaze_y < height:
                        cv2.circle(undistorted_image, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
            except Exception as e:
                    # silently skip if gaze projection fails for this frame
                    print(f"warning: gaze projection failed at {timestamp_ns} ns: {e}")
                    pass
            
        # draw velocity overlay
        undistorted_image = draw_velocity_axes(undistorted_image, right_vel, left_vel)
        if current_speech_text:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1
            color = (255, 255, 255)  # white text
            thickness = 2

            h, w = undistorted_image.shape[:2]

            # ---- WRAP TEXT ----
            max_text_width = int(w * 0.8)  # limit to 80% of screen width
            char_width = cv2.getTextSize("A", font, font_scale, thickness)[0][0]
            max_chars_per_line = max(1, max_text_width // char_width)
            wrapped_lines = textwrap.wrap(current_speech_text, width=max_chars_per_line)

            # ---- COMPUTE TEXT BLOCK SIZE ----
            line_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in wrapped_lines]
            line_height = max(sz[1] for sz in line_sizes) + 10
            total_text_height = line_height * len(wrapped_lines)
            max_line_width = max(sz[0] for sz in line_sizes)

            # Center horizontally
            x_start = (w - max_line_width) // 2
            y_start = 60  # top margin

            # Rectangle padding
            pad_x = 15
            pad_y = 10

            # ---- DRAW BACKGROUND RECTANGLE ----
            rect_left = x_start - pad_x
            rect_top = y_start - line_height + pad_y
            rect_right = x_start + max_line_width + pad_x
            rect_bottom = y_start + total_text_height - line_height + pad_y

            # Optional: semi-transparent background
            overlay = undistorted_image.copy()
            cv2.rectangle(overlay, (rect_left, rect_top), (rect_right, rect_bottom), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, undistorted_image, 0.5, 0, undistorted_image)

            # ---- DRAW TEXT LINES ----
            for i, line in enumerate(wrapped_lines):
                text_width, text_height = line_sizes[i]
                x = (w - text_width) // 2
                y = y_start + i * line_height
                cv2.putText(undistorted_image, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


        all_frame_data.append(frame_data)

        # write frame
        video_writer.write(undistorted_image)

    # cleanup
    video_writer.release()

    # save hand open states and rotations to json
    hand_open_states_json_path = os.path.join(args.output, "hand_open_states.json")
    print("\nsaving hand open states to json...")
    with open(hand_open_states_json_path, "w") as f:
        json.dump(hand_open_states, f, indent=2)
    
    hand_rot_json_path = os.path.join(args.output, "hand_rotations.json")
    print("saving hand rotations to json...")
    with open(hand_rot_json_path, "w") as f:
        json.dump(hand_rot, f, indent=2)

    


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
