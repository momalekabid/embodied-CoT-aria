import numpy as np
import projectaria_tools.core.mps as mps
# Example query: find the nearest eye gaze data outputs in relation to a specific timestamp
from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import (
    get_gaze_vector_reprojection,
    get_nearest_eye_gaze
)
from os import path
import platform
import os
import cv2
from projectaria_tools.core.calibration import distort_by_calibration
from projectaria_tools.core.calibration import CameraCalibration



# Requires:
# vrs_path: path to the VRS file (e.g. .../Orange_v1/Orange_v1.vrs)
# mps_path: path to the MPS data folder (e.g. .../Orange_v1/mps_Orange_v1_vrs)
#
# Returns:
# - np.ndarray of shape (N, 2) with the 2D gaze projections for each distorted frame in the VRS RGB stream
# def get_gaze(vrs_path, mps_path) -> np.ndarray:

def get_camera_calibration(vrs_data_provider, stream_id: StreamId) -> CameraCalibration:
    """retrieve camera calibration for given stream id"""
    device_calibration = vrs_data_provider.get_device_calibration()
    stream_label = vrs_data_provider.get_label_from_stream_id(stream_id)
    return device_calibration.get_camera_calib(stream_label)

# def undistort_image(provider, rgb_stream_id, image_distorted) -> np.ndarray:
#         rgb_camera_calibration = get_camera_calibration(provider, rgb_stream_id)
#         focal_lengths = rgb_camera_calibration.get_focal_lengths()
#         image_size = rgb_camera_calibration.get_image_size()

#         # create pinhole (undistorted) calibration
#         pinhole_calib = calibration.get_linear_camera_calibration(image_size[0], image_size[1], focal_lengths[0])
        
#         # undistort image
#         image_undistorted = distort_by_calibration(
#             image_distorted,
#             pinhole_calib,
#             rgb_camera_calibration
#         )

#         # rotate image to correct orientation
#         image_undistorted = cv2.rotate(image_undistorted, cv2.ROTATE_90_CLOCKWISE)
        
#         return image_undistorted

if __name__ == "__main__":
    gaze_path_base = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data/mps_Microsoft_office_1_vrs/eye_gaze/general_eye_gaze.csv"
    gaze_path_full = ""
    vrs_path = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data_1/Orange_v1/Orange_v1.vrs"
    mps_path = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data_1/Orange_v1/mps_Orange_v1_vrs"
    
    # vrs_path = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data/Microsoft_office_1.vrs"
    # mps_path = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data/mps_Microsoft_office_1_vrs/"


    use_general_gaze = True


    if platform.system() == "Windows":
        gaze_path_full = path.join("C:", gaze_path_base)
    elif platform.system() == "Linux":
        gaze_path_full = "/mnt/c" + gaze_path_base
    

    if platform.system() == "Windows":
        vrs_path = path.join("C:", vrs_path)
    elif platform.system() == "Linux":
        vrs_path = "/mnt/c" + vrs_path

    if platform.system() == "Windows":
        mps_path = path.join("C:", mps_path)
    elif platform.system() == "Linux":
        mps_path = "/mnt/c" + mps_path

    # vrs_file = "../../Microsoft_office_1.vrs"
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_path)
    
    rgb_stream_id = StreamId("214-1")


    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)


    mps_data_paths_provider = mps.MpsDataPathsProvider(mps_path)
    mps_data_paths = mps_data_paths_provider.get_data_paths()
    mps_data_provider = mps.MpsDataProvider(mps_data_paths)
    assert mps_data_provider.has_general_eyegaze(), "The sequence does not have Eye Gaze data"

    rgb_camera_label = "camera-rgb"

    # # get rgb camera calibration
    rgb_camera_calibration = get_camera_calibration(vrs_data_provider, rgb_stream_id)
    focal_lengths = rgb_camera_calibration.get_focal_lengths()
    image_size = rgb_camera_calibration.get_image_size()

    print(f"using camera: {rgb_camera_label}")
    print(f"rgb image size: {image_size}")
    print(f"focal length: {focal_lengths[0]}")


    # # create pinhole (undistorted) calibration
    pinhole_calib = calibration.get_linear_camera_calibration(image_size[0], image_size[1], focal_lengths[0])
    # get device to rgb camera transform
    device_calib = vrs_data_provider.get_device_calibration()
    T_device_camera = device_calib.get_transform_device_sensor(rgb_camera_label).inverse().to_matrix()


    gazes = []

    num_frames = vrs_data_provider.get_num_data(rgb_stream_id)

    take_timestamps = []
    images = []


    for frame_idx in range(0, num_frames):
        image_data_and_record = vrs_data_provider.get_image_data_by_index(rgb_stream_id, frame_idx)

        if image_data_and_record is None:
            continue

        image = image_data_and_record[0].to_numpy_array()
        timestamp_ns = image_data_and_record[1].capture_timestamp_ns

        take_timestamps.append(timestamp_ns)

        # undistort the frame using projectaria calibration
        # this converts fisheye -> pinhole projection
        undistorted_image = distort_by_calibration(image, pinhole_calib, rgb_camera_calibration)
        cv2.imwrite("debug_undistorted.png", undistorted_image)

        undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)

        images.append(undistorted_image)

    print("Total frames to process:", len(take_timestamps))

    for sample in take_timestamps:

        # get the eye gaze data at each timestamp with mps_data_provider
        if use_general_gaze:
            eye_gaze = mps_data_provider.get_general_eyegaze(sample)
        else:
            eye_gaze = mps_data_provider.get_personalized_eyegaze(sample)

        # compute the corresponding 3D vector and retrieve its depth. Depth is set to default of 1.0 if eye gaze data doesn't provide depth.
        depth_m = eye_gaze.depth or 1.0

        # reproject the eye_gaze vector at Depth on a given image (using Calibration data)
        gaze_projection = get_gaze_vector_reprojection(
            eye_gaze,
            rgb_stream_label,
            device_calibration,
            rgb_camera_calibration,
            depth_m,
        )

        # gaze_projection = distort_by_calibration(
        #     np.array([[gaze_projection[0], gaze_projection[1]]]),
        #     rgb_camera_calibration,
        #     pinhole_calib,
        # )
        # print(gaze_projection)
        gazes.append(gaze_projection)

    images = np.array(images)
    gazes = np.round(np.array(gazes), decimals=0).astype(int)


    # create video with gaze overlay
    output_path = "gaze_projection.mp4"
    # determine fps from timestamps if available
    try:
        diffs = np.diff(take_timestamps) / 1e9
        median_dt = np.median(diffs) if len(diffs) > 0 else 0.1
        fps = int(round(1.0 / median_dt)) if median_dt > 0 else 10
    except Exception:
        fps = 10

    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(min(len(images), len(gazes))):
        frame = images[i].copy()
        x, y = int(gazes[i][0]), int(gazes[i][1])
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"Wrote video: {output_path} (fps={fps})")





 # python process_fisheye_hands.py --vrs "C:/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data/Microsoft_office_1.vrs" --mps "C:/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data/mps_Microsoft_office_1_vrs/hand_tracking/hand_tracking_results.csv" --output .




    # rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)



    # gaze_cpf = mps.read_eyegaze(gaze_path_full)
    # # Set default eye gaze depth for 3D points to 1 meter
    # depth_m = 3.0
    # gaze_point_cpf = mps.get_eyegaze_point_at_depth(gaze_cpf[1].yaw, gaze_cpf[1].pitch, depth_m)



    # # Query Eye Gaze data at a desired timestamp
    # # For this example we use an eyegaze data timestamp
    # # You can also use a VRS timestamp (i.e timestamp from a loop reading all the images)

    # query_timestamp_ns = int(gaze_cpf[1].tracking_timestamp.total_seconds() * 1e9)

    # eye_gaze_info = get_nearest_eye_gaze(gaze_cpf, query_timestamp_ns)

    # if eye_gaze_info:
    #     # Re-project the eye gaze point onto the RGB camera data
    #     vrs_file = "../../Microsoft_office_1.vrs"
    #     vrs_data_provider = data_provider.create_vrs_data_provider(vrs_path)

    #     rgb_stream_id = StreamId("214-1")
    #     rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    #     device_calibration = vrs_data_provider.get_device_calibration()
    #     rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    #     gaze_projection = get_gaze_vector_reprojection(
    #     eye_gaze_info,
    #     rgb_stream_label,
    #     device_calibration,
    #     rgb_camera_calibration,
    #     depth_m,
    # )
    #     print(gaze_projection)
