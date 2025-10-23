"""
aria_camera_rectification.py

handles aria fisheye camera rectification and 3d→2d projection.
maps aria hand tracking coordinates to image coordinates for embodied-cot.

follows the same flow as gripper_positions.py but adapted for aria's camera model.
"""

import numpy as np
from typing import Tuple, Dict
from projectaria_tools.core import calibration
from projectaria_tools.core.calibration import (
    CameraCalibration,
    get_linear_camera_calibration,
    distort_by_calibration
)
import cv2


class AriaProjectionMapper:
    """
    maps aria 3d hand tracking to 2d image coordinates.

    handles:
    1. fisheye → linear (pinhole) rectification
    2. 3d point projection using rectified camera model
    3. coordinate frame transforms

    this replaces the OWL-ViT + SAM detection pipeline used in gripper_positions.py,
    since we have direct 3d tracking from aria.
    """

    def __init__(self,
                 aria_camera_calib: CameraCalibration,
                 target_image_size: Tuple[int, int] = (256, 256),
                 target_focal_length: float = 150):
        """
        args:
            aria_camera_calib: aria's fisheye camera calibration
            target_image_size: output image size (h, w)
            target_focal_length: focal length for rectified camera (adjust for desired fov)
        """
        self.fisheye_calib = aria_camera_calib
        self.target_size = target_image_size

        # create linear (pinhole) camera model for rectification
        self.linear_calib = get_linear_camera_calibration(
            target_image_size[1],  # width
            target_image_size[0],  # height
            target_focal_length,
            "pinhole"
        )

        # precompute rectification map
        # this maps fisheye pixels → rectified pixels
        self.rectify_map = None

    def setup_rectification_map(self, source_image_shape: Tuple[int, int]):
        """
        precomputes rectification map from fisheye to pinhole.

        args:
            source_image_shape: (height, width) of aria fisheye images
        """
        # get the mapping from linear → fisheye
        # we'll use this in reverse with cv2.remap
        self.rectify_map = distort_by_calibration(
            self.linear_calib,      # destination camera (rectified)
            self.fisheye_calib,     # source camera (fisheye)
            source_image_shape[1],  # source width
            source_image_shape[0]   # source height
        )

    def rectify_image(self, fisheye_image: np.ndarray) -> np.ndarray:
        """
        rectifies aria fisheye image to pinhole projection.

        this is crucial - embodied-cot expects rectified images, not fisheye.

        args:
            fisheye_image: aria rgb image (h, w, 3)

        returns:
            rectified image at target_size
        """
        if self.rectify_map is None:
            self.setup_rectification_map(fisheye_image.shape[:2])

        # apply rectification using precomputed map
        rectified = cv2.remap(
            fisheye_image,
            self.rectify_map[:, :, :2].astype(np.float32),
            None,
            cv2.INTER_LINEAR
        )

        return rectified

    def project_3d_to_rectified_2d(self,
                                     point_3d_device: np.ndarray,
                                     T_device_camera: np.ndarray) -> Tuple[int, int]:
        """
        projects 3d point (in device frame) to 2d coordinates in rectified image.

        this is the key function that replaces OWL-ViT detection!
        instead of detecting the gripper, we project the known 3d wrist position.

        args:
            point_3d_device: [x, y, z] in aria device frame (meters)
            T_device_camera: 4x4 transform from device to camera frame

        returns:
            (u, v) pixel coordinates in rectified image
        """
        # transform point from device frame to camera frame
        point_4d = np.append(point_3d_device, 1.0)
        point_camera = T_device_camera @ point_4d
        point_camera = point_camera[:3]  # [x, y, z] in camera frame

        # project using linear (pinhole) model
        # this gives us coordinates in the rectified image space
        projected_2d = self.linear_calib.project(point_camera)

        return int(projected_2d[0]), int(projected_2d[1])

    def get_gripper_positions_from_trajectory(self,
                                               wrist_positions_3d: list,
                                               T_device_camera: np.ndarray) -> list:
        """
        converts aria hand tracking trajectory to gripper positions.

        this replaces process_trajectory() from gripper_positions.py.
        instead of detecting gripper with vision models, we use direct 3d tracking.

        args:
            wrist_positions_3d: list of [x, y, z] wrist positions (device frame)
            T_device_camera: transform from device to rgb camera

        returns:
            list of (x, y) pixel coordinates in rectified 256x256 images
        """
        gripper_positions = []

        for wrist_3d in wrist_positions_3d:
            if wrist_3d is None:
                # tracking lost - mark as invalid
                gripper_positions.append(None)
            else:
                # project to 2d
                u, v = self.project_3d_to_rectified_2d(wrist_3d, T_device_camera)

                # check if in bounds
                if 0 <= u < self.target_size[1] and 0 <= v < self.target_size[0]:
                    gripper_positions.append((u, v))
                else:
                    # out of frame
                    gripper_positions.append(None)

        # handle missing tracking with interpolation (same as gripper_positions.py)
        gripper_positions = self._interpolate_missing(gripper_positions)

        return gripper_positions

    def _interpolate_missing(self, positions: list) -> list:
        """
        interpolates missing gripper positions (same strategy as original code).
        """
        result = []

        # find first and last valid indices
        valid_indices = [i for i, p in enumerate(positions) if p is not None]

        if not valid_indices:
            # no valid tracking in entire trajectory - can't use this episode
            return [(-1, -1)] * len(positions)

        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]

        for i, pos in enumerate(positions):
            if pos is not None:
                result.append(pos)
            else:
                # interpolate from neighbors
                if i < first_valid:
                    # use first valid
                    result.append(positions[first_valid])
                elif i > last_valid:
                    # use last valid
                    result.append(positions[last_valid])
                else:
                    # find nearest valid neighbors
                    prev_valid = max([idx for idx in valid_indices if idx < i])
                    next_valid = min([idx for idx in valid_indices if idx > i])

                    # use closer neighbor
                    if i - prev_valid < next_valid - i:
                        result.append(positions[prev_valid])
                    else:
                        result.append(positions[next_valid])

        return result


def get_device_to_camera_transform(device_calib) -> np.ndarray:
    """
    extracts transform from device frame to rgb camera frame.

    aria provides this in the calibration data.

    args:
        device_calib: aria device calibration object

    returns:
        4x4 transformation matrix
    """
    # get rgb camera label (main camera)
    rgb_label = "camera-rgb"

    # get the transform
    T_device_camera = device_calib.get_transform_device_sensor(rgb_label).to_matrix()

    return T_device_camera


# integration with existing embodied-cot pipeline
def aria_gripper_extraction_pipeline(aria_recording_path: str,
                                      mps_output_path: str,
                                      output_size: tuple = (256, 256)):
    """
    complete pipeline to extract gripper positions from aria data.

    this replaces the OWL-ViT + SAM pipeline in gripper_positions.py.

    workflow:
    1. load aria recording + mps hand tracking
    2. rectify fisheye images to 256x256 pinhole
    3. project 3d wrist positions to 2d rectified coordinates
    4. save in format compatible with embodied-cot

    args:
        aria_recording_path: path to .vrs file
        mps_output_path: path to mps output folder
        output_size: target image size

    returns:
        dict with episodes, each containing:
        - rectified_images: list of 256x256 numpy arrays
        - gripper_positions: list of (x, y) tuples
        - state_3d: list of [x, y, z] positions
    """
    from projectaria_tools.core import data_provider
    from aria_data_loader import AriaDataLoader

    # load aria data
    provider = data_provider.create_vrs_data_provider(aria_recording_path)
    device_calib = provider.get_device_calibration()

    # get camera calibration
    rgb_label = "camera-rgb"
    camera_calib = device_calib.get_camera_calib(rgb_label)

    # create projection mapper
    mapper = AriaProjectionMapper(
        aria_camera_calib=camera_calib,
        target_image_size=output_size,
        target_focal_length=150  # adjust based on desired fov
    )

    # get device→camera transform
    T_device_camera = get_device_to_camera_transform(device_calib)

    # load hand tracking from mps
    loader = AriaDataLoader(aria_recording_path, mps_output_path)

    # extract episodes (assuming you have episode segmentation)
    episodes = []

    # example: process entire recording as one episode
    # in practice, you'd segment into multiple episodes
    num_frames = provider.get_num_data(loader.rgb_stream_id)

    episode_data = {
        'rectified_images': [],
        'gripper_positions': [],
        'state_3d': [],
        'wrist_orientations': []
    }

    wrist_positions_3d = []

    # first pass: extract all data
    for frame_idx in range(0, num_frames, 3):  # subsample to ~10fps
        frame_data = loader.get_synced_frame_data(frame_idx)

        if frame_data is None:
            continue

        # rectify image
        rectified_img = mapper.rectify_image(frame_data['image'])
        episode_data['rectified_images'].append(rectified_img)

        # get wrist position
        hand_data = frame_data.get('hand_tracking', {}).get('right_hand')
        if hand_data and hand_data['is_valid']:
            wrist_3d = hand_data['wrist_position']
            wrist_positions_3d.append(wrist_3d)
            episode_data['state_3d'].append(wrist_3d.tolist())
            episode_data['wrist_orientations'].append(hand_data['wrist_orientation'].tolist())
        else:
            wrist_positions_3d.append(None)
            episode_data['state_3d'].append(None)
            episode_data['wrist_orientations'].append(None)

    # second pass: project 3d to 2d gripper positions
    gripper_positions = mapper.get_gripper_positions_from_trajectory(
        wrist_positions_3d,
        T_device_camera
    )
    episode_data['gripper_positions'] = gripper_positions

    # interpolate missing state_3d (same strategy)
    episode_data['state_3d'] = mapper._interpolate_missing(episode_data['state_3d'])

    episodes.append(episode_data)

    return episodes


# example usage matching embodied-cot format
def example_aria_to_ecot_features():
    """
    example showing how to format aria data for embodied-cot reasoning generation.
    """
    import json

    # extract aria data
    episodes = aria_gripper_extraction_pipeline(
        aria_recording_path="recording.vrs",
        mps_output_path="mps_output/"
    )

    # format for embodied-cot (matching reasonings_dataset.json structure)
    reasonings_format = {}

    for ep_idx, episode in enumerate(episodes):
        file_path = f"aria_recording_{ep_idx}"

        reasonings_format[file_path] = {
            str(ep_idx): {
                "features": {
                    "gripper_position": episode['gripper_positions'],  # list of [x, y]
                    "state_3d": episode['state_3d'],  # list of [x, y, z]
                    # move_primitive would be computed separately
                },
                "metadata": {
                    "episode_id": str(ep_idx),
                    "file_path": file_path,
                    "n_steps": len(episode['gripper_positions']),
                    "language_instruction": "pick up the cup"  # annotate separately
                }
                # "reasoning" will be generated by full_reasonings.py
            }
        }

    # save in format compatible with full_reasonings.py
    with open('aria_features.json', 'w') as f:
        json.dump(reasonings_format, f, indent=2)

    print(f"saved {len(episodes)} episodes in embodied-cot format")

    return reasonings_format


if __name__ == "__main__":
    # demonstrate the pipeline
    example_aria_to_ecot_features()
