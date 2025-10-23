"""
aria_to_rlds.py

converts aria glasses egocentric data to the augmented rlds format compatible with embodied-cot.
handles the mapping from human hand tracking to robot-compatible representations.
"""

import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional
from pathlib import Path


class AriaToRLDSConverter:
    """
    converts aria glasses data to embodied-cot compatible format.

    aria provides:
    - egocentric rgb images (1408x1408 from main camera)
    - hand tracking (wrist + finger positions in 3d)
    - eye gaze data
    - imu/slam poses

    we need to map to:
    - gripper positions (2d image coordinates)
    - state_3d (end-effector positions)
    - move primitives (discrete movement classifications)
    """

    def __init__(self,
                 aria_dataset_path: str,
                 output_rlds_dir: str,
                 image_size: tuple = (256, 256)):
        """
        args:
            aria_dataset_path: path to aria dataset (projectaria tools format)
            output_rlds_dir: where to save the rlds-compatible dataset
            image_size: target image size for the model
        """
        self.aria_path = Path(aria_dataset_path)
        self.output_dir = Path(output_rlds_dir)
        self.image_size = image_size

        # mapping constants
        self.movement_threshold = 0.02  # meters, for classifying primitives
        self.velocity_window = 5  # frames for velocity calculation

    def extract_hand_features(self, hand_tracking_data: Dict) -> Dict:
        """
        extracts relevant features from aria hand tracking.

        aria hand tracking provides:
        - wrist position (3d)
        - joint positions (21 keypoints per hand)
        - confidence scores

        we map to:
        - gripper_position (2d): project wrist to image plane
        - state_3d: use wrist position as end-effector proxy
        """
        features = {
            'gripper_position': [],
            'state_3d': [],
            'hand_orientation': [],
            'grasp_state': []  # inferred from hand pose
        }

        for frame in hand_tracking_data:
            # wrist tracking
            if frame['right_hand']['is_valid']:
                wrist_3d = frame['right_hand']['wrist_position']  # [x, y, z]

                # project to 2d image coordinates
                # assumes you have camera intrinsics from aria
                gripper_2d = self._project_to_image(
                    wrist_3d,
                    frame['camera_pose'],
                    frame['camera_intrinsics']
                )

                features['gripper_position'].append(gripper_2d)
                features['state_3d'].append(wrist_3d)

                # infer grasp state from finger distances
                grasp = self._infer_grasp_state(frame['right_hand']['joint_positions'])
                features['grasp_state'].append(grasp)

                # hand orientation from wrist rotation
                features['hand_orientation'].append(
                    frame['right_hand']['wrist_orientation']
                )
            else:
                # handle missing tracking (use previous or interpolate)
                features['gripper_position'].append(None)
                features['state_3d'].append(None)
                features['grasp_state'].append(None)
                features['hand_orientation'].append(None)

        return features

    def _project_to_image(self, point_3d: np.ndarray,
                          camera_pose: np.ndarray,
                          intrinsics: Dict) -> tuple:
        """
        projects 3d point to 2d image coordinates using aria camera model.

        aria uses fisheye camera model, need to:
        1. transform point to camera frame
        2. apply fisheye distortion
        3. project to pixel coordinates
        """
        # transform to camera frame
        point_cam = camera_pose @ np.append(point_3d, 1)

        # project using fisheye model (simplified - use projectaria tools for full)
        x, y, z = point_cam[:3]

        # normalize
        x_n = x / z
        y_n = y / z

        # apply intrinsics
        fx, fy = intrinsics['focal_length']
        cx, cy = intrinsics['principal_point']

        u = fx * x_n + cx
        v = fy * y_n + cy

        # scale to target image size
        u = u * self.image_size[0] / intrinsics['image_width']
        v = v * self.image_size[1] / intrinsics['image_height']

        return (int(u), int(v))

    def _infer_grasp_state(self, joint_positions: np.ndarray) -> float:
        """
        infers grasp state from hand joint positions.

        measures distance between thumb and index finger tips.
        returns value in [0, 1] where 0=open, 1=closed
        """
        thumb_tip = joint_positions[4]  # thumb tip index
        index_tip = joint_positions[8]  # index tip

        distance = np.linalg.norm(thumb_tip - index_tip)

        # normalize to [0, 1] range
        # typical open hand: ~0.15m, closed: ~0.02m
        grasp_value = 1.0 - np.clip((distance - 0.02) / 0.13, 0, 1)

        return grasp_value

    def classify_movement_primitives(self, trajectory_3d: List[np.ndarray]) -> List[str]:
        """
        classifies movements into primitives based on velocity analysis.

        primitives (matching bridge dataset):
        - move_forward, move_backward
        - move_left, move_right
        - move_up, move_down
        - rotate_left, rotate_right
        - grasp, ungrasp
        - no_movement
        """
        primitives = []

        for i in range(len(trajectory_3d)):
            if i < self.velocity_window:
                primitives.append("no_movement")
                continue

            # compute velocity over window
            delta = trajectory_3d[i] - trajectory_3d[i - self.velocity_window]
            velocity = delta / self.velocity_window

            # classify based on dominant direction
            abs_vel = np.abs(velocity)

            if np.max(abs_vel) < self.movement_threshold:
                primitives.append("no_movement")
            else:
                dominant_axis = np.argmax(abs_vel)
                direction = np.sign(velocity[dominant_axis])

                movement_map = {
                    (0, 1): "move_forward",
                    (0, -1): "move_backward",
                    (1, 1): "move_left",
                    (1, -1): "move_right",
                    (2, 1): "move_up",
                    (2, -1): "move_down"
                }

                primitives.append(movement_map.get((dominant_axis, direction), "no_movement"))

        return primitives

    def generate_reasoning_features(self,
                                     episode_data: Dict,
                                     hand_features: Dict) -> Dict:
        """
        generates the feature dictionary needed for reasoning generation.
        matches format expected by full_reasonings.py
        """
        # handle missing/invalid tracking with interpolation
        state_3d = self._interpolate_missing(hand_features['state_3d'])
        gripper_positions = self._interpolate_missing(hand_features['gripper_position'])

        # classify movements
        move_primitives = self.classify_movement_primitives(state_3d)

        features = {
            'state_3d': state_3d,
            'gripper_position': gripper_positions,
            'move_primitive': move_primitives,
            'grasp_state': hand_features['grasp_state']
        }

        return features

    def _interpolate_missing(self, values: List) -> List:
        """
        interpolates missing tracking data (none values).
        """
        result = []
        last_valid = None

        # forward pass - fill with last valid
        for val in values:
            if val is not None:
                last_valid = val
            result.append(last_valid if last_valid is not None else val)

        # backward pass for leading nones
        last_valid = None
        for i in range(len(result) - 1, -1, -1):
            if result[i] is not None:
                last_valid = result[i]
            elif last_valid is not None:
                result[i] = last_valid

        return result

    def convert_episode(self,
                        aria_episode: Dict,
                        language_instruction: str,
                        episode_id: int,
                        file_path: str) -> Dict:
        """
        converts a single aria episode to the embodied-cot format.

        returns dict compatible with reasonings_dataset.json structure
        """
        # extract hand tracking features
        hand_features = self.extract_hand_features(aria_episode['hand_tracking'])

        # generate reasoning features
        features = self.generate_reasoning_features(aria_episode, hand_features)

        # build metadata
        metadata = {
            'episode_id': str(episode_id),
            'file_path': file_path,
            'n_steps': len(features['state_3d']),
            'language_instruction': language_instruction
        }

        # structure matches what full_reasonings.py expects
        entry = {
            'features': features,
            'metadata': metadata,
            # 'reasoning' will be added by full_reasonings.py
        }

        return entry


# example usage
def process_aria_dataset():
    """
    example pipeline for converting aria dataset
    """
    converter = AriaToRLDSConverter(
        aria_dataset_path='/path/to/aria/data',
        output_rlds_dir='/path/to/output/rlds'
    )

    # load aria episodes (using projectaria_tools)
    # you'd use: from projectaria_tools.core import data_provider

    episodes = []  # load your aria episodes here

    results = {}

    for i, episode in enumerate(episodes):
        # you need to annotate with language instructions
        # could use vlm to generate from first frame, or manual annotation
        instruction = "pick up the cup"  # placeholder

        converted = converter.convert_episode(
            episode,
            language_instruction=instruction,
            episode_id=i,
            file_path=f"aria_episode_{i}"
        )

        file_path = converted['metadata']['file_path']
        episode_id = converted['metadata']['episode_id']

        if file_path not in results:
            results[file_path] = {}
        results[file_path][episode_id] = converted

    # save in format compatible with embodied-cot
    with open('aria_features.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"converted {len(episodes)} episodes")

    # next step: run full_reasonings.py to generate chain-of-thought annotations
    # then merge with reasonings_dataset.json


if __name__ == "__main__":
    process_aria_dataset()
