"""
hand tracking to gripper pose mapping with full 6dof + gripper state
uses hand landmarks (21 points) to compute position, rotation, and grasp state
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_hand_orientation(landmarks_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute hand orientation from landmarks using palm vectors
    uses improved landmark selection similar to teammate's approach

    args:
        landmarks_3d: (21, 3) array of hand landmarks in 3d space

    returns:
        rotation_matrix: (3, 3) rotation matrix for hand frame
        palm_normal: (3,) unit vector perpendicular to palm
    """
    # key points: wrist (0), index mcp (5), middle mcp (9), ring mcp (13)
    wrist = landmarks_3d[0]
    index_mcp = landmarks_3d[5]
    middle_mcp = landmarks_3d[9]
    ring_mcp = landmarks_3d[13]

    # build hand coordinate frame
    # x-axis: wrist to middle finger base (forward direction)
    x_axis = middle_mcp - wrist
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    # y-axis: sideways across palm (from ring to index)
    side_vec = index_mcp - ring_mcp
    side_vec = side_vec / (np.linalg.norm(side_vec) + 1e-8)

    # z-axis: palm normal (perpendicular to palm surface)
    z_axis = np.cross(x_axis, side_vec)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # recompute y to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    # rotation matrix [x, y, z] column vectors
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    return rotation_matrix, z_axis


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """
    convert rotation matrix to euler angles (roll, pitch, yaw)
    using ZYX convention (yaw-pitch-roll)

    args:
        R: (3, 3) rotation matrix

    returns:
        euler: (3,) array [roll, pitch, yaw] in radians
    """
    # handle gimbal lock
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])


def compute_gripper_state(landmarks_3d: np.ndarray) -> float:
    """
    compute gripper open/close state from hand landmarks
    uses multiple finger distances for robustness

    args:
        landmarks_3d: (21, 3) array of hand landmarks

    returns:
        gripper_state: float in [0, 1] where 0=open, 1=closed
    """
    # fingertip indices: thumb(4), index(8), middle(12), ring(16), pinky(20)
    thumb_tip = landmarks_3d[4]
    fingertips = np.array([
        landmarks_3d[8],   # index
        landmarks_3d[12],  # middle
        landmarks_3d[16],  # ring
    ])

    # compute distances from thumb to each fingertip
    # shape: (3, 3) -> (3,)
    distances = np.linalg.norm(fingertips - thumb_tip[None, :], axis=1)

    # use minimum distance (closest finger to thumb)
    min_distance = np.min(distances)

    # normalize: open hand (~0.15m) = 0, closed (~0.02m) = 1
    # using sigmoid-like mapping for smooth transition
    grasp_value = 1.0 - np.clip((min_distance - 0.02) / 0.13, 0, 1)

    return float(grasp_value)


def compute_hand_openness(landmarks_3d: np.ndarray) -> float:
    """
    alternative gripper state based on overall hand openness
    measures spread of all fingers

    args:
        landmarks_3d: (21, 3) array of hand landmarks

    returns:
        openness: float in [0, 1] where 0=closed fist, 1=open palm
    """
    palm = landmarks_3d[0]  # wrist base
    fingertips = np.array([landmarks_3d[i] for i in [4, 8, 12, 16, 20]])

    # compute variance of fingertip distances from palm
    distances = np.linalg.norm(fingertips - palm[None, :], axis=1)
    spread = np.std(distances)

    # normalize: closed fist (~0.01) = 0, open palm (~0.03) = 1
    openness = np.clip(spread / 0.03, 0, 1)

    return float(1.0 - openness)  # invert so 1=closed


def hand_to_gripper_transform(landmarks_3d: np.ndarray) -> Dict[str, np.ndarray]:
    """
    complete hand to gripper mapping with 6dof pose + gripper state

    args:
        landmarks_3d: (21, 3) array of hand landmarks in 3d space

    returns:
        transform dict containing:
            - position: (3,) wrist position
            - rotation_matrix: (3, 3) hand orientation
            - euler_angles: (3,) [roll, pitch, yaw] in radians
            - gripper_state: float in [0, 1]
            - palm_normal: (3,) palm facing direction
    """
    # position: use wrist (0) as end-effector reference
    position = landmarks_3d[0]

    # orientation: compute rotation matrix and palm normal
    rotation_matrix, palm_normal = compute_hand_orientation(landmarks_3d)

    # euler angles for easier interpretation
    euler_angles = rotation_matrix_to_euler(rotation_matrix)

    # gripper state from finger distances
    gripper_state = compute_gripper_state(landmarks_3d)

    return {
        'position': position,
        'rotation_matrix': rotation_matrix,
        'euler_angles': euler_angles,
        'gripper_state': gripper_state,
        'palm_normal': palm_normal,
    }


def batch_hand_to_gripper(landmarks_batch: np.ndarray) -> Dict[str, np.ndarray]:
    """
    vectorized version for processing multiple frames at once

    args:
        landmarks_batch: (N, 21, 3) array of landmarks across N frames

    returns:
        dict of batched outputs with shapes (N, ...)
    """
    N = landmarks_batch.shape[0]

    positions = np.zeros((N, 3))
    rotation_matrices = np.zeros((N, 3, 3))
    euler_angles = np.zeros((N, 3))
    gripper_states = np.zeros(N)
    palm_normals = np.zeros((N, 3))

    for i in range(N):
        result = hand_to_gripper_transform(landmarks_batch[i])
        positions[i] = result['position']
        rotation_matrices[i] = result['rotation_matrix']
        euler_angles[i] = result['euler_angles']
        gripper_states[i] = result['gripper_state']
        palm_normals[i] = result['palm_normal']

    return {
        'positions': positions,
        'rotation_matrices': rotation_matrices,
        'euler_angles': euler_angles,
        'gripper_states': gripper_states,
        'palm_normals': palm_normals,
    }


def gripper_velocity(positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    compute gripper velocity from position trajectory

    args:
        positions: (N, 3) positions over time
        timestamps: (N,) timestamps in microseconds

    returns:
        velocities: (N-1, 3) velocity vectors in m/s
    """
    # convert timestamps to seconds
    dt = np.diff(timestamps) / 1e6
    dt = np.maximum(dt, 1e-6)  # avoid division by zero

    # velocity = dx/dt using einsum for clean vectorization
    dpos = np.diff(positions, axis=0)
    velocities = np.einsum('ij,i->ij', dpos, 1.0 / dt)

    return velocities


def gripper_angular_velocity(euler_batch: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    compute angular velocity from euler angle trajectory

    args:
        euler_batch: (N, 3) euler angles [roll, pitch, yaw] over time
        timestamps: (N,) timestamps in microseconds

    returns:
        angular_velocities: (N-1, 3) angular velocity in rad/s
    """
    dt = np.diff(timestamps) / 1e6
    dt = np.maximum(dt, 1e-6)

    # handle angle wrapping for continuous derivatives
    deuler = np.diff(euler_batch, axis=0)
    deuler = np.arctan2(np.sin(deuler), np.cos(deuler))  # wrap to [-pi, pi]

    angular_velocities = np.einsum('ij,i->ij', deuler, 1.0 / dt)

    return angular_velocities
