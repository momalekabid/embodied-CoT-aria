"""
shared utilities for hand tracking visualization and velocity computation.
used by both slam and rgb camera processing scripts.
"""

import numpy as np
import cv2
from projectaria_tools.core.calibration import CameraCalibration
from projectaria_tools.core.stream_id import StreamId


def get_camera_calibration(vrs_data_provider, stream_id: StreamId) -> CameraCalibration:
    """retrieve camera calibration for given stream id"""
    device_calibration = vrs_data_provider.get_device_calibration()
    stream_label = vrs_data_provider.get_label_from_stream_id(stream_id)
    return device_calibration.get_camera_calib(stream_label)


def project_3d_to_2d(point_3d, T_device_camera, camera_calib):
    """project 3d point in device frame to 2d pixel coordinates"""
    # transform to camera frame
    point_4d = np.append(point_3d, 1.0)
    point_camera = T_device_camera @ point_4d
    point_camera_3d = point_camera[:3]

    # check if point is in front of camera
    if point_camera_3d[2] <= 0:
        return None

    # project to 2d using camera calibration
    try:
        pixel_2d = camera_calib.project(point_camera_3d)
        u, v = int(pixel_2d[0]), int(pixel_2d[1])
        return (u, v)
    except:
        return None


def draw_hand_skeleton(frame, landmarks_2d, hand_label="right"):
    """draw all 21 hand landmarks with skeletal connections"""
    # hand skeleton connections (mps hand tracking uses 21 landmarks)
    connections = [
        # thumb (0-4)
        (0, 1), (1, 2), (2, 3), (3, 4),
        # index finger (5-8)
        (0, 5), (5, 6), (6, 7), (7, 8),
        # middle finger (9-12)
        (0, 9), (9, 10), (10, 11), (11, 12),
        # ring finger (13-16)
        (0, 13), (13, 14), (14, 15), (15, 16),
        # pinky (17-20)
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    base_color = (0, 0, 255) if hand_label == "right" else (255, 0, 0)

    # draw connections
    for start_idx, end_idx in connections:
        if (start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d) and
            landmarks_2d[start_idx] is not None and landmarks_2d[end_idx] is not None):
            cv2.line(frame, landmarks_2d[start_idx], landmarks_2d[end_idx], base_color, 2)

    # draw landmarks on top
    for i, pt in enumerate(landmarks_2d):
        if pt is not None:
            # wrist (index 5) and palm (index 20) get bigger circles
            if i == 5:  # wrist joint
                cv2.circle(frame, pt, 8, base_color, -1)
                cv2.circle(frame, pt, 10, (255, 255, 255), 2)
            elif i == 20:  # palm center
                cv2.circle(frame, pt, 7, base_color, -1)
                cv2.circle(frame, pt, 9, (255, 255, 255), 2)
            else:
                cv2.circle(frame, pt, 4, base_color, -1)

    return frame


def compute_velocity(positions, timestamps, window_size=3):
    """
    compute velocity from positions and timestamps.

    args:
        positions: array of shape (N, 3) - 3d positions over time
        timestamps: array of shape (N,) - timestamps in microseconds
        window_size: number of frames to use for velocity estimation

    returns:
        velocities: array of shape (N, 3) - velocities in m/s
    """
    velocities = np.zeros_like(positions)

    for i in range(len(positions)):
        if i < window_size:
            # not enough history, use forward difference
            if i < len(positions) - 1:
                dt = (timestamps[i + 1] - timestamps[i]) / 1e6  # convert to seconds
                if dt > 0:
                    velocities[i] = (positions[i + 1] - positions[i]) / dt
        else:
            # use centered difference over window
            dt = (timestamps[i] - timestamps[i - window_size]) / 1e6
            if dt > 0:
                velocities[i] = (positions[i] - positions[i - window_size]) / dt

    return velocities


def draw_velocity_axes(frame, right_vel, left_vel):
    """
    draw 3d velocity axes for both hands in corner of frame.
    displays vx, vy, vz as growing/shrinking arrows with values.
    """
    # position for right hand axes (top left)
    right_origin = (180, 80)
    # position for left hand axes (top left, below right hand)
    left_origin = (180, 200)

    scale = 75  # pixels per m/s

    # draw right hand velocity axes
    if right_vel is not None:
        vx, vy, vz = right_vel

        # label
        cv2.putText(
            frame,
            "RIGHT HAND",
            (right_origin[0] - 50, right_origin[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # vx axis (red)
        end_x = (int(right_origin[0] + vx * scale), right_origin[1])
        cv2.arrowedLine(frame, right_origin, end_x, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(
            frame,
            f"vx:{vx:.2f}",
            (right_origin[0] + 5, right_origin[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

        # vy axis (green)
        end_y = (right_origin[0], int(right_origin[1] + vy * scale))
        cv2.arrowedLine(frame, right_origin, end_y, (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(
            frame,
            f"vy:{vy:.2f}",
            (right_origin[0] - 70, right_origin[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        # vz axis (blue) - diagonal
        end_z = (int(right_origin[0] - vz * scale * 0.7), int(right_origin[1] - vz * scale * 0.7))
        cv2.arrowedLine(frame, right_origin, end_z, (255, 0, 0), 2, tipLength=0.3)
        cv2.putText(
            frame,
            f"vz:{vz:.2f}",
            (right_origin[0] - 70, right_origin[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )

    # draw left hand velocity axes
    if left_vel is not None:
        vx, vy, vz = left_vel

        # label
        cv2.putText(
            frame, "LEFT HAND", (left_origin[0] - 50, left_origin[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        # vx axis (red)
        end_x = (int(left_origin[0] + vx * scale), left_origin[1])
        cv2.arrowedLine(frame, left_origin, end_x, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(
            frame,
            f"vx:{vx:.2f}",
            (left_origin[0] + 5, left_origin[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

        # vy axis (green)
        end_y = (left_origin[0], int(left_origin[1] + vy * scale))
        cv2.arrowedLine(frame, left_origin, end_y, (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(
            frame,
            f"vy:{vy:.2f}",
            (left_origin[0] - 70, left_origin[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        # vz axis (blue) - diagonal
        end_z = (int(left_origin[0] - vz * scale * 0.7), int(left_origin[1] - vz * scale * 0.7))
        cv2.arrowedLine(frame, left_origin, end_z, (255, 0, 0), 2, tipLength=0.3)
        cv2.putText(
            frame,
            f"vz:{vz:.2f}",
            (left_origin[0] - 70, left_origin[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
        )

    return frame


def rotate_to_portrait(frame):
    """rotate frame 90 degrees counterclockwise to portrait orientation"""
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_3d_coords_to_portrait(coords_3d):
    """
    rotate 3d coordinates to portrait orientation.
    R = [[0,-1,0], [1,0,0], [0,0,1]]
    """
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return (R @ coords_3d.T).T
