"""
minimal utility for loading and using aria eye gaze data
"""

import numpy as np
import csv
from pathlib import Path
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection


class GazeLoader:
    """load and query eye gaze data from aria mps output"""

    def __init__(self, gaze_csv_path):
        self.data = []
        with open(gaze_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'tracking_timestamp_us': int(row['tracking_timestamp_us']),
                    'left_yaw_rads_cpf': float(row['left_yaw_rads_cpf']),
                    'right_yaw_rads_cpf': float(row['right_yaw_rads_cpf']),
                    'pitch_rads_cpf': float(row['pitch_rads_cpf']),
                    'depth_m': float(row['depth_m']) if row['depth_m'] else None,
                })

        self.timestamps = np.array([d['tracking_timestamp_us'] for d in self.data])
        print(f"loaded {len(self.data)} gaze samples")

    def get_gaze_at_timestamp(self, timestamp_us, tolerance_us=50000):
        time_diffs = np.abs(self.timestamps - timestamp_us)
        closest_idx = time_diffs.argmin()

        if time_diffs[closest_idx] > tolerance_us:
            return None

        row = self.data[closest_idx]

        return {
            'timestamp_us': row['tracking_timestamp_us'],
            'left_yaw': row['left_yaw_rads_cpf'],
            'right_yaw': row['right_yaw_rads_cpf'],
            'pitch': row['pitch_rads_cpf'],
            'depth': row['depth_m'],
            'combined_yaw': (row['left_yaw_rads_cpf'] + row['right_yaw_rads_cpf']) / 2.0
        }

    def yaw_pitch_to_vector(self, yaw, pitch):
        """
        convert yaw/pitch angles to 3d gaze direction vector in cpf frame

        args:
            yaw: yaw angle in radians
            pitch: pitch angle in radians

        returns:
            numpy array [x, y, z] representing gaze direction (unit vector)
        """
        # based on aria coordinate conventions
        # z is forward, x is right, y is down
        x = np.sin(yaw) * np.cos(pitch)
        y = np.sin(pitch)
        z = np.cos(yaw) * np.cos(pitch)

        vec = np.array([x, y, z])
        return vec / np.linalg.norm(vec)  # normalize

    def get_gaze_point_3d(self, gaze_data):
        """
        get 3d gaze point in cpf frame

        args:
            gaze_data: dict from get_gaze_at_timestamp()

        returns:
            numpy array [x, y, z] of gaze point, or None if depth not available
        """
        if gaze_data is None or gaze_data['depth'] is None:
            return None

        # use combined yaw for backwards compatibility
        direction = self.yaw_pitch_to_vector(
            gaze_data['combined_yaw'],
            gaze_data['pitch']
        )

        # scale by depth to get 3d point
        return direction * gaze_data['depth']

    def get_gaze_direction(self, gaze_data):
        """
        get gaze direction vector (useful even without depth)

        args:
            gaze_data: dict from get_gaze_at_timestamp()

        returns:
            numpy array [x, y, z] direction vector
        """
        if gaze_data is None:
            return None

        return self.yaw_pitch_to_vector(
            gaze_data['combined_yaw'],
            gaze_data['pitch']
        )

    def project_gaze_to_2d(self, gaze_data, camera_calib, T_device_cpf):
        """
        project gaze point to 2d image coordinates

        args:
            gaze_data: dict from get_gaze_at_timestamp()
            camera_calib: projectaria camera calibration object
            T_device_cpf: 4x4 transform from cpf to device frame

        returns:
            (u, v) pixel coordinates or None
        """
        gaze_point_3d = self.get_gaze_point_3d(gaze_data)
        if gaze_point_3d is None:
            return None

        # transform from cpf to device frame
        point_4d = np.append(gaze_point_3d, 1.0)
        point_device = T_device_cpf @ point_4d

        # project to camera
        try:
            pixel_2d = camera_calib.project(point_device[:3])
            return (int(pixel_2d[0]), int(pixel_2d[1]))
        except:
            return None


def filter_objects_by_gaze(objects, gaze_point_2d, max_distance=100):
    """
    filter objects based on proximity to gaze point

    args:
        objects: list of dicts with 'bbox' key containing [x1, y1, x2, y2]
        gaze_point_2d: (u, v) pixel coordinates of gaze
        max_distance: max pixel distance from gaze to object center

    returns:
        list of objects sorted by distance to gaze point
    """
    if gaze_point_2d is None:
        return objects

    gaze_u, gaze_v = gaze_point_2d

    # compute distance from gaze to each object center
    scored_objects = []
    for obj in objects:
        bbox = obj['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        dist = np.sqrt((center_x - gaze_u)**2 + (center_y - gaze_v)**2)

        if dist <= max_distance:
            scored_objects.append((dist, obj))

    # sort by distance (closest first)
    scored_objects.sort(key=lambda x: x[0])

    return [obj for _, obj in scored_objects]


def is_point_in_bbox(point_2d, bbox):
    """
    check if 2d point is inside bounding box

    args:
        point_2d: (u, v) pixel coordinates
        bbox: [x1, y1, x2, y2]

    returns:
        bool
    """
    u, v = point_2d
    x1, y1, x2, y2 = bbox
    return x1 <= u <= x2 and y1 <= v <= y2


class GazeMPSLoader:
    """
    load and project eye gaze using mps data provider
    uses projectaria_tools native gaze projection (more accurate than csv-based approach)
    """

    def __init__(self, mps_path, use_general_gaze=True):
        """
        args:
            mps_path: path to mps output folder (contains eye_gaze subfolder)
            use_general_gaze: if true, use general eye gaze, else personalized
        """
        self.use_general_gaze = use_general_gaze

        # load mps data
        mps_data_paths_provider = mps.MpsDataPathsProvider(mps_path)
        mps_data_paths = mps_data_paths_provider.get_data_paths()
        self.mps_data_provider = mps.MpsDataProvider(mps_data_paths)

        # verify eye gaze data exists
        assert self.mps_data_provider.has_general_eyegaze(), "no eye gaze data in mps"
        print(f"loaded gaze mps data from {mps_path}")

    def get_gaze_projection(self, timestamp_ns, rgb_stream_label, device_calibration, camera_calibration, depth_m=1.0):
        """
        get gaze projection for a given timestamp

        args:
            timestamp_ns: timestamp in nanoseconds
            rgb_stream_label: camera stream label (e.g., "camera-rgb")
            device_calibration: device calibration object
            camera_calibration: camera calibration object
            depth_m: gaze depth in meters (default 1.0)

        returns:
            (x, y) pixel coordinates or None if no gaze data
        """
        # get eye gaze at timestamp
        if self.use_general_gaze:
            eye_gaze = self.mps_data_provider.get_general_eyegaze(timestamp_ns)
        else:
            eye_gaze = self.mps_data_provider.get_personalized_eyegaze(timestamp_ns)

        if eye_gaze is None:
            return None

        # use depth from eye gaze if available, otherwise use provided default
        actual_depth = eye_gaze.depth if eye_gaze.depth else depth_m

        # project gaze to 2d using projectaria utilities
        gaze_projection = get_gaze_vector_reprojection(
            eye_gaze,
            rgb_stream_label,
            device_calibration,
            camera_calibration,
            actual_depth,
        )

        return gaze_projection


if __name__ == "__main__":
    # example usage
    gaze_csv = "/Users/mabid/Desktop/f25/mixedreality/ecot2/embodied-CoT-aria/utils/Aria/Bottle_v2/mps_Bottle_v2_vrs/eye_gaze/general_eye_gaze.csv"

    loader = GazeLoader(gaze_csv)

    # get gaze at specific timestamp
    timestamp = 210557949  # first timestamp in csv
    gaze = loader.get_gaze_at_timestamp(timestamp)

    print(f"\ngaze at {timestamp}:")
    print(f"  yaw: {gaze['combined_yaw']:.3f} rad")
    print(f"  pitch: {gaze['pitch']:.3f} rad")
    print(f"  depth: {gaze['depth']:.3f} m")

    # get 3d gaze point
    point_3d = loader.get_gaze_point_3d(gaze)
    print(f"\n3d gaze point in cpf: {point_3d}")

    # get direction vector
    direction = loader.get_gaze_direction(gaze)
    print(f"gaze direction: {direction}")

    # example: filter objects by gaze
    example_objects = [
        {'name': 'bottle', 'bbox': [100, 100, 200, 200]},
        {'name': 'cup', 'bbox': [300, 150, 350, 250]},
        {'name': 'hand', 'bbox': [150, 120, 180, 180]},
    ]

    # assuming gaze projects to (160, 140)
    example_gaze_2d = (160, 140)
    filtered = filter_objects_by_gaze(example_objects, example_gaze_2d, max_distance=50)

    print(f"\nobjects near gaze point {example_gaze_2d}:")
    for obj in filtered:
        print(f"  - {obj['name']}")
