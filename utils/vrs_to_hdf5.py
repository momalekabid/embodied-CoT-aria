#!/usr/bin/env python3
"""
vrs_to_hdf5.py

[DRAFT // WIP]
extends full HDF5 conversion for robomimic format from vrs.
probably need to run the right undistortion for camera before
outputs dataset ready for robomimic_rlds_dataset_builder.

"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

# import from vrs_to_viz
import sys

sys.path.insert(0, str(Path(__file__).parent))
from vrs_to_viz import VRSVisualizer, AriaMPSHandTracker, AriaCalibration, HAS_ARIA

# import timedomain for timestamps
try:
    from projectaria_tools.core.sensor_data import TimeDomain
except ImportError:
    TimeDomain = None


class VRSToHDF5Converter:
    """converts vrs episode to robomimic HDF5 format"""

    def __init__(
        self, vrs_path: str, mps_path: Optional[str] = None, output_hdf5: Optional[str] = None, demo_id: int = 0
    ):
        """
        args:
            vrs_path: path to .vrs file
            mps_path: path to aria mps output (optional)
            output_hdf5: where to save HDF5 (default: vrs_path.hdf5)
            demo_id: episode index in HDF5
        """
        self.vrs_path = vrs_path
        self.mps_path = mps_path
        self.demo_id = demo_id

        if output_hdf5 is None:
            output_hdf5 = str(Path(vrs_path).with_suffix(".hdf5"))
        self.output_hdf5 = output_hdf5

        # initialize visualizer (contains all extraction logic)
        self.viz = VRSVisualizer(vrs_path, mps_path)
        self.config = AriaCalibration()

        # metadata
        self.metadata = {
            "vrs_path": vrs_path,
            "mps_path": mps_path,
            "has_hand_tracking": mps_path is not None and HAS_ARIA,
        }

    def convert_episode(
        self,
        hand: str = "right",
        max_frames: Optional[int] = None,
        stride: int = 1,
        horizon: int = 10,
        step_size: int = 3,
    ) -> Dict:
        """
        convert full episode to HDF5

        args:
            hand: 'left', 'right', or 'bimanual'
            max_frames: limit frames for testing (None = all)
            stride: subsample frames
            horizon: action sequence length
            step_size: frames between action steps

        returns:
            metadata dict with summary
        """
        print(f"converting {self.vrs_path} to HDF5...")

        # get frame count
        if not HAS_ARIA or not self.viz.provider:
            print("error: projectaria_tools required")
            return None

        rgb_timestamps = self.viz.provider.get_timestamps_ns(self.viz.stream_id_rgb, TimeDomain.DEVICE_TIME)
        frame_count = len(rgb_timestamps)

        if max_frames:
            frame_count = min(frame_count, max_frames)

        print(f"total frames: {frame_count}")

        # create HDF5 file
        with h5py.File(self.output_hdf5, "w") as h5f:
            demo_grp = h5f.create_group(f"data/demo_{self.demo_id}")
            obs_grp = demo_grp.create_group("obs")

            # infer dimensions
            sample_frame = self.viz.get_rgb_frame(0)
            if sample_frame is None:
                print("error: could not load sample frame")
                return None

            # rectified image size
            rect_h, rect_w = self.config.RECT_HEIGHT, self.config.RECT_WIDTH

            # action dimension
            action_dim = 6 if hand == "bimanual" else 3

            # pre-allocate datasets
            obs_grp.create_dataset("image_primary", (frame_count, rect_h, rect_w, 3), dtype=np.uint8)

            if self.metadata["has_hand_tracking"]:
                obs_grp.create_dataset("ee_pose", (frame_count, action_dim), dtype=np.float32)

                demo_grp.create_dataset("actions", (frame_count, horizon, action_dim), dtype=np.float32)

            # optional masks (placeholder)
            obs_grp.create_dataset("image_mask", (frame_count, rect_h, rect_w), dtype=np.uint8)

            # fill datasets
            valid_frames = 0

            for t in tqdm(range(0, frame_count, stride), total=frame_count // stride):
                # get rectified image
                rgb = self.viz.get_rgb_frame(t)
                if rgb is None:
                    continue

                rect_img = self.viz.rectifier.rectify_image(rgb)
                obs_grp["image_primary"][valid_frames] = rect_img

                # get end-effector observation
                if self.metadata["has_hand_tracking"]:
                    ee_pos = self.viz.hand_tracker.get_hand_position_at_frame(
                        t, hand=hand.split("_")[0]  # "bimanual" -> "bi"
                    )

                    if ee_pos is not None:
                        # transform to camera frame
                        ee_cam = self.viz.projector.T_device_camera @ np.append(ee_pos, 1.0)
                        obs_grp["ee_pose"][valid_frames] = ee_cam[:action_dim]

                        # get future actions
                        trajectory = self.viz.compute_future_trajectory(
                            t, hand=hand.split("_")[0], horizon=horizon, step_size=step_size
                        )

                        if trajectory is not None and len(trajectory) == horizon:
                            # transform all future points to camera frame
                            actions = []
                            for fut_pos in trajectory:
                                fut_cam = self.viz.projector.T_device_camera @ np.append(fut_pos, 1.0)
                                actions.append(fut_cam[:action_dim])

                            demo_grp["actions"][valid_frames] = np.array(actions)

                valid_frames += 1

            # trim datasets to valid frames (in case of missing data)
            if valid_frames < frame_count:
                print(f"warning: only {valid_frames}/{frame_count} frames were valid")

                for key in obs_grp.keys():
                    obs_grp[key].resize((valid_frames,) + obs_grp[key].shape[1:])

                if "actions" in demo_grp:
                    demo_grp["actions"].resize((valid_frames,) + demo_grp["actions"].shape[1:])

            # add metadata
            demo_grp.attrs["num_samples"] = valid_frames
            demo_grp.attrs["episode_id"] = self.demo_id
            demo_grp.attrs["hand"] = hand
            demo_grp.attrs["horizon"] = horizon
            demo_grp.attrs["step_size"] = step_size

            # add dataset metadata
            h5f.attrs["dataset_name"] = Path(self.vrs_path).stem
            h5f.attrs["num_demos"] = 1
            h5f.attrs["has_hand_tracking"] = self.metadata["has_hand_tracking"]
            h5f.attrs["action_dim"] = action_dim
            h5f.attrs["image_height"] = rect_h
            h5f.attrs["image_width"] = rect_w

        print(f"saved HDF5 to: {self.output_hdf5}")

        return {
            "hdf5_path": self.output_hdf5,
            "num_samples": valid_frames,
            "frame_count": frame_count,
            "action_dim": action_dim,
            "hand": hand,
        }


class HDF5Info:
    """utility to inspect HDF5 structure"""

    @staticmethod
    def print_structure(hdf5_path: str, indent: int = 0):
        """recursively print HDF5 structure"""

        def _print_structure(group, indent=0):
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    print("  " * indent + f"{key}: {item.shape} {item.dtype}")
                elif isinstance(item, h5py.Group):
                    print("  " * indent + f"{key}/ ")
                    _print_structure(item, indent + 1)

        with h5py.File(hdf5_path, "r") as h5f:
            _print_structure(h5f)

    @staticmethod
    def get_summary(hdf5_path: str) -> Dict:
        """get dataset summary"""
        with h5py.File(hdf5_path, "r") as h5f:
            summary = {
                "dataset_name": h5f.attrs.get("dataset_name", "unknown"),
                "num_demos": h5f.attrs.get("num_demos", 0),
                "demos": {},
            }

            for demo_key in h5f["data"].keys():
                demo = h5f["data"][demo_key]
                summary["demos"][demo_key] = {
                    "num_samples": demo.attrs.get("num_samples", 0),
                    "hand": demo.attrs.get("hand", "unknown"),
                    "obs_keys": list(demo["obs"].keys()),
                    "action_shape": demo.get("actions", None).shape if "actions" in demo else None,
                }

            return summary


# ============================================================================
# MAIN
# ============================================================================


def main():
    """example usage"""
    from pathlib import Path as PathlibPath

    # find VRS file
    possible_vrs = [
        PathlibPath("/Users/mabid/Desktop/f25/mixedreality/example/First_Test.vrs"),
        PathlibPath("/Users/mabid/Desktop/f25/mixedreality/First_Test.vrs"),
    ]

    vrs_path = None
    for p in possible_vrs:
        if p.exists():
            vrs_path = str(p)
            break

    if not vrs_path:
        print("error: could not find First_Test.vrs")
        return

    print(f"found VRS file at: {vrs_path}")

    mps_path = None  # set if available

    # possible mps paths
    possible_mps = [
        PathlibPath(vrs_path).parent / "mps_First_Test_vrs",
        PathlibPath(vrs_path).parent / "mps",
    ]
    for p in possible_mps:
        if p.exists():
            mps_path = str(p)
            print(f"found MPS data at: {mps_path}")
            break

    # convert
    output_hdf5 = str(PathlibPath(vrs_path).with_suffix(".hdf5"))
    converter = VRSToHDF5Converter(vrs_path=vrs_path, mps_path=mps_path, output_hdf5=output_hdf5)

    result = converter.convert_episode(hand="right", max_frames=100, horizon=10, step_size=3)  # limit for testing

    if result:
        print(f"\nconversion successful!")
        print(f"output: {result['hdf5_path']}")
        print(f"frames: {result['num_samples']}")

        # inspect structure
        print("\nHDF5 structure:")
        HDF5Info.print_structure(result["hdf5_path"])

        # get summary
        print("\nHDF5 summary:")
        summary = HDF5Info.get_summary(result["hdf5_path"])
        import json

        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
