import json
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.calibration import distort_by_calibration
from tqdm import tqdm
import numpy as np
import argparse


from sys import path as sys_path
import os
from os import path

# Add the parent directory (or any path you need) to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys_path.append(parent_dir)

from utils.hand_tracking_utils import get_camera_calibration
from utils.angles import rotation_matrix_to_euler

import cv2
import platform


IMAGE_SIZE = (1408, 1408, 3)
IMAGE_SIZE_DOWNSAMPLED = (224, 224, 3)

# RLDS Step class
class Step:
    def __init__(self) -> None:

        # this dictionary will hold all information for the step
        self._information: dict = {
            # is_first and is_last is currently not used, because the RLDS dataset formatter infers this from the episode length
            # could be deleted in future versions because the aria_dataset builder does not require these fields (creates them automatically)
            "is_first": False,
            "is_last": False,

            # tracks time since episode start in nanoseconds
            "time_since_episode_start_ns": None,

            "language_instruction": "dummy instruction",

            "image": None,
            
            # action are of shape (2, 7): [[hand_vel_left(3), hand_rot_left(3), hand_open_left(1)],
            #                             [hand_vel_right(3), hand_rot_right(3), hand_open_right(1)]]

            "action": np.zeros((2, 7), dtype=np.float32),

            # observation state are of shape (2, 7): [[hand_pos_left(3), hand_rot_left(3), hand_open_left(1)],
            #                                   [hand_pos_right(3), hand_rot_right(3), hand_open_right(1)]]
            "state": np.zeros((2, 7), dtype=np.float32)

        }
    
    def set_is_first(self) -> None:
        self._information["is_first"] = True
    
    def set_is_last(self) -> None:
        self._information["is_last"] = True

    def set_observation_time(self, value) -> None:
        # self._observations["time_since_episode_start_ns"] = value
        self._information["time_since_episode_start_ns"] = value
    
    def get_observation_time(self):
        return self._information["time_since_episode_start_ns"]
        
    def set_language_instruction(self, annotations: str) -> None:
        self._information["language_instruction"] = annotations

    def get_language_instruction(self) -> str:
        return self._information["language_instruction"]
        
    def get_is_first(self) -> bool:
        return self._information["is_first"]
        
    def get_is_last(self) -> bool:
        return self._information["is_last"]
    
    # set hand positions and velocities
    def set_hand_pos_and_speed(self, hand_pos_left: list[float], hand_pos_right: list[float], hand_vel_left: list[float], hand_vel_right: list[float]) -> None:
        self._information["state"][0, 0:3] = np.array(hand_pos_left, dtype=np.float32)
        self._information["state"][1, 0:3] = np.array(hand_pos_right, dtype=np.float32)
        self._information["action"][0, 0:3] = np.array(hand_vel_left, dtype=np.float32)
        self._information["action"][1, 0:3] = np.array(hand_vel_right, dtype=np.float32)

    def set_hand_open_states(self, hand_open_left: bool, hand_open_right: bool) -> None:
        self._information["state"][0, 6] = 1.0 if hand_open_left else 0.0
        self._information["state"][1, 6] = 1.0 if hand_open_right else 0.0
    
    def set_hand_rotation_states(self, hand_rot_left: list[float], hand_rot_right: list[float]) -> None:
        self._information["state"][0, 3:6] = np.array(hand_rot_left, dtype=np.float32)
        self._information["state"][1, 3:6] = np.array(hand_rot_right, dtype=np.float32)
    
    def set_hand_rotation_velocities(self, hand_rot_vel_left: list[float], hand_rot_vel_right: list[float]) -> None:
        self._information["action"][0, 3:6] = np.array(hand_rot_vel_left, dtype=np.float32)
        self._information["action"][1, 3:6] = np.array(hand_rot_vel_right, dtype=np.float32)
    
    def set_hand_open_changes(self, hand_open_change_left: bool, hand_open_change_right: bool) -> None:
        self._information["action"][0, 6] = 1.0 if hand_open_change_left else 0.0
        self._information["action"][1, 6] = 1.0 if hand_open_change_right else 0.0

    def set_image(self, image_array: np.ndarray) -> None:
        # if image_array.shape != IMAGE_SIZE:
        #     raise ValueError(f"Image array must have shape {IMAGE_SIZE}, but got {image_array.shape}")
        # self.image = image_array
        image_resized = cv2.resize(image_array, (IMAGE_SIZE_DOWNSAMPLED[0], IMAGE_SIZE_DOWNSAMPLED[1]))
        self._information["image"] = image_resized
    
    def get_image(self) -> np.ndarray:
        # return self._image
        return self._information["image"]
    
    def return_information_dict(self) -> dict:
        return self._information


# RLDS Episode class
# The episode fields are currently not used in the conversion to rlds.
class Episode:
    def __init__(self, episode_id: str, agent_id: str) -> None:
        self.episode_id: str = episode_id
        self.agent_id: str = agent_id
        self.invalid: bool = False
        self.description: str = ""
        self.steps: list[Step] = []

    def mark_invalid(self) -> None:
        self.invalid = True

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def save_to_np(self, save_path: str) -> None:
        episode_data = {
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "invalid": self.invalid,
            "task_description": self.description,
            "steps": [step.return_information_dict() for step in self.steps],
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


        # np.save(save_path, [step.return_information_dict() for step in self.steps])
        np.save(save_path, episode_data)


        print(f"Episode {self.episode_id} saved to {save_path}")


    def set_description(self, description: str) -> None:
        self.description = description



class VrsToRldsNpyConverter:
    # expects 
    #   - paths to folders containing the extracted VRS data hand velocities data,
    #   - list of timestamps (ns) that specify the start of each episode
    def __init__(self, vrs_data_path: str, vrs_file_name: str,processed_data_path : str,episode_name : str, episodes_timestamps: list[int] = [0]) -> None:
        self.vrs_data_path = vrs_data_path
        self.processed_data_path = processed_data_path
        self.episodes = []
        self.episodes_timestamps = episodes_timestamps
        self.hand_data_left, self.hand_data_right = self.restructure_hand_velocities()
        self.provider = data_provider.create_vrs_data_provider(path.join(vrs_data_path, vrs_file_name))
        self.speech_annotations = self.load_speech_annotations(processed_data_path + "speech_data.json")
        self.episode_name = episode_name
        self.hand_rot_open_states, self.hand_rot_open_vel = self.load_hand_rotation_and_open_states()

        self._rgb_stream_id = StreamId("214-1")
        self._rgb_camera_label = "rgb_camera"


    def undistort_image(self, image_distorted) -> np.ndarray:
        rgb_camera_calibration = get_camera_calibration(self.provider, self._rgb_stream_id)
        focal_lengths = rgb_camera_calibration.get_focal_lengths()
        image_size = rgb_camera_calibration.get_image_size()

        # create pinhole (undistorted) calibration
        pinhole_calib = calibration.get_linear_camera_calibration(image_size[0], image_size[1], focal_lengths[0])
        
        # undistort image
        image_undistorted = distort_by_calibration(
            image_distorted,
            pinhole_calib,
            rgb_camera_calibration
            
        )

        # rotate image to correct orientation
        image_undistorted = cv2.rotate(image_undistorted, cv2.ROTATE_90_CLOCKWISE)

        return image_undistorted

    # REQUIRES: 
    # - speech_data_path: path to json file containing speech annotations
    # - speech data to have the structure: 
                # [
                #     [
                #         [
                #         starttime_ns,
                #         endtime_ns
                #         ],
                #         "annotation text"
                #     ],
                # ]
    # ENSURES: 
    # - returns a dictionary mapping timestamps (us) to annotations (str)

    def load_speech_annotations(self, speech_data_path: str) -> dict:
        with open(speech_data_path, "r") as speech_file:
            speech_data = json.load(speech_file)
        
        annotations_dict = {}
        annotations_dict = {"General Task": speech_data[1][1]}
        for i, entry in enumerate(speech_data[2:]):
            timestamp_us = int(entry[0][0]/1000) # convert from ns to us
            annotation = entry[1]
            annotations_dict[timestamp_us] = annotation
        
        return annotations_dict

    def load_hand_rotation_and_open_states(self) -> dict:
        hand_rot_open_states_path = self.processed_data_path + "hand_rot_open_states.json"
        with open(hand_rot_open_states_path, "r") as hros_file:
            hand_rot_open_states = json.load(hros_file)
        
        hand_rot_open_changes = self.processed_data_path + "hand_rot_open_vel.json"
        with open(hand_rot_open_changes, "r") as hroc_file:
            hand_rot_open_changes = json.load(hroc_file)
        
        return hand_rot_open_states, hand_rot_open_changes

    # given a timestamp (ns), find the corresponding rgb frame index    
    def match_timestamp_to_rgb_frame_id(self, timestamp: int) -> int:
        all_frames_data = self.processed_data_path + "all_frames.json"

        image_data = self.provider.get_image_data_by_ns_timestamp(StreamId("214-1"), timestamp)
        return image_data.frame_index
            
    
    
    def process_episodes(self) -> None:

        count = 0

        start_idx = 0
        end_idx = 0
        # fills the episode_frame_idxs with the start and end frame indices for each episode (start inclusive; end exclusive)
        for i, episode_timestamp in enumerate(self.episodes_timestamps):
            time_domain = TimeDomain.DEVICE_TIME  # query data based on DEVICE_TIME
            option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time   

            # set the current base timestamp (start of episode), used to normalize timestamps within episode
            cur_base_timestamp = int(episode_timestamp/1000)

            if i == len(self.episodes_timestamps) - 1:
                # episode_frame_idxs[i] = (start_idx, len(self.episodes_timestamps))
                start_idx = end_idx
                end_idx = self.provider.get_num_data(self._rgb_stream_id)
            else:
                start_idx = end_idx
                end_idx = self.match_timestamp_to_rgb_frame_id(self.episodes_timestamps[i+1])
                # episode_frame_idxs[i] = (start_idx, end_idx)
            
            episode_id = f"episode_{self.episode_name}_{i}"
            agent_id = "human"

            cur_episode = Episode(episode_id, agent_id)
            cur_episode.set_description(self.speech_annotations.pop("General Task", ""))
            
            # prepare list of speech annotation timestamps (us) for easier lookup
            speech_annotation_timestamps = [0] + [ts for ts in self.speech_annotations.keys()]
            speech_annotation_timestamps = sorted(speech_annotation_timestamps)

            # use index to track current speech annotation
            speech_annotation_idx = 0

            for frame_idx in range(start_idx, end_idx):
                
                # create step object
                cur_step = Step()

                image_data = self.provider.get_image_data_by_index(self._rgb_stream_id, frame_idx)

                if image_data is None:
                    continue
                
                # safety check that frame_idx is within bounds
                if frame_idx > self.provider.get_num_data(self._rgb_stream_id):
                    raise ValueError("Frame index exceeds number of frames in stream.")
                
                # get image (distorted) as numpy array and its timestamp
                step_image_distorted = image_data[0].to_numpy_array()
                step_image_undistorted = self.undistort_image(step_image_distorted)
                cur_step.set_image(step_image_undistorted)

                step_timestamp = int(image_data[1].capture_timestamp_ns /1000 )
                # normalize timestamp to episode start
                step_normalized_timestamp = step_timestamp - cur_base_timestamp

                cur_step.set_observation_time(step_normalized_timestamp)

                # logic for setting the language instruction based on timestamp
                # speech annotations are taken from the most recent timestamp that is <= current step timestamp
                if speech_annotation_idx < len(speech_annotation_timestamps)-1:
                    if step_timestamp < speech_annotation_timestamps[speech_annotation_idx + 1]:
                        if speech_annotation_idx==0:
                            cur_step.set_language_instruction("")
                        else:
                            cur_step.set_language_instruction(self.speech_annotations.get(speech_annotation_timestamps[speech_annotation_idx]))
                    else:
                        speech_annotation_idx += 1
                        cur_step.set_language_instruction(self.speech_annotations.get(speech_annotation_timestamps[speech_annotation_idx]))
                else:
                    cur_step.set_language_instruction(self.speech_annotations.get(speech_annotation_timestamps[speech_annotation_idx]))
                
                if cur_step.get_language_instruction() is None:
                    print(f"Step timestamp: {step_timestamp}, Annotation timestamp: {speech_annotation_timestamps[speech_annotation_idx]}, Annotation: {cur_step.get_language_instruction()}"
                      )
                    raise ValueError(f"Warning: No language instruction found for timestamp {step_timestamp}. Setting to empty string.")

             
                # get hand data for current timestamp
                left_hand_data = self.hand_data_left.get(step_timestamp)
                right_hand_data = self.hand_data_right.get(step_timestamp)

                

                # handle missing hand data
                if(left_hand_data is None and right_hand_data is None):
                    count += 1
                    continue
                elif (left_hand_data is None):
                    left_hand_data = {
                        "positions_3d_m": [0.0, 0.0, 0.0],
                        "velocities_3d_ms": [0.0, 0.0, 0.0]
                    }
                elif (right_hand_data is None):
                    right_hand_data = {
                        "positions_3d_m": [0.0, 0.0, 0.0],
                        "velocities_3d_ms": [0.0, 0.0, 0.0]
                    }

                # set hand data in step
                cur_step.set_hand_pos_and_speed(
                    hand_pos_left=left_hand_data["positions_3d_m"],
                    hand_pos_right=right_hand_data["positions_3d_m"],
                    hand_vel_left=left_hand_data["velocities_3d_ms"],
                    hand_vel_right=right_hand_data["velocities_3d_ms"]
                )

                # set hand rotation and open states (state and action/change)
                cur_hand_rot_open_state = self.hand_rot_open_states[str(step_timestamp)]
                cur_hand_rot_open_vel = self.hand_rot_open_vel[str(step_timestamp)]

                # convert rotation matrices to euler angles [roll, pitch, yaw]
                L_rot_matrix = np.array(cur_hand_rot_open_state["L_rot"])
                R_rot_matrix = np.array(cur_hand_rot_open_state["R_rot"])
                L_euler = rotation_matrix_to_euler(L_rot_matrix) if L_rot_matrix.any() else np.zeros(3)
                R_euler = rotation_matrix_to_euler(R_rot_matrix) if R_rot_matrix.any() else np.zeros(3)

                cur_step.set_hand_rotation_states(
                    hand_rot_left=L_euler.tolist(),
                    hand_rot_right=R_euler.tolist()
                )

                # compute euler angle velocities (angular velocities)
                # for now, use finite differences on the rotation matrices (simplified)
                # todo: proper angular velocity computation from rotation matrices
                L_rot_vel_matrix = np.array(cur_hand_rot_open_vel["L_rot_vel"])
                R_rot_vel_matrix = np.array(cur_hand_rot_open_vel["R_rot_vel"])

                # approximate euler angle velocities from matrix differences
                # proper method would use angular velocity formula, but this is acceptable for learning
                L_euler_vel = L_rot_vel_matrix[2] if L_rot_vel_matrix.shape == (3, 3) else np.zeros(3)
                R_euler_vel = R_rot_vel_matrix[2] if R_rot_vel_matrix.shape == (3, 3) else np.zeros(3)

                cur_step.set_hand_rotation_velocities(
                    hand_rot_vel_left=L_euler_vel.tolist(),
                    hand_rot_vel_right=R_euler_vel.tolist()
                )

                cur_step.set_hand_open_states(
                    hand_open_left=cur_hand_rot_open_state["L_open"],
                    hand_open_right=cur_hand_rot_open_state["R_open"]
                )

                cur_step.set_hand_open_changes(
                    hand_open_change_left=cur_hand_rot_open_vel["L_grip_change"],
                    hand_open_change_right=cur_hand_rot_open_vel["R_grip_change"]
                )

                # add step to episode
                cur_episode.add_step(cur_step)

            cur_episode.steps[0].set_is_first()
            cur_episode.steps[-1].set_is_last()
            # add finished episode to list
            self.episodes.append(cur_episode)
            



        for ep in self.episodes:
            print(f"Episode ID: {ep.episode_id}, Number of steps: {len(ep.steps)}")

        # episode = Episode(episode_id, agent_id)
        # self.episodes.append(episode)
        # return episode


    def restructure_hand_velocities(self)-> list[dict]:
        hand_velocities_files = [self.processed_data_path + "left_hand_velocity.json", self.processed_data_path + "right_hand_velocity.json"]
        restructured = {}
        both_hands = []
        for file_path in hand_velocities_files:
            with open(file_path, "r") as hv_file:
                hv_data = json.load(hv_file)
                # transform the structure from 
                # {
                #   "timestamps_us": [ts1, ts2, ...],
                #   "positions_3d_m": [[p1_x, p1_y, p1_z], [p2_x, p2_y, p2_z], ...],
                #   "velocities_3d_ms": [[p1_x, p1_y, p1_z], [p2_x, p2_y, p2_z], ...],
                #   "velocity_magnitudes_ms": [vm1, vm2, vm3, ...]
                # }
                # to
                # {
                #   "ts1": {
                #      "positions_3d_m": [p1_x, p1_y, p1_z],
                #      "velocities_3d_ms": [p1_x, p1_y, p1_z],
                #      "velocity_magnitudes_ms": vm1
                #   },
                #
                #   "ts2": {
                #      "positions_3d_m": [p2_x, p2_y, p2_z],
                #      "velocities_3d_ms": [p2_x, p2_y, p2_z],
                #      "velocity_magnitudes_ms": vm2
                #   },
                # ...
                # }
            for i, ts in enumerate(hv_data["timestamps_us"]):
                restructured[ts] = {
                    "positions_3d_m": hv_data["positions_3d_m"][i],
                    "velocities_3d_ms": hv_data["velocities_3d_ms"][i],
                    "velocity_magnitudes_ms": hv_data["velocity_magnitudes_ms"][i]
                }
            both_hands.append(restructured)
            restructured = {}

        return both_hands  # [left_hand_dict, right_hand_dict]
            
            

    # TODO: implement actually differing between train and val episodes
    def save_episodes(self, save_dir: str) -> None:
        for episode in self.episodes:
            print(f"Saving episode {episode.episode_id} with {len(episode.steps)} steps.")
            save_path_train = path.join(save_dir, f"train/{episode.episode_id}.npy")
            save_path_val = path.join(save_dir, f"val/{episode.episode_id}.npy")
            episode.save_to_np(save_path_train)
            episode.save_to_np(save_path_val)
            print(f"Episode {episode.episode_id} saved to {save_path_train} and {save_path_val}.")



# NOTE: The conversion from vrs to npy must be run with numpy version 1.24.3 (some others might also work, not >=2.0.0 though!), otherwise RLDS dataset formatter will not be able to read the saved npy files.
# Not using a compatible numpy version will lead to weird errors when loading the npy files with RLDS dataset formatter, e.g. numpy core not found.

# NOTE: Update the paths below to your local setup before running!
# Requires:
# - processed VRS data (using /utils/process_fisheye_with_hands_voice.py) in a folder structure:
#   - recording_name/placeholder.json
# where placeholder is to be replaced with all_frames, hand_rot_open_states, left_hand_velocity, right_hand_velocity, speech_data
# - VRS file in within the following folder structure:
#   - allrecording_names/recording_name/recording_name.vrs
def main():
    parser = argparse.ArgumentParser(description="Convert VRS data to RLDS npy format.")
    parser.add_argument("--recording_names", nargs="+", type=str, help="List of recording names to process.", required=False)
    parser.add_argument("--vrs_data_base_path", type=str, help="Base path to VRS data folders. Each folder must be named as its corresponding recording name.", required=False)
    parser.add_argument("--processed_data_base_path", type=str, help="Base path to processed data folders. Each folder must be named as its corresponding recording name.", required=False)
    parser.add_argument("--save_base_path", type=str, help="Base path to save RLDS npy files. Each folder will be named as its corresponding recording name.", required=False)
    args = parser.parse_args()

    # Set default values if values not provided via command line arguments
    # list of episode (recording) names to process
    # recording_names = ["Banana_v1", "Banana_v2", "Bottle_v1", "Bottle_v2", "Orange_v1", "Sponge_v1", "Sponge_v2", "Stack_bowls_in_drawer_v1", "Pot_into_pot_corrective_behavior_v1"]
    # recording_names = ["Stack_bowls_in_drawer_v1", "Pot_into_pot_corrective_behavior_v1"]
    recording_names = ["Banana_v1"]
    shared_path_vrs_data = f"/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data_1/"
    processed_data_base_path = f"/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/utils/output"
    save_rlds_npy_data_base_path = f"/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_rlds_builder-main/aria_dataset/data_new_format/"

    # Check if values through flags were provided
    if args.recording_names is not None:
        print("--recording_names flag was used.")
        recording_names = args.recording_names
    if args.vrs_data_base_path is not None:
        print("--vrs_data_base_path flag was used.")
        shared_path_vrs_data = args.vrs_data_base_path
    if args.processed_data_base_path is not None:
        print("--processed_data_base_path flag was used.")
        processed_data_base_path = args.processed_data_base_path
    if args.save_base_path is not None:
        print("--save_base_path flag was used.")
        save_rlds_npy_data_base_path = args.save_base_path


    
    for name in tqdm(recording_names):
        print(f"Processing episode: {name}")

        # IMPORTANT: Update these paths to your local setup
        path_vrs_data = f"{shared_path_vrs_data}{name}/"
        vrs_file_name = f"{name}.vrs"
        processed_data = f"{processed_data_base_path}/{name}_output/"
        path_save_rlds_data = f"{save_rlds_npy_data_base_path}{name}/"
        
        # Define base paths for Windows and Linux
        windows_base_path = "C:"
        linux_base_path = "/mnt/c"

        # Initialize converter and process/save episodes based on the operating system
        if platform.system() == "Windows":
            converter = VrsToRldsNpyConverter(
                vrs_data_path= path.join(windows_base_path,path_vrs_data),
                vrs_file_name=vrs_file_name,
                processed_data_path= path.join(windows_base_path,processed_data),
                episode_name = name
            )
            converter.process_episodes()
            converter.save_episodes(save_dir=
                                    path.join(windows_base_path,path_save_rlds_data)
            )
        elif platform.system() == "Linux":
            print("Running on Linux system.")
            print("Hand velocities path:", path.join(linux_base_path,processed_data))
            converter = VrsToRldsNpyConverter(
                vrs_data_path= linux_base_path + path_vrs_data,
                
                vrs_file_name=vrs_file_name,
                processed_data_path= linux_base_path + processed_data,
                episode_name = name
            )
            converter.process_episodes()
            converter.save_episodes(save_dir= linux_base_path + path_save_rlds_data
            )



if __name__ == "__main__":
    main()