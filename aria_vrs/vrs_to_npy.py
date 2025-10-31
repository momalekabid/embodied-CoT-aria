import jsons
import json
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from projectaria_tools.core.calibration import distort_by_calibration
from tqdm import tqdm
import numpy as np

from sys import path as sys_path
import os


# Add the parent directory (or any path you need) to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys_path.append(parent_dir)

from utils.hand_tracking_utils import (
    get_camera_calibration,
    project_3d_to_2d,
    draw_hand_skeleton,
    draw_velocity_axes,
    compute_velocity,
)
from os import path
import cv2
import platform


IMAGE_SIZE = (1408, 1408, 3)

# RLDS Step class
class Step:
    def __init__(self) -> None:

        # this dictionary will hold all information for the step
        self._information: dict = {
            # is_first and is_last is currently not used, because the RLDS dataset formatter infers this from the episode length
            "is_first": False,
            "is_last": False,


            "time_since_episode_start_ns": None,

            #TODO: set actual language instruction
            "language_instruction": "dummy instruction",
            # currently only images of shape (1408, 1408, 3) are supported
            "image": None,
            
            # action are of shape (2, 3): [hand_vel_left(3), hand_vel_right(3)]
            "action": np.zeros((2, 3), dtype=np.float32),

            # state is of shape (2, 3): [hand_pos_left(3), hand_pos_right(3)]
            "state": np.zeros((2, 3), dtype=np.float32)

            

        }

        # self._is_first = is_first
        # self._is_last = is_last
        # self._observations = {
        #     "time_since_episode_start_ns": None,
        #     "language_instruction": None,
        # }
        # self._image = None  # to be filled with numpy array representing the image
        # self._action = {
        #     "hand_pos_left": None,
        #     "hand_pos_right": None,
        #     "hand_vel_left": None,
        #     "hand_vel_right": None
        # }

    
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
    
    def set_hand_data(self, hand_pos_left: list[float], hand_pos_right: list[float], hand_vel_left: list[float], hand_vel_right: list[float]) -> None:
        self._information["state"][0, :] = np.array(hand_pos_left, dtype=np.float32)
        self._information["state"][1, :] = np.array(hand_pos_right, dtype=np.float32)
        self._information["action"][0, :] = np.array(hand_vel_left, dtype=np.float32)
        self._information["action"][1, :] = np.array(hand_vel_right, dtype=np.float32)


    def set_image(self, image_array: np.ndarray) -> None:
        if image_array.shape != IMAGE_SIZE:
            raise ValueError(f"Image array must have shape {IMAGE_SIZE}, but got {image_array.shape}")
        # self.image = image_array
        self._information["image"] = image_array
    
    def get_image(self) -> np.ndarray:
        # return self._image
        return self._information["image"]
    
    def return_information_dict(self) -> dict:
        return self._information


# RLDS Episode class
class Episode:
    def __init__(self, episode_id: str, agent_id: str) -> None:
        self.episode_id: str = episode_id
        self.agent_id: str = agent_id
        self.invalid: bool = False
        self.steps: list[Step] = []

    def mark_invalid(self) -> None:
        self.invalid = True

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    # TODO: delete - this is a tempory test function
    def create_fake_episode(self,path):
        episode = []
        for step in range(1):
            episode.append({
                'image': np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8),
                'wrist_image': np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8),
                'state': np.asarray(np.random.rand(7,), dtype=np.float32),
                'action': np.asarray(np.random.rand(7,), dtype=np.float32),
                'language_instruction': 'dummy instruction',
            })
        print(episode)
        np.save(path, episode)



    def save_to_np(self, save_path: str) -> None:
        episode_data = {
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "invalid": self.invalid,
            "steps": [step.return_information_dict() for step in self.steps],
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


        np.save(save_path, [step.return_information_dict() for step in self.steps])

        # self.create_fake_episode(save_path)

        

        print(f"Episode {self.episode_id} saved to {save_path}")



class VrsToRldsConverter:
    # expects 
    #   - paths to folders containing the extracted VRS data hand velocities data,
    #   - list of timestamps (ns) that specify the start of each episode
    def __init__(self, vrs_data_path: str, vrs_file_name: str,hand_velocities_data_path : str, episodes_timestamps: list[int] = [0]) -> None:
        self.vrs_data_path = vrs_data_path
        self.hand_velocities_data_path = hand_velocities_data_path
        self.episodes = []
        self.episodes_timestamps = episodes_timestamps
        self.hand_data_left, self.hand_data_right = self.restructure_hand_velocities()
        self.provider = data_provider.create_vrs_data_provider(path.join(vrs_data_path, vrs_file_name))


        self._rgb_stream_id = StreamId("214-1")
        self._rgb_camera_label = "rgb_camera"




    # ------------------- START: OUTDATED -------------------
    # this function peeks into the VRS metadata file and prints out the metadata objects
    # def peek_into_vrs_metadata(self, file_name: str) -> None:

    #     vrs_file_path = f"{self.vrs_data_path}/metadata.jsons"


    #     with open(vrs_file_path, "r") as metadata_file:
    #         for line in metadata_file:
    #             metadata_obj = json.loads(line)
    #             print(metadata_obj)
    # ------------------- END: OUTDATED -------------------

    def undistort_image(self, image_distorted) -> np.ndarray:
        rgb_camera_calibration = get_camera_calibration(self.provider, self._rgb_stream_id)
        focal_lengths = rgb_camera_calibration.get_focal_lengths()
        image_size = rgb_camera_calibration.get_image_size()

        # create pinhole (undistorted) calibration
        pinhole_calib = calibration.get_linear_camera_calibration(image_size[0], image_size[1], focal_lengths[0])
        # get device to rgb camera transform
        
        # undistort image
        image_undistorted = distort_by_calibration(
            image_distorted,
            rgb_camera_calibration,
            pinhole_calib,
        )

        # BGR_dist = cv2.cvtColor(image_distorted, cv2.COLOR_RGB2BGR)
        # BGR_undist = cv2.cvtColor(image_undistorted, cv2.COLOR_RGB2BGR)

        # Show distorted and undistorted images side by side, rotated -90 degrees
        # BGR_dist_rot = cv2.rotate(BGR_dist[::2, ::2,], cv2.ROTATE_90_CLOCKWISE)
        # BGR_undist_rot = cv2.rotate(BGR_undist[::2, ::2,], cv2.ROTATE_90_CLOCKWISE)
        # combined = np.hstack((BGR_dist_rot, BGR_undist_rot))
        # cv2.imshow('Distorted (left) vs Undistorted (right)', combined)
        # cv2.waitKey(0)

        image_undistorted = cv2.rotate(image_undistorted, cv2.ROTATE_90_CLOCKWISE)
        
        return image_undistorted




    def match_timestamp_to_rgb_frame_id(self, timestamp: int) -> int:
        all_frames_data = self.hand_velocities_data_path + "all_frames.json"

        image_data = self.provider.get_image_data_by_ns_timestamp(StreamId("214-1"), timestamp)
        return image_data.frame_index
            

    def process_episodes(self) -> None:

        episode_frame_idxs = {}

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
            
            # TODO: needs to be changed such that only unqiue episodes are added to the dataset
            # create episode object
            episode_id = f"episode_{i}"
            agent_id = "human"

            cur_episode = Episode(episode_id, agent_id)


            # for each episode, process the frames within the start and end indices
            # each frame corresponds to one step in RLDS
            first_useful_frame_found = False
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

                # print(step_normalized_timestamp)
                # print(self.hand_data_left.keys())

                # get hand data for current timestamp
                left_hand_data = self.hand_data_left.get(step_timestamp)
                right_hand_data = self.hand_data_right.get(step_timestamp)

                
                # assert(left_hand_data is not None and right_hand_data is not None), f"Hand data for timestamp {step_timestamp} not found."

                if(left_hand_data is None or right_hand_data is None):
                    count += 1
                    continue
                else:
                    print(f"Found hand data for timestamp {step_timestamp}.")

                cur_step.set_hand_data(
                    hand_pos_left=left_hand_data["positions_3d_m"],
                    hand_pos_right=right_hand_data["positions_3d_m"],
                    hand_vel_left=left_hand_data["velocities_3d_ms"],
                    hand_vel_right=right_hand_data["velocities_3d_ms"]
                )

                



                cur_episode.add_step(cur_step)

            cur_episode.steps[0].set_is_first()
            cur_episode.steps[-1].set_is_last()
            # add finished episode to list
            self.episodes.append(cur_episode)
            



        print(f"Number of missing hand data entries: {count}")
        for ep in self.episodes:
            print(f"Episode ID: {ep.episode_id}, Number of steps: {len(ep.steps)}")

        # episode = Episode(episode_id, agent_id)
        # self.episodes.append(episode)
        # return episode


    def restructure_hand_velocities(self)-> list[dict]:
        hand_velocities_files = [self.hand_velocities_data_path + "left_hand_velocity.json", self.hand_velocities_data_path + "right_hand_velocity.json"]
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
            
            
        # format: left hand data, right hand data
        with open("hand_data_debug.json", "w") as debug_file:
            json.dump(both_hands, debug_file, indent=4)
        return both_hands[0], both_hands[1]

    # TODO: implement actually differing between train and val episodes
    def save_episodes(self, save_dir: str) -> None:
        for episode in self.episodes:
            save_path_train = path.join(save_dir, f"train/{episode.episode_id}.npy")
            save_path_val = path.join(save_dir, f"val/{episode.episode_id}.npy")
            episode.save_to_np(save_path_train)
            episode.save_to_np(save_path_val)
    


if __name__ == "__main__":
    shared_path_vrs_data = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data"
    vrs_file_name = "Microsoft_office_1.vrs"
    shared_path_hand_velocities = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/utils/final_output/"
    shared_path_save_rlds_data = "/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_rlds_builder-main/aria_dataset/data/"
    
    windows_base_path = "C:"
    linux_base_path = "/mnt/c"

    if platform.system() == "Windows":
        converter = VrsToRldsConverter(
            vrs_data_path= path.join(windows_base_path,shared_path_vrs_data),
            vrs_file_name=vrs_file_name,
            hand_velocities_data_path= path.join(windows_base_path,shared_path_hand_velocities),
        )
        converter.process_episodes()
        converter.save_episodes(save_dir=
                                path.join(windows_base_path,shared_path_save_rlds_data)
        )
    elif platform.system() == "Linux":
        print("Running on Linux system.")
        print("Hand velocities path:", path.join(linux_base_path,shared_path_hand_velocities))
        converter = VrsToRldsConverter(
            vrs_data_path= linux_base_path + shared_path_vrs_data,
            
            # path.join(linux_base_path,shared_path_vrs_data),
            vrs_file_name=vrs_file_name,
            hand_velocities_data_path= linux_base_path + shared_path_hand_velocities
            # path.join(linux_base_path, shared_path_hand_velocities),
        )
        converter.process_episodes()
        converter.save_episodes(save_dir= linux_base_path + shared_path_save_rlds_data
                                # path.join(linux_base_path,shared_path_save_rlds_data)
        )