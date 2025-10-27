import jsons
import json
from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from tqdm import tqdm
import numpy as np
from os import path

class Step:
    def __init__(self, is_first: bool, is_last: bool) -> None:
        self._is_first = is_first
        self._is_last = is_last
        self._observations = {
            "time_since_episode_start_ns": None,
        }
        self._actions = {
            "hand_pos_left": None,
            "hand_pos_right": None,
            "hand_vel_left": None,
            "hand_vel_right": None
        }

    

    def set_observation_time(self, value) -> None:
        self._observations["time_since_episode_start_ns"] = value
    
    def set_hand_data(self, hand_pos_left: list[float], hand_pos_right: list[float], hand_vel_left: list[float], hand_vel_right: list[float]) -> None:
        self._actions["hand_pos_left"] = hand_pos_left
        self._actions["hand_pos_right"] = hand_pos_right
        self._actions["hand_vel_left"] = hand_vel_left
        self._actions["hand_vel_right"] = hand_vel_right
        


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



    # ------------------- START: OUTDATED -------------------
    # this function peeks into the VRS metadata file and prints out the metadata objects
    # def peek_into_vrs_metadata(self, file_name: str) -> None:

    #     vrs_file_path = f"{self.vrs_data_path}/metadata.jsons"


    #     with open(vrs_file_path, "r") as metadata_file:
    #         for line in metadata_file:
    #             metadata_obj = json.loads(line)
    #             print(metadata_obj)
    # ------------------- END: OUTDATED -------------------

    def match_timestamp_to_rgb_frame_id(self, timestamp: int) -> int:
        all_frames_data = self.hand_velocities_data_path + "all_frames.json"

        image_data = self.provider.get_image_data_by_ns_timestamp(StreamId("214-1"), timestamp)
        return image_data.frame_index
            

    def process_episodes(self) -> None:
        rgb_stream_id = StreamId("214-1")

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
                end_idx = self.provider.get_num_data(rgb_stream_id)
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
            for frame_idx in range(start_idx, end_idx):
                cur_step = Step(is_first=(frame_idx == start_idx), is_last=(frame_idx == end_idx - 1))

                image_data = self.provider.get_image_data_by_index(rgb_stream_id, frame_idx)
    
                if image_data is None:
                    continue
                
                # safety check that frame_idx is within bounds
                if frame_idx > self.provider.get_num_data(rgb_stream_id):
                    raise ValueError("Frame index exceeds number of frames in stream.")
                
                # get image as numpy array and its timestamp
                step_image = image_data[0].to_numpy_array()
                step_timestamp = int(image_data[1].capture_timestamp_ns /1000 )
                # normalize timestamp to episode start
                step_normalized_timestamp = step_timestamp - cur_base_timestamp

                cur_step.set_observation_time(step_normalized_timestamp)

                # print(step_normalized_timestamp)
                # print(self.hand_data_left.keys())

                # get hand data for current timestamp
                left_hand_data = self.hand_data_left.get(step_timestamp)
                right_hand_data = self.hand_data_right.get(step_timestamp)

                # print(f"Looking for hand data at timestamp {step_timestamp}...")
                # if(left_hand_data is not None):
                #     print(f"Left hand data for timestamp {step_timestamp} found.")
                # if(right_hand_data is not None):
                #     print(f"Right hand data for timestamp {step_timestamp} found.")

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






                # process the frame data here
                # ...
                cur_episode.add_step(cur_step)

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

        
    


if __name__ == "__main__":
    converter = VrsToRldsConverter(vrs_data_path="C:/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_vrs/vrs_data", vrs_file_name="Microsoft_office_1.vrs", hand_velocities_data_path= "C:/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/utils/final_output/")
    converter.process_episodes()