import jsons
import json


class Step:
    def __init__(self, is_first: bool, is_last: bool) -> None:
        self.is_first = is_first
        self.is_last = is_last



class Episode:
    def __init__(self, episode_id: str, agent_id: str) -> None:
        self.episode_id = episode_id
        self.agent_id = agent_id
        self.invalid = False

    def mark_invalid(self) -> None:
        self.invalid = True

class VrsToRldsConverter:

    def __init__(self, vrs_data_path: str) -> None:
        self.vrs_data_path = vrs_data_path
        self.episodes = []

    def peek_into_vrs_metadata(self, file_name: str) -> None:

        vrs_file_path = f"{self.vrs_data_path}/metadata.jsons"


        with open(vrs_file_path, "r") as metadata_file:
            for line in metadata_file:
                metadata_obj = json.loads(line)
                print(metadata_obj)

    def create_new_episode(self, episode_id: str, agent_id: str) -> Episode:
        episode = Episode(episode_id, agent_id)
        self.episodes.append(episode)
        return episode


        
    


if __name__ == "__main__":
    converter = VrsToRldsConverter(vrs_data_path="./aria_vrs/extracted/data/ms_office")
    converter.peek_into_vrs_metadata()