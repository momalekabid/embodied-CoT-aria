from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from aria_dataset.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_examples(episode_path):
        # load raw data --> this should change for your dataset
        data = np.load(episode_path, allow_pickle=True)  # this is a list of dicts in our case
        print(f"Parsing {episode_path} with {len(data)} examples.")
        for k, example in enumerate(data):
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            
            instruction = example['language_instruction'][0]
            if instruction:
                language_embedding = _embed([instruction])[0].numpy()
            else:
                language_embedding = np.zeros(512, dtype=np.float32)

            language_embedding = np.zeros(512, dtype=np.float32)


            for i in range(len(example)):
                observation = {
                    'state': example['state'].astype(np.float32),
                }

                for image_idx in range(4):
                    orig_key = f'images{image_idx}'
                    new_key = f'image_{image_idx}'
                    # if orig_key in example['observations'][i]:
                    #     observation[new_key] = example['observations'][i][orig_key]
                    # else:
                    #     observation[new_key] = np.zeros_like(example['observations'][i]['images0'])

                    if image_idx == 0:
                        observation[new_key] = example['image'].astype(np.uint8)
                    else:
                        observation[new_key] = np.zeros((1,1,3), dtype=np.uint8)



                episode.append({
                    'observation': observation,
                    'action': example['action'].astype(np.float32),
                    'discount': 1.0,
                    'reward': float(i == (len(example) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(example) - 1),
                    'is_terminal': i == (len(example) - 1),
                    'language_instruction': instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_id': k,
                }
            }

            # mark dummy values
            for image_idx in range(1):
                orig_key = f'images{image_idx}'
                new_key = f'image_{image_idx}'
                if image_idx == 0:
                    sample['episode_metadata'][f'has_{new_key}'] = True
                else:
                    sample['episode_metadata'][f'has_{new_key}'] = False
            sample['episode_metadata']['has_language'] = bool(instruction)

            # if you want to skip an example for whatever reason, simply return None
            yield episode_path + str(k), sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        for id, sample in _parse_examples(sample):
            yield id, sample


class AriaDataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 1          # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 10   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_0': tfds.features.Image(
                            shape=(1408, 1408, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(2,3),
                            dtype=np.float32,
                            doc='Hand positions (l,r) given as x,y,z coordinates in meters (measured in ARIA frame = frame of left SLAM camera).',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(2,3),
                        dtype=np.float32,
                        doc='Hand movements (l,r) given as delta x,y,z coordinates in meters/second (measured in ARIA frame = frame of left SLAM camera).',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='ID of episode in file_path.'
                    ),
                    'has_image_0': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image0 exists in observation, otherwise dummy value.'
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if language exists in observation, otherwise empty string.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        base_paths = ["""/mnt/c/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_rlds_builder-main/aria_dataset/data/Banana_v1"""]
        train_filenames, val_filenames = [], []
        for path in base_paths:
          for filename in glob.glob(f'{path}/**/*.npy', recursive=True):
            if '/train/' in filename:
                train_filenames.append(filename)
            elif '/val/' in filename:
                val_filenames.append(filename)
            else:
                raise ValueError(filename)
        print(f"Converting {len(train_filenames)} training and {len(val_filenames)} validation files.")
        return {
            'train': train_filenames,
            'val': val_filenames,
        }

