import numpy as np

PATH = """/mnt/c/Users/konst/OneDrive/Dokumente/ETH/Jahr 2025 - 2026/Mixed Reality/embodied-CoT-aria/aria_rlds_builder-main/aria_dataset/data/train/episode_0.npy"""

data = np.load(PATH, allow_pickle=True)

print(data[0])
print(data[186])
print(data.shape)