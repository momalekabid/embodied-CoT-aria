from setuptools import setup, find_packages

setup(
    name="aria_dataset",
    packages=find_packages(),
    entry_points={
        "tensorflow_datasets": [
            "aria_dataset = aria_dataset:AriaDataset",
        ],
    },
)
