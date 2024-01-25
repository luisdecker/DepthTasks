"Dataloading utilities for TartanAir dataset"

from glob import glob
import json
import os

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from datasets.dataset import Dataset

from datasets.image_transforms import ImageTransformer

SKY_INDEXES = {
    "neighborhood": 146,
    "abandonedfactory": 196,
    "abandonedfactory_night": 196,
    "amusement": 182,
    "carwelding": None,
    "endofworld": 146,
    "gascola": 112,
    "hospital": None,
    "japanesealley": None,
    "ocean": 231,
    "office": None,
    "office2": None,
    "oldtown": 223,
    "seasidetown": 130,
    "seasonsforest": 196,
    "seasonsforest_winter": 146,
    "soulcity": 130,
    "westerndesert": 196,
}

# fmt: off
"""DATASET STRUCTURE
-scene1
    - Easy
        - Pxxx
            - depth_left
                - kkkk_left_depth.npy
                - kkkl_left_depth.npy
                - ...
            - depth_right
                - kkkk_right_depth.npy
                - kkkl_right_depth.npy
                - ...
            - image_left
                - kkkk_left.png
                - kkkl_left.png
            - image_right
                - kkkk_right.png
                - kkkl_right.png
            - seg_left
                - kkkk_left_seg.npy
                - kkkl_left_seg.npy
            - seg_right
                - kkkk_right_seg.npy
                - kkkl_right_seg.npy
        - Pxxy
    - Hard
-scene2 
...


"""


# fmt: on
"Utility Functions_____________________________________________________________"


def get_split_from_json(split, file):
    with open(file, "r") as f:
        return json.load(f)[split]


def is_valid_folder(folder_path):
    """Validates if the folder contains the expected subfolders"""

    # Get subfolders
    subfolders = [f for f in glob(folder_path + "/*") if os.path.isdir(f)]
    subfolders = [f.split("/")[-1] for f in subfolders]

    # Verify if subfolders exists
    expected_subfolders = ["Easy", "Hard"]
    return all([sf in subfolders for sf in expected_subfolders])


def get_scenes_paths(dataset_path):
    "Get the path of all the scene folders found in the dataset"

    # Verify if dataset_path has a / in the end
    dataset_path = (
        dataset_path + "/" if not dataset_path.endswith("/") else dataset_path
    )

    # Get all folder candidate in dataset root
    folders = [x for x in glob(dataset_path + "*") if not x[:-3].endswith(".")]

    # Filter no-folders
    folders = [f for f in folders if os.path.isdir(f)]

    # Validate if is a scene folder
    folders = [f for f in folders if is_valid_folder(f)]

    return folders


def get_path_ids(path):
    "Get the id of all the available files in a path"
    all_depths = glob(os.path.join(path, "depth_left") + "/*")
    return [x.split("/")[-1].split("_")[0] for x in all_depths]


def get_data_from_id(id, path):
    "Get the path from all the data from a id"

    return {
        "depth_l": os.path.join(path, "depth_left", f"{id}_left_depth.npy"),
        "depth_r": os.path.join(path, "depth_right", f"{id}_right_depth.npy"),
        "image_l": os.path.join(path, "image_left", f"{id}_left.png"),
        "image_r": os.path.join(path, "image_right", f"{id}_right.png"),
        "seg_l": os.path.join(path, "seg_left", f"{id}_left_seg.npy"),
        "seg_r": os.path.join(path, "seg_right", f"{id}_right_seg.npy"),
    }


# Tartanair Loader _____________________________________________________________


class TartanAir(Dataset):
    "Dataloader for tartanair dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        super().__init__(dataset_root, split, split_json, **args)
        self.scenes = get_split_from_json(split, split_json)

    def gen_file_list(self, dataset_path, split_json, split):
        """
        Generates a file list with all the images from the dataset

        The file in the list contains:
            - Scene name
            - Depth L
            - Depth R
            - Image L
            - Image R
            - Segmentation L
            - Segmentation R
        """
        scenes = get_split_from_json(split, split_json)
        file_list = []
        scene_folders = get_scenes_paths(dataset_path)
        found_scenes = [scene.split("/")[-1] for scene in scene_folders]
        assert all(
            scene in found_scenes for scene in scenes
        ), f"{scene} not found!\n{found_scenes}"

        for scene in scene_folders:
            scene_name = scene.split("/")[-1]
            if scene_name not in scenes:
                continue

            for difficulty in ["Easy", "Hard"]:
                # Get all the available paths
                paths_root = os.path.join(scene, difficulty)
                paths = glob(paths_root + "/*")

                for path in paths:
                    # Get available image ids
                    ids = get_path_ids(path)
                    for id in ids:
                        # get all the data for this id
                        all_paths_id = get_data_from_id(id, path)

                        # add extra info
                        all_paths_id["scene"] = scene_name

                        file_list.append(all_paths_id)
        return file_list

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        if image_path.endswith(".png"):
            # Loads image
            assert os.path.isfile(image_path), f"{image_path} is not a file!"
            img = Image.open(image_path)

        else:
            img = TartanAir._load_npz(image_path)

        # Resizes if shape is provided
        img = self._crop_center(img)
        if resize_shape:
            img = img.resize(resize_shape, resample=Image.BICUBIC)

        return img

    @staticmethod
    def _load_npz(filepath, resize_shape=None):
        assert os.path.isfile(filepath), f"{filepath} is not a file!"
        data = np.load(filepath)
        data_img = Image.fromarray(data).convert("F")

        if resize_shape:
            data_img = data_img.resize(resize_shape, resample=Image.BICUBIC)

        return data_img
