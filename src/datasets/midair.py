"Dataloading utilities for MidAir dataset"

from glob import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

from datasets.image_transforms import ImageTransformer

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


def get_path_ids(climate_root, trajec):
    "Get the id of all the available files in a path"

    all_depths = glob(os.path.join(climate_root, "depth", trajec) + "/*.PNG")
    return [Path(x).stem for x in all_depths]


def get_data_from_id(id, climate_root, trajec):
    "Get the path from all the data from a id"

    return {
        "depth_l": os.path.join(climate_root, "depth", trajec, f"{id}.PNG"),
        "image_l": os.path.join(
            climate_root, "color_left", trajec, f"{id}.JPEG"
        ),
        "image_r": os.path.join(
            climate_root, "color_right", trajec, f"{id}.JPEG"
        ),
        "seg_l": os.path.join(
            climate_root, "segmentation", trajec, f"{id}.PNG"
        ),
    }


def gen_file_list(dataset_path, scenes):
    """
    Generates a file list with all the images from the dataset

    The file in the list contains:
        - Scene name
        - Depth L
        - Image L
        - Image R
        - Segmentation L

    Scenes is a dict:
        map{
            climate[
                trajectories
            ]
        }
    """
    file_list = []
    for map in scenes:
        for climate in scenes[map]:
            climate_root = os.path.join(dataset_path, map, climate)
            for trajec in scenes[map][climate]:

                ids = get_path_ids(climate_root, trajec)

                for id in ids:
                    all_paths_id = get_data_from_id(id, climate_root, trajec)
                    file_list.append(all_paths_id)
    return file_list


# MidAir Loader _____________________________________________________________


class MidAir:
    "Dataloader for tartanair dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        self.dataset_root = dataset_root
        self.scenes = get_split_from_json(split, split_json)
        self.file_list = gen_file_list(self.dataset_root, self.scenes)
        self.target_size = args.get("target_size")
        self.features = args["features"]
        self.split = split
        self.depth_clip = args.get("depth_clip")
        self.mask_sky = args.get("mask_sky")

        self.image_transformer = ImageTransformer(self.split).get_transform()

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def _load_image(image_path, resize_shape=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"
        img = Image.open(image_path)

        # Resizes if shape is provided
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

    def __getitem__(self, idx):

        data_paths = self.file_list[idx]

        input_features, label_features = self.features

        input_data = self.get_data_from_features(data_paths, input_features)
        input_data = torch.stack([input_data[f] for f in input_features])

        label_data = self.get_data_from_features(data_paths, label_features)
        label_data = torch.stack([label_data[f] for f in label_features])

        return input_data, label_data

    def get_data_from_features(self, data_paths, features):
        data = {}
        for feature in features:
            read_data = MidAir._load_image(
                data_paths[feature], self.target_size
            )

            # Depth resizing
            if feature.startswith("depth"):
                depth = np.array(read_data)
                depth = depth / 100
                read_data = Image.fromarray(depth).convert("F")
                
            # Depth clipping
            if feature.startswith("depth") and self.depth_clip:
                depth = np.array(read_data)
                np.clip(depth, a_min=0, a_max=self.depth_clip)
                read_data = Image.fromarray(depth).convert("F")

            # Sky masking
            # if feature.startswith("seg") and self.mask_sky:
            #     seg_map = np.floor(np.array(read_data))
            #     scene = data_paths["scene"]
            #     if SKY_INDEXES[scene]:
            #         ground_pixels = seg_map != SKY_INDEXES[scene]
            #         read_data = Image.fromarray(ground_pixels).convert("F")
            #     else:
            #         read_data = Image.fromarray(
            #             np.ones_like(seg_map) * 255
            #         ).convert("F")

            data[feature] = read_data

        return self.image_transformer(data)

    def build_dataloader(self, shuffle, batch_size, num_workers):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
