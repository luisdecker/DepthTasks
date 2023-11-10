"Dataloading utilities for Hypersim dataset"

from glob import glob
import json
import os
from pathlib import Path

import skimage
import numpy as np
from PIL import Image
import h5py

import torch
from torch.utils.data import DataLoader

from datasets.image_transforms import ImageTransformer

"Utility Functions_____________________________________________________________"


def get_split_from_json(split, file):
    with open(file, "r") as f:
        return json.load(f)[split]


def get_data_from_row(dataset_path, scene, camera, id):
    """Gets the paths for all data for a given id"""

    basepath = os.path.join(dataset_path, scene, "images")
    rgb = os.path.join(
        basepath, f"scene_{camera}_final_preview", f"frame.{id:0>4}.color.jpg"
    )
    depth = os.path.join(
        basepath,
        f"scene_{camera}_geometry_hdf5",
        f"frame.{id:0>4}.depth_meters.hdf5",
    )  # size? unit?
    semantic = os.path.join(
        basepath,
        f"scene_{camera}_geometry_hdf5",
        f"frame.{id:0>4}.semantic.hdf5",
    )  # size?

    return {"depth_l": depth, "image_l": rgb, "seg_l": semantic}


def gen_file_list(dataset_path, scenes):
    """
    Generates a file list with all the images from the dataset

    The file in the list contains:
        - Scene name
        - Depth L
        - Image L
        - Image R
        - Segmentation L


    """
    file_list = [get_data_from_row(dataset_path, *row) for row in scenes]

    return file_list


"MidAir Loader _____________________________________________________________"


class HyperSim:
    "Dataloader for hypersim dataset"

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
        self.augmentation = args.get("augmentation", False)

        self.image_transformer = ImageTransformer(
            self.split, augmentation=self.augmentation
        ).get_transform()

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
    def _load_hdf5(image_path, resize_shape=None, is_depth=False):
        with h5py.File(image_path) as f:
            img = f["dataset"][()].astype(np.float32)
            if is_depth:
                img = HyperSim._distance_to_depth(img)
                if np.isnan(img).any():
                    mask = np.isnan(img)
                    try:
                        img = skimage.restoration.inpaint_biharmonic(img, mask)
                    except:
                        print("Deu pau")

            img = Image.fromarray(img).convert("F")

            if resize_shape:
                img = img.resize(resize_shape, resample=Image.BICUBIC)

            return img

    @staticmethod
    def _distance_to_depth(distance):
        "Converts from distance to camera point to distance to camera plane"

        intWidth = 1024
        intHeight = 768
        fltFocal = 886.81

        imgPlaneX = (
            np.linspace(
                (-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth
            )
            .reshape(1, intWidth)
            .repeat(intHeight, 0)
            .astype(np.float32)[:, :, None]
        )

        imgPlaneY = (
            np.linspace(
                (-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight
            )
            .reshape(intHeight, 1)
            .repeat(intWidth, 1)
            .astype(np.float32)[:, :, None]
        )

        imgPlaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)

        imagePlane = np.concatenate([imgPlaneX, imgPlaneY, imgPlaneZ], 2)

        return distance / np.linalg.norm(imagePlane, 2, 2) * fltFocal

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
            read_data = (
                HyperSim._load_image(data_paths[feature], self.target_size)
                if data_paths[feature].endswith(".jpg")
                else HyperSim._load_hdf5(
                    data_paths[feature],
                    self.target_size,
                    is_depth=feature.startswith("depth"),
                )
            )

            # Depth clipping
            if feature.startswith("depth") and self.depth_clip:
                depth = np.array(read_data)
                np.clip(depth, a_min=0, a_max=self.depth_clip)
                read_data = Image.fromarray(depth).convert("F")

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
