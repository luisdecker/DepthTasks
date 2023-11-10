"Dataloading utilities for KITTI dataset"

from glob import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader

from datasets.image_transforms import ImageTransformer

"Utility Functions_____________________________________________________________"


def load_filelist(filepath):
    "Loads a list of files from a text file"

    with open(filepath, "r") as f:
        data = f.readlines()
    data = [line.split(" ") for line in data]
    return data


def gen_file_list(dataset_path, split_file):
    """
    Generates a file list with all the images from the dataset

    The file in the list contains:
        - Depth L
        - Image L

    """
    file_list = []
    # Get data from split filelist
    raw_file_list = load_filelist(split_file)
    for sample in raw_file_list:
        image, depth = sample
        image = os.path.join(dataset_path,image)
        depth = os.path.join(dataset_path,depth)
        file_list.append({"image_l": image, "depth_l": depth})

    return file_list


# MidAir Loader _____________________________________________________________


class Kitti:
    "Dataloader for kitti dataset"

    def __init__(self, dataset_root, split, split_file, **args):
        """"""
        self.dataset_root = dataset_root
        self.file_list = gen_file_list(self.dataset_root, split_file)
        self.target_size = args.get("target_size")
        self.features = args["features"]
        self.split = split
        self.depth_clip = args.get("depth_clip")
        self.mask_sky = args.get("mask_sky")

        self.image_transformer = ImageTransformer(self.split).get_transform()

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def _load_image(image_path, resize_shape=None, feature=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"

        def _crop_center(img):
            "Crops only the center of the image"
            height, width = img.shape
            crop1 = (width - height)//2
            crop2 = (width - height)//2 + height
            return img[:,crop1:crop2]

        
        if feature.startswith("depth"):
            img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED).astype('float32')
            img = _crop_center(img)
            img = Image.fromarray(img.astype(np.float32))
        
        if feature.startswith("image"):
            img = cv2.imread(image_path)
            img = _crop_center(img)
            img = cv2.cvtColor(cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        # Resizes if shape is provided
        if resize_shape:
            img = img.resize(resize_shape, resample=Image.BICUBIC)

        return img

    
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
            read_data = Kitti._load_image(
                data_paths[feature], self.target_size, feature
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
