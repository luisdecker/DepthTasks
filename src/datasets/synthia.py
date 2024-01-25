"Dataloading utilities for Synthia dataset"

import json
import os

import numpy as np
from PIL import Image
import cv2

from datasets.dataset import Dataset


"Utility Functions_____________________________________________________________"


def gen_paths_from_id(root, idx):
    """
    Extends the paths with the dataset root
    """
    idx["image_l"] = os.path.join(root, idx["image_l"])
    idx["depth_l"] = os.path.join(root, idx["depth_l"])
    idx['seg_l'] = os.path.join(root, idx["seg_l"])
    return idx


# MidAir Loader _____________________________________________________________


class Synthia(Dataset):
    "Dataloader for synthia dataset"

    def __init__(self, dataset_root, split, split_json, **args):
        """"""
        super().__init__(dataset_root, split, split_json, **args)

    def gen_file_list(self, dataset_path, split_file, split):
        """
        Generates a file list with all the images from the dataset

        The file in the list contains:
            - Depth L
            - Image L

        """
        print("Generating Synthia Filelist")
        with open(split_file, "r") as handler:
            split_ids = json.load(handler)[split]

        return [gen_paths_from_id(dataset_path, id) for id in split_ids]

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"

        if feature.startswith("depth"):
            img = cv2.imread(image_path, -1)
            R = img[:, :, 2]
            G = img[:, :, 1] * 256
            B = img[:, :, 0] * 256 * 265
            dividendo = 256 * 256 * 256 - 1

            depth = 5000 * ((R + G + B) / dividendo)
            img = Image.fromarray(depth.astype(np.float32))

        if feature.startswith("image"):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        # Resizes if shape is provided
        img = np.array(img)
        img = self._crop_center(img)
        img = Image.fromarray(img)
        if resize_shape:
            img = img.resize(resize_shape, resample=Image.BICUBIC)

        return img
