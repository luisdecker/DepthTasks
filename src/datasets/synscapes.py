"Dataloading utilities for Synscapes dataset"

import json
import os

import numpy as np
from PIL import Image
import cv2

from datasets.dataset import Dataset


"Utility Functions_____________________________________________________________"


def gen_paths_from_id(root, idx):
    rgb_path = os.path.join(root, f"img/rgb/{idx}.png")
    depth_path = os.path.join(root, f"img/depth/{idx}.npy")
    seg_path = os.path.join(root, f"img/class/{idx}.png")
    return {"image_l": rgb_path, "depth_l": depth_path, "seg_l": seg_path}


# MidAir Loader _____________________________________________________________


class Synscapes(Dataset):
    "Dataloader for synscapes dataset"

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
        print("Generating Synscapes Filelist")
        with open(split_file, "r") as handler:
            split_ids = json.load(handler)[split]

        return [gen_paths_from_id(dataset_path, id) for id in split_ids]

    def _load_image(self, image_path, resize_shape=None, feature=None):
        """"""
        # Loads image
        assert os.path.isfile(image_path), f"{image_path} is not a file!"

        if feature.startswith("depth"):
            img = np.load(image_path).astype("float32")
            img = self._crop_center(img)
            img = Image.fromarray(img.astype(np.float32))

        if feature.startswith("image"):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._crop_center(img)
            img = Image.fromarray(img)

        # Resizes if shape is provided
        if resize_shape:
            img = img.resize(resize_shape, resample=Image.BICUBIC)

        return img
