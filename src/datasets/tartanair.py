"Dataloading utilities for TartanAir dataset"

from glob import glob
import os

from torch.utils.data import DataLoader

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
"Utility Functions"


def validate_folder(folder_path):
    """Validates if the folder contains the expected subfolders"""

    # Get subfolders
    subfolders = [f for f in glob(folder_path + "/*") if os.path.isdir(f)]
    subfolders = [f.split("/")[-1] for f in subfolders]

    # Verify if subfolders exists
    expected_subfolders = ["Easy", "Hard"]
    return all([sf in subfolders for sf in expected_subfolders])
    

def gen_file_list(dataset_path):
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

    # Verify if dataset_path has a / in the end
    dataset_path = (
        dataset_path + "/" if not dataset_path.endswith("/") else dataset_path
    )

    # Get all folder candidate in dataset root
    folders = [x for x in glob(dataset_path + "*") if not x[:-3].endswith(".")]

    # Filter no-folders
    folders = [f for f in folders if os.path.isdir(f)]

    # Validate if is a scene folder


class TartanAir:
    "Dataloader for tartanair dataset"

    def __init__(self, dataset_root, file_list_path):
        """"""
        self.dataset_root = dataset_root
        self.file_list_path = file_list_path
