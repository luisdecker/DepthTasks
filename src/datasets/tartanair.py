"Dataloading utilities for TartanAir dataset"

import os

from torch.utils.data import DataLoader


class TartanAir:
    "Dataloader for tartanair dataset"

    def __init__(self, dataset_root, file_list_path):
        ""
        self.dataset_root = dataset_root
        self.file_list_path = file_list_path

    def load_file_list(self,file_list_path):
        ""
        with open(file_list_path, 'r') as handler:
            files = handler.read().splitlines()






