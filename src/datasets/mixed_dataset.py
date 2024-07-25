"Union of several datasets"

from datasets.dataset import Dataset


class MixedDataset(Dataset):
    """"""

    def __init__(self, **args):
        self.datasets = args.get("datasets")
        self.ds_indexes = []
        self._get_dataset_indexes()
        self.size = None

    def __len__(self):
        "Get number of samples of dataset"
        if self.size is not None:
            return self.size
        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)
        return self.size

    def __getitem__(self, idx):
        "Get a dataset sample, given the correct dataset"
        # Get correct dataset, searching the list

        last_min = 0
        for max_idx, loader in self.ds_indexes:
            if idx < max_idx:
                dataset = loader
                break
            last_min = max_idx

        return dataset[idx - (last_min + 1)]

    def _get_dataset_indexes(self):
        "Generate the dataset index list"
        full_ds_size = 0
        for dataset in self.datasets:
            ds_size = len(dataset)
            full_ds_size += ds_size
            self.ds_indexes.append((full_ds_size, dataset))
        self.ds_indexes = sorted(self.ds_indexes)
