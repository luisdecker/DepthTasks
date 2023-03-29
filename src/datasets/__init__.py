""""""
from .tartanair import TartanAir
from .nyu import NYUDepthV2


def get_dataloader(dataset):
    """Gets the dataloader for the specified dataset"""

    datasets = {"tartanair": TartanAir, "nyu": NYUDepthV2}

    return datasets[dataset.lower()]
