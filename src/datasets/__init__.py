""""""
from .tartanair import TartanAir
from .nyu import NYUDepthV2
from .midair import MidAir


def get_dataloader(dataset):
    """Gets the dataloader for the specified dataset"""

    datasets = {"tartanair": TartanAir, "nyu": NYUDepthV2, "midair": MidAir}

    return datasets[dataset.lower()]
