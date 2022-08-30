from tartanair import TartanAir


def get_dataloader(dataset):
    """Gets the dataloader for the specified dataset"""

    datasets = {"tartanair": TartanAir}

    return datasets[dataset.lower()]
