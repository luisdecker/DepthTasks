""""""

import argparse
import multiprocessing

from datasets import get_dataloader
from file_utils import read_json
from models.convNext import ConvNext
from models.decoder import SimpleDecoder
from models.simple_encoder import SimpleEncoder
from models.task import DenseRegression


def read_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)

    args = vars(parser.parse_args())
    args.update(read_json(args["config"]))

    return args


def prepare_dataset(dataset_root, dataset, target_size=None, **args):
    """Prepares datasets for train, validation and test"""
    Dataset = get_dataloader(dataset)
    batch_size = args["batch_size"]
    num_workers = args.get("num_workers", multiprocessing.cpu_count())

    print("Preparing train dataset...")
    train_dataset = Dataset(
        dataset_root=dataset_root,
        split="train",
        target_size=target_size,
        **args
    )
    train_loader = train_dataset.build_dataloader(
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    # __________________________________________________________________________
    print("Preparing validation dataset...")

    val_dataset = Dataset(
        dataset_root=dataset_root,
        split="validation",
        target_size=target_size,
        **args
    )
    val_loader = val_dataset.build_dataloader(
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # __________________________________________________________________________
    print("Preparing test dataset...")

    test_dataset = Dataset(
        dataset_root=dataset_root,
        split="test",
        target_size=target_size,
        **args
    )
    test_loader = test_dataset.build_dataloader(
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def debug():
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    DEVICE = [0]

    import pytorch_lightning as pl

    args = read_args()

    # Build a task
    decoder = SimpleDecoder

    task = DenseRegression(
        decoder=decoder, features=args["features"][-1], channels=1
    )

    train_loader, val_loader, test_loader = prepare_dataset(**args)

    # Try to train some network
    model = SimpleEncoder(tasks=[task], features=args["features"]).to_gpu(
        DEVICE[0]
    )

    trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=DEVICE)

    trainer.fit(model=model, train_dataloaders=val_loader)

    print("DONE!")


def debug_convnext():
    net = ConvNext(pretrained_encoder=False)
    import torch

    x = torch.zeros([1, 3, 256, 256])
    net(x)


debug()
