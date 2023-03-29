""""""

import argparse
import multiprocessing
import os

import pytorch_lightning as pl
import torch
import torchmetrics
from rich.progress import track

from datasets import get_dataloader
from file_utils import read_json, save_json
from models.convNext import ConvNext
from models.resnet import Resnet
from models.decoder import (
    ConvNextDecoder,
    SimpleDecoder,
    UnetDecoder,
    get_decoder,
)
from models.simple_encoder import SimpleEncoder
from models.task import DenseRegression, get_task


def read_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)

    args = vars(parser.parse_args())
    args.update(read_json(args["config"]))

    args["input_args"] = args.copy()
    if "tasks" in args:
        task_list = []
        for task in args["tasks"]:
            Task = get_task(task["type"])
            Decoder = get_decoder(task["decoder"]["type"])

            _task = Task(
                name=task["name"],
                decoder=Decoder,
                features=task["features"],
                channels=task["channels"],
                mask_feature=task.get("mask_feature"),
                decoder_args=task["decoder"]["args"],
                train_on_disparity=task.get("train_on_disparity", False),
            )

            task_list.append(_task)
        args["tasks"] = task_list

    return args


def prepare_dataset(dataset_root, dataset, target_size=None, **args):
    """Prepares datasets for train, validation and test"""

    Dataset = get_dataloader(dataset)
    batch_size = args["batch_size"]
    num_workers = args.get("num_workers", multiprocessing.cpu_count())

    train_loader = None
    val_loader = None
    test_loader = None

    if args.get("train", False):
        print("Preparing train dataset...")
        train_dataset = Dataset(
            dataset_root=dataset_root,
            split="train",
            target_size=target_size,
            **args,
        )
        train_loader = train_dataset.build_dataloader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    # __________________________________________________________________________
    if args.get("validation", False):

        print("Preparing validation dataset...")

        val_dataset = Dataset(
            dataset_root=dataset_root,
            split="validation",
            target_size=target_size,
            **args,
        )
        val_loader = val_dataset.build_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    # __________________________________________________________________________
    if args.get("test", False):
        print("Preparing test dataset...")

        test_dataset = Dataset(
            dataset_root=dataset_root,
            split="test",
            target_size=target_size,
            **args,
        )
        test_loader = test_dataset.build_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return train_loader, val_loader, test_loader


def get_last_exp(logpath):
    try:
        folders = next(os.walk(logpath))[1]
    except StopIteration:
        return 0
    after_ = [f.split("_")[-1] for f in folders if len(f.split("_")) > 1]
    numbers = [int(f) for f in after_ if f.isdigit()]
    if len(numbers) == 0:
        return 0
    last_number = sorted(numbers)[-1]
    return last_number + 1


def debug():

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = 2

    import pytorch_lightning as pl

    # Read CLI and json args
    args = read_args()

    # Generate log path

    logpath = os.path.join("logs", args["expname"])

    logpath = os.path.join(logpath, f"_{get_last_exp(logpath)}")
    os.makedirs(logpath, exist_ok=False)

    # Saves the args in the path
    args_path = os.path.join(logpath, "args.json")
    save_json(args_path, args["input_args"])

    tasks = args["tasks"]

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataset(
        train=True, validation=True, **args
    )

    # Try to train some network
    model = Resnet(
        tasks=tasks, features=args["features"], pretrained_encoder=True
    ).to_gpu(DEVICE)

    trainer = pl.Trainer(
        max_epochs=args["epochs"],
        accelerator="gpu",
        devices=[DEVICE],
        default_root_dir=logpath,
    )

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    print("DONE!")


def test():
    """"""
    DEVICE = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    network_path = "/home/luiz.decker/code/DepthTasks/lightning_logs/version_23/checkpoints/epoch=99-step=796700.ckpt"

    # Read CLI and json args
    args = read_args()

    # Build a task
    decoder = UnetDecoder

    task = DenseRegression(
        decoder=decoder, features=args["features"][-1], channels=1
    )

    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataset(
        train=True, validation=True, **args
    )

    # Try to train some network
    model = ConvNext(tasks=[task], features=args["features"]).to_gpu(DEVICE)

    state_dict = torch.load(network_path)["state_dict"]

    model.load_state_dict(state_dict=state_dict)

    model = model.eval()

    val_iter = iter(val_loader)

    metrics = {"rmse": [], "mape": []}

    rmse = torchmetrics.MeanSquaredError(squared=False)
    mape = torchmetrics.MeanAbsolutePercentageError()

    with torch.no_grad():

        for batch in track(val_iter, description="Evaluating..."):
            x, y = batch

            _y = model(x.cuda(DEVICE)).detach().cpu()

            for i in range(len(_y)):
                metrics["rmse"].append(float(rmse(_y[i], y[i])))
                metrics["mape"].append(float(mape(_y[i], y[i])))

    metrics["global_rmse"] = float(rmse.compute())
    metrics["global_mape"] = float(mape.compute())

    print("Done!")


if __name__ == "__main__":
    args = debug()
    print("Done!")
