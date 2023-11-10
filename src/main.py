""""""

import argparse
import multiprocessing
import os

import pytorch_lightning as pl
import torch

from datasets import get_dataloader
from datasets.nyu import NYUDepthV2
from datasets.midair import MidAir
from datasets.hypersim import HyperSim
from datasets.kitti import Kitti
from eval_utils import eval_dataset

from metrics import get_metric
from file_utils import read_json, save_json
from models.convNext import ConvNext
from models.resnet import Resnet
from models.decoder import get_decoder
from models.task import get_task


def read_tasks(args_tasks):
    """Reads a task argument list and convert to task objects"""
    task_list = []
    for task in args_tasks:
        Task = get_task(task["type"])
        Decoder = get_decoder(task["decoder"]["type"])
        metrics = {name: get_metric(name)() for name in task["metrics"]}

        _task = Task(
            name=task["name"],
            decoder=Decoder,
            metrics=metrics,
            features=task["features"],
            channels=task["channels"],
            mask_feature=task.get("mask_feature"),
            decoder_args=task["decoder"]["args"],
            train_on_disparity=task.get("train_on_disparity", False),
        )

        task_list.append(_task)
    return task_list


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--evaluate_path", type=str, default=None)

    args = vars(parser.parse_args())
    if not args["evaluate_path"]:
        args.update(read_json(args["config"]))

    args["input_args"] = args.copy()
    if "tasks" in args:
        args["tasks"] = read_tasks(args["tasks"])

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


def train(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = 2

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
    model = ConvNext(
        tasks=tasks,
        features=args["features"],
        pretrained_encoder=True,
        encoder_name=args.get("encoder_name"),
    ).to_gpu(DEVICE)

    trainer = pl.Trainer(
        # limit_train_batches=100,
        # limit_val_batches=100,
        max_epochs=args["epochs"],
        accelerator="gpu",
        devices=[DEVICE],
        default_root_dir=logpath,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    print("DONE!")


def test(args):
    """"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = 0

    # Get model path
    model_path = args["evaluate_path"]

    # Locate experiment folder
    exp_path = args["evaluate_path"].split("lightning")[0]

    # Load experiment args
    exp_args = read_json(os.path.join(exp_path, "args.json"))
    args.update(exp_args)
    args["tasks"] = read_tasks(args["tasks"])

    # Get model weights
    print("Loading model weights")
    model_weights = torch.load(model_path)

    # Build a model and load weights
    # TODO: Get model from args
    model = ConvNext(
        tasks=args["tasks"],
        features=args["features"],
        pretrained_encoder=False,
    )
    model.load_state_dict(model_weights["state_dict"])

    # Load datasets
    # TODO: load by args
    # TODO: Fix this args override mess!

    print("Loading NYU validation dataset")
    args["split_json"] = "/home/luiz.decker/code/DepthTasks/configs/nyu.json"
    args["dataset_root"] = "/hadatasets/nyu"
    nyu_val_ds = NYUDepthV2(split="validation", **args)
    nyu_dataloader = nyu_val_ds.build_dataloader(
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args.get("num_workers", multiprocessing.cpu_count()),
    )

    print("Loading MidAir validation dataset")
    args[
        "split_json"
    ] = "/home/luiz.decker/code/DepthTasks/configs/midair_splits.json"
    args["dataset_root"] = "/hadatasets/midair/MidAir"
    midair_val_ds = MidAir(split="validation", **args)
    midair_dataloader = midair_val_ds.build_dataloader(
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args.get("num_workers", multiprocessing.cpu_count()),
    )

    print("Loading HyperSim validation dataset")
    args[
        "split_json"
    ] = "/home/luiz.decker/code/DepthTasks/configs/hypersim_splits_nonan.json"
    args["dataset_root"] = "/hadatasets/hypersim"
    hypersim_val_ds = HyperSim(split="validation", **args)
    hypersim_dataloader = hypersim_val_ds.build_dataloader(
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args.get("num_workers", multiprocessing.cpu_count()),
    )

    print("Loading KITTI validation dataset")
    args[
        "split_file"
    ] = "/home/luiz.decker/code/DepthTasks/configs/kitti_eigen_val.txt"
    args["dataset_root"] = "/hadatasets/kitti"
    kitti_val_ds = Kitti(split="validation", **args)

    # Create trainer object
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=[DEVICE],
    )

    # Test on each dataset
    global_metrics_nyu, sample_metrics_nyu = eval_dataset(model, nyu_val_ds)
    global_metrics_midair, sample_metrics_midair = eval_dataset(
        model, midair_val_ds
    )
    global_metrics_hypersim, sample_metrics_hypersim = eval_dataset(
        model, hypersim_val_ds
    )

    global_metrics_kitti, sample_metrics_kitti = eval_dataset(
        model, kitti_val_ds
    )

    print("--->nyu", global_metrics_nyu)
    print("--->midair", global_metrics_midair)
    print("--->hypersim", global_metrics_hypersim)
    print("--->kitti",global_metrics_kitti)


if __name__ == "__main__":
    args = read_args()
    if args["evaluate_path"]:
        test(args)
    else:
        train(args)
    print("Done!")
