""""""

import argparse
import multiprocessing
import os
from matplotlib.font_manager import json_dump

from numpy import DataSource
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar

import pandas as pd

from callbacks import UnfreezeEncoder
from datasets import get_dataloader
from datasets.nyu import NYUDepthV2
from datasets.midair import MidAir
from datasets.hypersim import HyperSim
from datasets.kitti import Kitti
from datasets.virtual_kitti import VirtualKitti
from datasets.synthia import Synthia
from datasets.synscapes import Synscapes
from eval_utils import eval_dataset

from metrics import get_metric
from file_utils import read_json, save_json
from models import get_loss
from models.convNext import ConvNext
from models.decoder import get_decoder
from models.task import get_task

torch.set_float32_matmul_precision("high")


def freeze_encoder(model):
    "Freezes model encoder"
    for param in model.encoder.parameters():
        param.requires_grad = False


def read_tasks(args_tasks):
    """Reads a task argument list and convert to task objects"""
    task_list = []
    for task in args_tasks:
        Task = get_task(task["type"])
        Decoder = get_decoder(task["decoder"]["type"])
        metrics = {name: get_metric(name)() for name in task["metrics"]}
        loss = get_loss(task["loss"], **task.get("loss_params", {}))

        _task = Task(
            name=task["name"],
            decoder=Decoder,
            metrics=metrics,
            features=task["features"],
            channels=task["channels"],
            mask_feature=task.get("mask_feature"),
            decoder_args=task["decoder"]["args"],
            train_on_disparity=task.get("train_on_disparity", False),
            loss=loss,
        )

        task_list.append(_task)
    return task_list


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--evaluate_path", type=str, default=None)
    parser.add_argument("--gpu", type=str, default=0)
    parser.add_argument("--ckpt_path", type=str, default=None)

    args = vars(parser.parse_args())
    if not args["evaluate_path"]:
        args.update(read_json(args["config"]))

    args["input_args"] = args.copy()
    if "tasks" in args:
        args["tasks"] = read_tasks(args["tasks"])

    return args


def prepare_dataset(dataset_root, dataset, target_size=None, **args):
    """Prepares datasets for train, validation and test"""

    dataset_roots = dataset_root  # Retrocomp
    datasets = dataset
    batch_size = args["batch_size"]
    num_workers = args.get("num_workers", multiprocessing.cpu_count())
    splits = args["split_json"]

    train_loaders = []
    val_loaders = []
    test_loaders = []

    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(dataset_roots, str):
        dataset_roots = [dataset_roots]
    if isinstance(splits, str):
        splits = [splits]

    # __________________________________________________________________________
    if args.get("train", False):
        print("Preparing train datasets...")
        for dataset, dataset_root, split_json in zip(
            datasets, dataset_roots, splits
        ):
            Dataset = get_dataloader(dataset)
            args["split_json"] = split_json
            train_dataset = Dataset(
                dataset_root=dataset_root,
                split="train",
                target_size=target_size,
                **args,
            )
            train_loaders.append(
                train_dataset.build_dataloader(
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                )
            )
    # __________________________________________________________________________
    if args.get("validation", False):
        print("Preparing validation datasets...")
        for dataset, dataset_root, split_json in zip(
            datasets, dataset_roots, splits
        ):
            Dataset = get_dataloader(dataset)
            args["split_json"] = split_json

            val_dataset = Dataset(
                dataset_root=dataset_root,
                split="validation",
                target_size=target_size,
                **args,
            )
            val_loaders.append(
                val_dataset.build_dataloader(
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
            )
    # __________________________________________________________________________
    if args.get("test", False):
        print("Preparing test dataset...")
        for dataset, dataset_root, split_json in zip(
            datasets, dataset_roots, splits
        ):
            Dataset = get_dataloader(dataset)
            args["split_json"] = split_json

            test_dataset = Dataset(
                dataset_root=dataset_root,
                split="test",
                target_size=target_size,
                **args,
            )
            test_loaders.append(
                test_dataset.build_dataloader(
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
            )

    return train_loaders, val_loaders, test_loaders


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
    DEVICE = int(args["gpu"])

    # Generate log path

    logpath = os.path.join("logs", args["expname"])

    # logpath = os.path.join(logpath, f"_{get_last_exp(logpath)}")
    os.makedirs(logpath, exist_ok=True)

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
        pretrained_encoder=args.get("pretrained_encoder", False)
        | args.get("start_frozen", False),
        encoder_name=args.get("encoder_name"),
    )

    if args.get("start_frozen"):
        print("===================================")
        print("Warning: Encoder starting frozen!!!")
        print("===================================")

        freeze_encoder(model)

    callbacks = [
        RichModelSummary(max_depth=2),
        RichProgressBar(refresh_rate=1, leave=True),
    ]
    if epoch := args.get("unfreeze_epoch", None):
        callbacks.append(UnfreezeEncoder(epoch))

    trainer = pl.Trainer(
        # limit_train_batches=1,
        # limit_val_batches=100,
        max_epochs=args["epochs"],
        accelerator="gpu",
        devices=[0, 2, 3, 4],
        default_root_dir=logpath,
        callbacks=callbacks,
        # precision="bf16",
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args["ckpt_path"],
    )

    print("DONE!")


def test(args):
    """"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    DEVICE = int(args["gpu"])

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
    model_weights = torch.load(
        model_path, map_location=torch.device(f"cuda:{DEVICE}")
    )

    # Build a model and load weights
    # TODO: Get model from args
    model = ConvNext(
        tasks=args["tasks"],
        features=args["features"],
        pretrained_encoder=False,
        encoder_name=args.get("encoder_name"),
    ).to_gpu(DEVICE)
    model.load_state_dict(model_weights["state_dict"])

    # Load datasets
    # TODO: load by args
    # TODO: Fix this args override mess!

    args["features"] = [["image_l"], ["depth_l"]]
    # Test on each dataset

    print("Loading Virtual KITTI validation dataset")
    args["split_json"] = (
        "/home/luiz.decker/code/DepthTasks/configs/virtual_kitty_splits.json"
    )
    args["dataset_root"] = "/hadatasets/virtual_kitty/"
    virtual_kitti_val_ds = VirtualKitti(split="validation", **args)
    global_metrics_virtualkitti, sample_metrics_virtualkitti = eval_dataset(
        model, virtual_kitti_val_ds
    )
    del virtual_kitti_val_ds

    print("Loading NYU validation dataset")
    args["split_json"] = "/home/luiz.decker/code/DepthTasks/configs/nyu.json"
    args["dataset_root"] = "/hadatasets/nyu"
    nyu_val_ds = NYUDepthV2(split="validation", **args)
    global_metrics_nyu, sample_metrics_nyu = eval_dataset(model, nyu_val_ds)
    del nyu_val_ds

    print("Loading synscapes validation dataset")
    args["split_json"] = (
        "/home/luiz.decker/code/DepthTasks/configs/synscapes_split.json"
    )
    args["dataset_root"] = "/hadatasets/synscapes/"
    synscapes_val_ds = Synscapes(split="validation", **args)
    global_metrics_synscapes, sample_metrics_synscapes = eval_dataset(
        model, synscapes_val_ds
    )
    del synscapes_val_ds

    print("Loading HyperSim validation dataset")
    args["split_json"] = (
        "/home/luiz.decker/code/DepthTasks/configs/hypersim_splits_nonan.json"
    )
    args["dataset_root"] = "/hadatasets/hypersim"
    hypersim_val_ds = HyperSim(split="validation", **args)
    global_metrics_hypersim, sample_metrics_hypersim = eval_dataset(
        model, hypersim_val_ds
    )
    del hypersim_val_ds

    print("Loading KITTI validation dataset")
    args["split_json"] = (
        "/home/luiz.decker/code/DepthTasks/configs/kitti_eigen_val.txt"
    )
    args["dataset_root"] = "/hadatasets/kitti"
    kitti_val_ds = Kitti(split="validation", **args)
    global_metrics_kitti, sample_metrics_kitti = eval_dataset(
        model, kitti_val_ds
    )
    del kitti_val_ds

    print("Loading synthia validation dataset")
    args["split_json"] = (
        "/home/luiz.decker/code/DepthTasks/configs/synthia_splits.json"
    )
    args["dataset_root"] = "/hadatasets/SYNTHIA-AL/"
    synthia_val_ds = Synthia(split="validation", **args)
    global_metrics_synthia, sample_metrics_synthia = eval_dataset(
        model, synthia_val_ds
    )
    del synthia_val_ds

    print("Loading MidAir validation dataset")
    args["split_json"] = (
        "/home/luiz.decker/code/DepthTasks/configs/midair_splits.json"
    )
    args["dataset_root"] = "/hadatasets/midair/MidAir"
    midair_val_ds = MidAir(split="validation", **args)
    global_metrics_midair, sample_metrics_midair = eval_dataset(
        model, midair_val_ds
    )
    del midair_val_ds

    print("--->nyu", global_metrics_nyu)
    print("--->midair", global_metrics_midair)
    #   print("--->midair", "!!!MIDAIR DESATIVADO!!!")
    print("--->hypersim", global_metrics_hypersim)
    print("--->kitti", global_metrics_kitti)
    print("--->virtual kitti", global_metrics_virtualkitti)
    print("--->synscapes", global_metrics_synscapes)
    print("--->synthia", global_metrics_synthia)

    pd.DataFrame(
        {
            "nyu": global_metrics_nyu,
            "midair": global_metrics_midair,
            "hypersim": global_metrics_hypersim,
            "kitti": global_metrics_kitti,
            "virtual_kitti": global_metrics_virtualkitti,
            "synscapes": global_metrics_synscapes,
            "synthia": global_metrics_synthia,
        }
    ).to_csv(os.path.join(exp_path, "metrics.csv"))


if __name__ == "__main__":
    args = read_args()
    if args["evaluate_path"]:
        test(args)
    else:
        train(args)
    print("Done!")
