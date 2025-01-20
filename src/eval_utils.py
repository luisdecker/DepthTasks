"Utils for evaluating a newtwork"
from random import shuffle
import torch
from rich.progress import Progress
from metrics import get_metric, FunctionalMetrics
import numpy as np

from models import losses
from models.losses import GlobalMeanRemovedLoss


def normalize_depth(depth_map):
    "Normalizes a depth map between the values of 0 and 1"
    maxv = torch.max(depth_map)
    minv = torch.min(depth_map)
    normalized_map = (depth_map - minv) / (maxv - minv)
    return normalized_map


def eval_dataset(
    network, dataset, batch_size=1, functional=True, normalize=True, disparity=True, invert_pred = True
):
    """Evaluates a network in all elements of a dataset. Returns the global
    metrics and a list with the metrics by sample"""

    computed_metrics = []

    loader = dataset.build_dataloader(
        shuffle=False, batch_size=batch_size, num_workers=8
    )
    num_batches = len(loader)

    loader = iter(loader)

    if functional:

        metrics = {
            "ssi_a1": FunctionalMetrics.ssi_alpha1,
            "ssi_a2": FunctionalMetrics.ssi_alpha2,
            "ssi_a3": FunctionalMetrics.ssi_alpha3,
            "ssi_absrel": FunctionalMetrics.ssi_absrel,
            "a1": FunctionalMetrics.alpha1,
            "a2": FunctionalMetrics.alpha2,
            "a3": FunctionalMetrics.alpha3,
            "absrel": FunctionalMetrics.absrel,
            "loss": losses.MidasLoss(disparity=True),
        }
    else:
        metrics = {
            metric_name: get_metric(metric_name)().to(network.device)
            for metric_name in [
                "a1",
                "a2",
                "a3",
                "logrmse",
                "absrel",
                "squaredrel",
                "rmse",
            ]
        }

    with Progress() as p:
        batch_progress = p.add_task("Eval...", total=num_batches * batch_size)
        with torch.no_grad():
            while True:
                try:
                    image, gt = next(loader)
                    image = image.to(network.device)
                    gt = gt.to(network.device)[:, 0, 0, ...]

                    #                   all,task,feat,all
                    pred = network(image)[:, 0, 0, ...]

                    if disparity:
                        gt_invalid = gt <= 0
                        gt = 1.0 / gt
                        gt[gt_invalid] = -1

                    for sample in range(batch_size):

                        sample_metrics = {}
                        # Get one sample from batch
                        sample_pred = pred[sample]
                        sample_gt = gt[sample]
                        # Generate valid filter
                        valid_gt_map = sample_gt > 0
                        # Filter data
                        valid_gt = sample_gt[valid_gt_map]
                        valid_pred = sample_pred[valid_gt_map]

                        if normalize:
                            "Normalize both prediction and GT between 0 and 1"
                            valid_gt = normalize_depth(valid_gt)
                            valid_pred = normalize_depth(valid_pred)

                            if invert_pred:
                                valid_pred = 1 - valid_pred

                        # Compute each metric for the sample
                        for metric_name, metric in metrics.items():
                            if metric_name.startswith(("loss", "ssi")):
                                sample_metrics[metric_name] = metric(pred, gt)
                            else:
                                sample_metrics[metric_name] = metric(
                                    valid_pred, valid_gt
                                )
                        computed_metrics.append(sample_metrics)
                    p.update(batch_progress, advance=batch_size)
                except Exception as e:
                    print(e)
                    break

    if functional:
        global_metrics = {name: [] for name in metrics}
        for sample in computed_metrics:
            for metric in sample:
                global_metrics[metric].append(sample[metric])

        for metric in global_metrics:
            global_metrics[metric] = (
                torch.tensor(global_metrics[metric]).cpu().numpy().mean()
            )

    else:
        global_metrics = {
            name: metric.compute().cpu().numpy() for name, metric in metrics.items()
        }

    for metric, val in global_metrics.items():
        if val == np.nan:
            print(f"{metric} is nan!")
    return global_metrics, computed_metrics
