"Utils for evaluating a newtwork"
from random import shuffle
import torch
from rich.progress import Progress
from metrics import get_metric
import numpy as np

from models.losses import GlobalMeanRemovedLoss


def eval_dataset(network, dataset, batch_size=1):
    """Evaluates a network in all elements of a dataset. Returns the global
    metrics and a list with the metrics by sample"""

    computed_metrics = []

    loader = dataset.build_dataloader(
        shuffle=False, batch_size=batch_size, num_workers=8
    )
    num_batches = len(loader)

    loader = iter(loader)

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
                    if network.tasks[0].train_on_disparity:
                        gt = 1 / gt

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
                        # Compute each metric for the sample
                        for metric_name, metric in metrics.items():
                            sample_metrics[metric_name] = metric(
                                valid_pred, valid_gt
                            )
                        computed_metrics.append(sample_metrics)
                    p.update(batch_progress, advance=batch_size)
                except:
                    break

    global_metrics = {
        name: metric.compute().cpu().numpy()
        for name, metric in metrics.items()
    }

    for metric, val in global_metrics.items():
        if val == np.nan:
            print (f"{metric} is nan!")
    return global_metrics, computed_metrics
