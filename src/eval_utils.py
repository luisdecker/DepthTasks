"Utils for evaluating a newtwork"
from metrics import get_metric


def eval_dataset(network, dataset):
    """Evaluates a network in all elements of a dataset. Returns the global
    metrics and a list with the metrics by sample"""

    computed_metrics = []

    metrics = {
        metric_name: get_metric(metric_name)()
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

    for sample in dataset:
        sample_metrics = {}
        image, gt = sample
        valid_gt_map = gt > 0

        pred = network(image.unsqueeze(dim=1)).squeeze(dim=1)

        valid_gt = gt[valid_gt_map]
        valid_pred = pred[valid_gt_map]

        for metric_name, metric in metrics.items():
            sample_metrics[metric_name] = metric(valid_pred, valid_gt)
        computed_metrics.append(sample_metrics)

    global_metrics = {
        name: metric.compute() for name, metric in metrics.items()
    }

    return global_metrics, computed_metrics
