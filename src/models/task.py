"Definition of a model task"

from functools import partial
import torchmetrics
import torch.nn as nn
from torchgeometry.losses import SSIM

from models.losses import GlobalMeanRemovedLoss


def get_task(task: str):
    "Gets a task class from a string"
    available_tasks = {"dense_regression": DenseRegression}
    return available_tasks[task.lower()]


class Task:
    """A network task"""

    def __init__(self, decoder, features, channels, **args) -> None:
        """"""

        self.decoder = decoder  # class of Decoder

        self.features = features  #  [label_feats : str]

        self.channels = channels

        self.train_on_disparity = args.get("train_on_disparity", False)

    def compute_metric(self, pred, true):
        results = {}
        for metric in self.metric:
            results[metric] = self.metric[metric](pred, true)
        return results

    def compute_loss(self, pred, true):
        """Computes loss for a batch of predicitons"""
        return self.loss(pred, true)


class DenseRegression(Task):
    "Simple dense regression task"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        # Default loss if none specified
        self.loss = args.get("loss") or CombinedLoss()
        # Metrics
        self.metric = args.get("metrics") or {
            "rmse": torchmetrics.MeanSquaredError(squared=False)
        }

        self.name = args.get("name") or "dense_regression"

        self.mask_feature = args["mask_feature"]


class CombinedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        huber = nn.functional.huber_loss(a, b)
        ssim = SSIM(11, reduction="mean")(a, b)
        return huber + ssim
