"""Depth estimation metrics"""

from typing import Any
import torch
from torchmetrics import Metric, MeanSquaredError, MeanSquaredLogError
from functools import partial


def get_metric(metric):
    """Gets a metric class by a string identifier"""

    return {
        "a1": get_alpha(1),
        "a2": get_alpha(2),
        "a3": get_alpha(3),
        "logrmse": MeanSquaredLogError,
        "absrel": AbsoluteRelative,
        "squaredrel": AbsoluteRelativeSquared,
        "rmse": RMSE,
    }[metric.lower()]


class AlphaError(Metric):
    """Compútes the alpha error metric in a given power, considering 1.25 as
    threshold"""

    def __init__(self, power):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.power = power

    def update(self, pred, gt):
        "Updates the internal states"
        thresh = torch.max((gt / pred), (pred / gt))
        sum = (thresh < (1.25**self.power)).float().sum()
        self.sum += sum
        self.n += pred.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n


def get_alpha(power):
    class AlphaX(AlphaError):
        def __init__(self):
            super().__init__(power)

    return AlphaX


class RMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(squared=False)


class LogRMSE(Metric):
    """Computes the rmse of two depth maps in log space"""

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.))
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, gt):
        "Updates the internal states"
        self.sum += torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2)).sum()
        self.n += gt.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n


class AbsoluteRelative(Metric):
    """Computes the absolute relative error between two depth maps."""

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.))
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, gt):
        "Updates the internal states"
        self.sum += (torch.abs(gt - pred) / gt).sum()
        self.n += gt.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n


class AbsoluteRelativeSquared(Metric):
    """Computes the squared absolute relative error between two depth maps."""

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.))
        self.add_state("n", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, pred, gt):
        "Updates the internal states"
        self.sum += ((gt - pred) ** 2 / gt).sum()
        self.n += gt.numel()

    def compute(self):
        "Computes the final metric"
        return self.sum / self.n
