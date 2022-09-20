"Definition of a model task"

import torchmetrics
import torch.nn as nn


class Task:
    """A network task"""

    def __init__(self, decoder, features, channels, **args) -> None:
        """"""

        self.decoder = decoder  # class of Decoder

        self.features = features  #  [label_feats : str]

        self.channels = channels

    def compute_metric(self, pred, true):
        return self.metric(pred, true)

    def compute_loss(self, pred, true):
        """Computes loss for a batch of predicitons"""
        return self.loss(pred, true)


class DenseRegression(Task):
    "Simple dense regression task"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        # Default loss if none specified
        self.loss = args.get("loss") or nn.MSELoss()

        # Metrics
        self.metric = args.get("metrics") or torchmetrics.MeanSquaredError(
            squared=False
        )

        self.name = args.get("name") or "dense_regression"
