"Definition of a model task"

import torchmetrics


class Task:
    """A network task"""

    def __init__(self, decoder, features, channels, **args) -> None:
        """"""

        self.decoder = decoder  # class of Decoder

        self.features = features  #  [label_feats : str]

        self.channels = channels

    def evaluate(self, y, _y):
        "Evaluates a task using the metrics"

        computed_metrics = {}
        for metric in self.metrics:
            metric_name = f"{self.name}/{metric.name}"
            computed_metrics[metric_name] = metric(y, _y)
        return computed_metrics

    def compute_loss(self, pred, true):
        """Computes loss for a batch of predicitons"""
        # Split the outputs
        return self.loss(pred, true)


class DenseRegression(Task):
    "Simple dense regression task"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        # Default loss if none specified
        self.loss = args.get("loss") or torchmetrics.MeanSquaredError(
            squared=True
        )

        # Metrics
        self.metrics = args.get("metrics") or torchmetrics.MeanSquaredError(
            squared=False
        )

        self.name = args.get("name") or "dense_regression"
