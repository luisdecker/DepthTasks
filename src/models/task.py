"Definition of a model task"

import torchmetrics


class Task:
    """A network task"""

    def __init__(self, **args) -> None:

        # Loss function for this task
        self.loss = args["loss"]

        # Metrics to be evaluated considering this task
        self.metrics = args["metrics"]

        # The task name
        self.name = args["name"]

    def evaluate(self, y, _y):
        "Evaluates a task using the and metrics"

        computed_metrics = {}
        for metric in self.metrics:
            metric_name = f"{self.name}/{metric.name}"
            computed_metrics[metric_name] = metric(y, _y)
        return computed_metrics


class DenseRegression(Task):
    "Simple dense regression task"

    def __init__(self, **args) -> None:
        super().__init__(**args)

        # Default loss if none specified
        self.loss = args.get("loss", None) or torchmetrics.MeanSquaredError(
            squared=True
        )

        # Metrics
        self.metrics = args.get(
            "metrics", None
        ) or torchmetrics.MeanSquaredError(squared=False)

        self.name = args.get("name", None) or "dense_regression"
