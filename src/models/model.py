"Definition a abstract model and implementation of common model functions"

from ctypes.wintypes import tagPOINT
from pytorch_lightning import LightningModule
import torch


class Model(LightningModule):
    """Abstract Model"""

    def __init__(self, **args):
        super().__init__()

        # Configure each one of the models tasks
        self.tasks = args["tasks"]

        metrics = {}
        for task in self.tasks:
            metrics[task.name] = torch.nn.ModuleDict(task.metric)
            metrics[task.name]["loss"] = task.loss

        self.metrics = torch.nn.ModuleDict(metrics)

        self.features = args["features"]

        self.savepath = args.get("savepath")

    def training_step(self, batch, batch_i):
        """One step of training"""
        x, y = batch
        _y = self.forward(x)

        loss = self.compute_loss(_y, y)
        self.log("train_step_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _y = self.forward(x)

        metrics = self.compute_metric(_y, y)
        metrics = {("val_step_" + name): val for name, val in metrics.items()}
        # self.log_dict(metrics, logger=True)

        return metrics

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self, outputs=None):
        if outputs:
            _metrics = {}

            for task in self.tasks:
                if outputs and isinstance(outputs[0], dict):
                    task_losses = [
                        output[f"val_step_{task.name}_loss"]
                        for output in outputs
                    ]
                    _metrics[f"test_{task.name}_loss"] = torch.stack(
                        task_losses
                    ).mean()

                for metric_name, metric in self.metrics[task.name].items():
                    if hasattr(metric, "compute"):
                        _metrics[
                            f"test_{task.name}_{metric_name}"
                        ] = metric.compute()
                        metric.reset()
            self.log_dict(_metrics)

    def training_epoch_end(self, outputs):
        """"""

        _metrics = {}
        if outputs and isinstance(outputs[0], dict):
            outputs = [output["loss"] for output in outputs]

        _metrics["train_loss"] = torch.stack(outputs).mean()

        self.log_dict(_metrics, logger=True, prog_bar=True)

    def validation_epoch_end(self, outputs):
        """"""

        _metrics = {}

        for task in self.tasks:
            if outputs and isinstance(outputs[0], dict):
                task_losses = [
                    output[f"val_step_{task.name}_loss"] for output in outputs
                ]
                _metrics[f"val_{task.name}_loss"] = torch.stack(
                    task_losses
                ).mean()

            for metric_name, metric in self.metrics[task.name].items():
                if hasattr(metric, "compute"):
                    _metrics[
                        f"val_{task.name}_{metric_name}"
                    ] = metric.compute()
                    metric.reset()
        self.log_dict(_metrics, logger=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[20, 80], gamma=0.1
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def compute_loss(self, pred, true):
        """Computes loss for all the tasks"""

        # Tasks must be in order
        total_loss = 0
        for task_index, task in enumerate(self.tasks):
            label_idx = [
                self.features[1].index(feat) for feat in task.features
            ]
            task_pred = pred[:, [task_index], ...]
            task_true = true[:, label_idx, ...]

            if task.train_on_disparity:
                # Gt converted to disparity on loss
                task_true = 1 / task_true

            has_nans = task_true.isnan().any()
            if task.mask_feature or has_nans:
                feat_mask = torch.ones_like(task_true).bool()
                nan_mask = ~task_true.isnan()
                if task.mask_feature:
                    feat_mask = true[
                        :, self.features[1].index(task.mask_feature), ...
                    ]
                    feat_mask = torch.unsqueeze(feat_mask, 1)
                    feat_mask = feat_mask / 255
                    feat_mask = feat_mask.bool()

                mask = nan_mask & feat_mask

                batch_loss = 0
                for m, t, p in zip(mask, task_true, task_pred):
                    batch_loss += self.metrics[task.name]["loss"](p[m], t[m])
                batch_loss /= true.shape[0]
                total_loss += batch_loss
                continue

            task_pred = task_pred.squeeze(dim=1)
            task_true = task_true.squeeze(dim=1)

            if hasattr(task, "num_classes"):
                task_true = task_true.flatten()
                task_pred = task_pred.swapaxes(-1, -3)
                task_pred = task_pred.reshape(-1, task.num_classes)

            total_loss += self.metrics[task.name]["loss"](task_pred, task_true)

        return total_loss

    def compute_metric(self, pred, true):
        """Computes metric for all the tasks"""

        # Tasks must be in order
        metrics = {}
        for task_index, task in enumerate(self.tasks):
            label_idx = [
                self.features[1].index(feat) for feat in task.features
            ]
            task_pred = pred[:, [task_index], ...]
            task_true = true[:, label_idx, ...]

            if task.train_on_disparity:
                # Reconstruct the image for metric computing
                task_pred = torch.nn.functional.relu(task_pred) + 1e-6
                task_pred = 1 / task_pred

            has_nans = task_true.isnan().any()
            if task.mask_feature or has_nans:
                nan_mask = ~task_true.isnan()
                feat_mask = torch.ones_like(task_true).bool()
                if task.mask_feature:
                    feat_mask = true[
                        :, self.features[1].index(task.mask_feature), ...
                    ]
                    feat_mask = torch.unsqueeze(mask, 1)
                    feat_mask = feat_mask / 255
                    feat_mask = feat_mask.bool()

                mask = nan_mask & feat_mask

                _metrics = {}
                for metric_name, metric in self.metrics[task.name].items():
                    batch_metric = 0
                    for t, p, m in zip(task_true, task_pred, mask):
                        batch_metric += metric(p[m], t[m])
                    batch_metric /= task_true.shape[0]
                    _metrics[f"{task.name}_{metric_name}"] = batch_metric
                metrics.update(_metrics)

                continue

            task_metrics = self.metrics[task.name]
            _metrics = {}
            for metric_name, metric in task_metrics.items():
                if metric_name == "loss" and hasattr(task, "num_classes"):
                    _metrics[metric_name] = metric(
                        task_pred.squeeze(dim=1)
                        .swapaxes(-1, -3)
                        .reshape(-1, task.num_classes),
                        task_true.squeeze(dim=1).flatten(),
                    )
                    continue

                _metrics[metric_name] = metric(
                    task_pred.squeeze(dim=1), task_true.squeeze(dim=1)
                )

            for metric in _metrics:
                value = _metrics[metric]
                name = f"{task.name}_{metric}"
                metrics[name] = value

        return metrics

    def to_gpu(self, gpu=0):
        self = self.cuda(gpu)
        [decoder.cuda(gpu) for decoder in self.decoders]

        return self
