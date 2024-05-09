"Definition a abstract model and implementation of common model functions"

from pytorch_lightning import LightningModule
import torch
import torchmetrics


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

        self.train_loss = torchmetrics.MeanMetric()
        self.val_outputs = []

    def expand_shape(self, x):
        """Expands all tensors in the list x to have the same shape in the
        channels dimension.

        x's shape is expected to be (batch, channels, ...)
        """
        largest_feature_size = max([t.channels for t in self.tasks])
        for i, _x in enumerate(x):
            if _x.shape[1] == largest_feature_size:
                continue
            expanded_shape = list(_x.shape)
            expanded_shape[1] = largest_feature_size
            expanded = torch.zeros(expanded_shape).to(_x.device)
            expanded[:, : _x.shape[1], ...] = _x
            x[i] = expanded
        return x

    # Torchlightning steps ====================================================
    def training_step(self, batch, batch_idx, dataset_idx=None):
        """One step of training"""
        if dataset_idx is not None:
            batch = [batch]
        loss = 0
        for loader_batch in batch:
            x, y = loader_batch
            _y = self.forward(x)

            loss += self.compute_loss(_y, y)
        loss = loss / len(batch)
        self.log("train_step_loss", loss.detach(), sync_dist=False)
        self.train_loss.update(loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        if dataset_idx is not None:
            batch = [batch]
        for loader_batch in batch:
            x, y = loader_batch

            _y = self.forward(x)

            metrics = self.compute_metric(_y, y)
            metrics = {
                ("val_step_" + name): val for name, val in metrics.items()
            }
            self.val_outputs.append(metrics)
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
                        _metrics[f"test_{task.name}_{metric_name}"] = (
                            metric.compute().mean().detach()
                        )
                        metric.reset()
            self.log_dict(_metrics, sync_dist=True)

    def on_training_epoch_end(self):
        """"""

        with torch.no_grad():
            _metrics = {}
            _metrics["train_loss"] = self.train_loss.compute()
            self.train_loss.reset()

        self.log_dict(_metrics, logger=True, prog_bar=True, sync_dist=False)

    def on_validation_epoch_end(self):
        """"""
        _metrics = {}

        for task in self.tasks:
            if self.val_outputs and isinstance(self.val_outputs[0], dict):
                task_losses = [
                    output[f"val_step_{task.name}_loss"]
                    for output in self.val_outputs
                ]
                _metrics[f"val_{task.name}_loss"] = (
                    torch.stack(task_losses).mean().detach()
                )

            for metric_name, metric in self.metrics[task.name].items():
                if hasattr(metric, "compute"):
                    _metrics[f"val_{task.name}_{metric_name}"] = (
                        metric.compute().mean().detach()
                    )
                    metric.reset()
        self.log_dict(_metrics, logger=True, prog_bar=True, sync_dist=False)

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
            # Gets data from the specified task
            task_pred = pred[:, [task_index], ...]
            # Remove any expanded data
            task_pred = task_pred[:, :, : task.channels, ...]
            task_true = true[:, label_idx, ...]

            if task.train_on_disparity:
                # Gt converted to disparity on loss
                task_true = 1.0 / task_true
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
                assert task_true.max() <= (
                    task.num_classes
                ), "class gt larger than number of classes"

                task_pred = task_pred.swapaxes(-1, -3)
                task_pred = task_pred.reshape(-1, task.num_classes)
                task_true = task_true.flatten().long()

                ignore_mask = task_true != -1
                task_pred = task_pred[ignore_mask]
                task_true = task_true[ignore_mask]

            total_loss += self.metrics[task.name]["loss"](task_pred, task_true)

        return total_loss

    def compute_metric(self, pred, true):
        """Computes metric for all the tasks"""
        with torch.no_grad():
            # Tasks must be in order
            metrics = {}
            for task_index, task in enumerate(self.tasks):
                label_idx = [
                    self.features[1].index(feat) for feat in task.features
                ]
                task_pred = pred[:, [task_index], ...]
                task_pred = task_pred[:, :, : task.channels, ...]
                task_true = true[:, label_idx, ...]

                if task.train_on_disparity:
                    # Reconstruct the image for metric computing
                    task_true = 1.0 / task_true

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
                        _metrics[f"{task.name}_{metric_name}"] = (
                            batch_metric.detach()
                        )
                    metrics.update(_metrics)

                    continue

                task_metrics = self.metrics[task.name]
                _metrics = {}
                for metric_name, metric in task_metrics.items():
                    if metric_name == "loss" and hasattr(task, "num_classes"):
                        _metrics[metric_name] = metric(
                            task_pred.squeeze(dim=1)
                            .swapaxes(-1, -3)
                            .reshape(-1, task.num_classes)
                            .float(),
                            task_true.squeeze(dim=1).flatten().long(),
                        ).detach()
                        continue

                    _metrics[metric_name] = metric(
                        task_pred.squeeze(dim=1), task_true.squeeze(dim=1)
                    ).detach()

                for metric in _metrics:
                    value = _metrics[metric]
                    name = f"{task.name}_{metric}"
                    metrics[name] = value

        return metrics

    def to_gpu(self, gpu=0):  # I need to remove this
        self = self.cuda(gpu)
        [decoder.cuda(gpu) for decoder in self.decoders]
        return self
