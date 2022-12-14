"Definition a abstract model and implementation of common model functions"

from pytorch_lightning import LightningModule
import torch


class Model(LightningModule):
    """Abstract Model"""

    def __init__(self, **args):
        super().__init__()

        # Configure each one of the models tasks
        self.tasks = args["tasks"]

        self.features = args["features"]

    def training_step(self, batch, batch_i):
        """One step of training"""
        x, y = batch
        _y = self.forward(x)

        loss = self.compute_loss(_y, y)
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _y = self.forward(x)

        metrics = self.compute_metric(_y, y)
        for metric in metrics:
            self.log(metric, metrics[metric])

        return metrics

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

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

            if task.mask_feature:
                mask = true[:, self.features[1].index(task.mask_feature), ...]
                mask = torch.unsqueeze(mask, 1)
                mask = mask / 255
                task_true = task_true * mask 
                task_pred = task_pred * mask

            total_loss += task.compute_loss(task_pred, task_true)

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

            if task.mask_feature:
                mask = true[:, self.features[1].index(task.mask_feature), ...]
                mask = torch.unsqueeze(mask, 1)
                mask = mask / 255
                task_true = task_true * mask 
                task_pred = task_pred * mask

            metrics[task.name] = task.compute_metric(task_pred, task_true)

        return metrics

    def to_gpu(self, gpu=0):
        self = self.cuda(gpu)
        [decoder.cuda(gpu) for decoder in self.decoders]

        for task in self.tasks:
            task.loss = task.loss.cuda(gpu)
            task.metric = task.metric.cuda(gpu)
        return self
