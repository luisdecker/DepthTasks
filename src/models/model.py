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

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=1)

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

            total_loss += task.compute_loss(task_pred, task_true)

        return total_loss

    def to_gpu(self, gpu=0):
        self = self.cuda(gpu)
        for task in self.tasks:
            task.loss = task.loss.cuda(gpu)
            task.encoder = task.decoder.cuda(gpu)
        return self
