"Definition a abstract model and implementation of common model functions"

from pytorch_lightning import LightningModule


class Model(LightningModule):
    """Abstract Model"""

    def __init__(self, **args):
        super().__init__(*args)

        # Configure each one of the models tasks
        self.tasks = args["tasks"]

        # Configure dataloaders
        self.dataloader = args["train_data"]

    def training_step(self, batch, batch_i):
        """One step of training"""
        x, y = batch
        _y = self.forward(x, y)
        # compute_loss is a function that should compute the loss for all the
        # tasks
        loss = self.compute_loss(_y, y)
        self.log("training_loss", loss)
