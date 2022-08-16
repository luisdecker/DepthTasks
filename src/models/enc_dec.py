"Basic encoder-decoder"

from model import Model

import torch
import torch.nn as nn
from task import DenseRegression


class Encoder(nn.Module):
    """Encoder"""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

    def forward(self, x):

        # Swap axis
        x = torch.swapaxes(x, 1, 2)
        # apply convs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.swapaxes(x, 1, 2)
        return x


class Decoder(nn.Module):
    "Decoder"

    def __init__(self) -> None:
        super().__init__()

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3
        )
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3
        )
        self.upconv3 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=3
        )

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)

        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)

        x = torch.swapaxes(x, 1, 2)


class EncoderDecoder(Model):
    """Encoder-decoder"""

    def __init__(self, *args):
        super().__init__(*args)

        self.encoder = Encoder()

        self.decoder = Decoder()

        self.task = DenseRegression(name="autoencode")

    def forward(self, x):
        "Forward step"
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compute_loss(self, _y, y):
        "Computes the autoencoding loss"
        return self.task.loss(_y, y)
