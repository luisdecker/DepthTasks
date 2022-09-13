import torch.nn as nn


class UpsampleConv(nn.Module):
    "Upsample followed by convolution"

    def __init__(self, in_channels, out_channels, **args) -> None:
        super().__init__()

        kernel_size = args.get("kernel_size", 3)
        stride = args.get("stride", 1)
        padding = args.get("padding", 1)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):

        x = self.upsample(x)
        x = self.conv(x)

        return x
