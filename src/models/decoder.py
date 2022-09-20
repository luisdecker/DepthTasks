"Decoders to be appended as tasks to a feature extractor"

import torch.nn as nn

from models.layers import UpsampleConv


class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels, skip_dimensions=None):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.skip_dimensions = skip_dimensions


class SimpleDecoder(Decoder):
    def __init__(self, input_channels, output_channels):
        super().__init__(input_channels, output_channels)

        self.upconv1 = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=3,
        )
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=3
        )
        self.upconv3 = nn.ConvTranspose2d(
            in_channels=32, out_channels=output_channels, kernel_size=3
        )

    def forward(self, x, encoder_partial_maps):

        x = self.upconv1(x)
        x = x + encoder_partial_maps[-1]
        x = self.upconv2(x)
        x = x + encoder_partial_maps[-2]
        x = self.upconv3(x)

        return x


class UnetDecoder(Decoder):
    """Traditional unet decoder (upsample with skip connections)"""

    def __init__(self, input_channels, output_channels, skip_dimensions):
        super().__init__(input_channels, output_channels, skip_dimensions)

        assert (
            type(skip_dimensions) == list
        ), "skip_dimensions must be a list with [(map_channels, map_factor)]"

        self.activation = nn.GELU  # TODO: Parametrize this!

        self.layers = []
        # Reverse since we are decoding
        skip_dimensions = list(reversed(skip_dimensions))

        for i, params in enumerate(skip_dimensions):
            channels, factor = params

            # Check if we are in the last upscale (final data)
            channels = channels if channels != -1 else output_channels

            # Check if we are in the first upscale (from encoder)
            in_channels = (
                input_channels if i == 0 else skip_dimensions[i - 1][0]
            )

            # Create each decoding stage
            stage_layers = []
            num_upscales = factor // 2

            for j in range(num_upscales):

                stage_layers.append(
                    UpsampleConv(
                        in_channels,
                        # Keep channels until dimensions are correct
                        channels if j == num_upscales - 1 else in_channels,
                    )
                )
            self.layers.append(stage_layers)

        self.layers = nn.ModuleList(
            [nn.Sequential(*stage) for stage in self.layers]
        )
        self.apply(self._init_weights)

    def forward(self, x, encoder_partial_maps):

        encoder_partial_maps = list(reversed(encoder_partial_maps))

        for i, stage in enumerate(self.layers[:-1]):
            x = stage(x)
            x = x + encoder_partial_maps[i]

        # Last layer does not have a skip connection
        x = self.layers[-1](x)

        return x

    def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

