"Basic encoder-decoder"

from .model import Model

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
from .task import DenseRegression



class ConvNext(Model):
    """Encoder-decoder"""

    def __init__(self, **args):
        super().__init__(**args)

        self.convnext = (
            convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            if args.get("pretrained_encoder")
            else convnext_tiny()
        )

        self.encoder = self.convnext.features
        
        for task in self.tasks:
            task.decoder = task.decoder(
                input_channels=1024, output_channels=task.channels
            )


    def forward(self, x):
        "Forward step"

        # remove inputs dimension
        x = x[:, 0, ...]

        # x = (b, c, h, w)

        partial_maps = []
        x = self.encoder[0](x)  # -> x = (b, 128, h/4, w/4) ( ^*4 )
        x = self.encoder[1](x)  # -> x = (b, 128, h/4, w/4)
        enc_skip_1 = x
        partial_maps.append(enc_skip_1)

        x = self.encoder[2](x)  # -> x = (b, 256, h/8, w/8) ( ^*2 )
        x = self.encoder[3](x)  # -> x = (b, 256, h/8, w/8)
        enc_skip_2 = x
        partial_maps.append(enc_skip_2)

        x = self.encoder[4](x)  # -> x = (b, 512, h/16, w/16) ( ^*2 )
        x = self.encoder[5](x)  # -> x = (b, 512, h/16, w/16)
        enc_skip_3 = x
        partial_maps.append(enc_skip_3)

        x = self.encoder[6](x)  # -> x = (b, 1024, h/32, w/32) ( ^*2 )
        x = self.encoder[7](x)

        x = [task.decoder(x, partial_maps) for task in self.tasks]
        

        return x

    def compute_loss(self, _y, y):
        "Computes the autoencoding loss"
        return self.task.loss(_y, y)
