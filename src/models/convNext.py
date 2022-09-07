"Basic encoder-decoder"

from .model import Model

from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch.nn as nn
from .task import DenseRegression


class ConvNext(Model):
    """Encoder-decoder"""

    def __init__(self, **args):
        super().__init__(**args)

        self.convnext = (
            convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            if args.get("pretrained_encoder")
            else convnext_base()
        )

        self.encoder = self.convnext.features
        self.task = DenseRegression(name="autoencoder")
        # self.task.loss = self.task.loss.cuda()

    def forward(self, x):
        "Forward step"
        # x = (b, c, h, w)
        x = self.encoder[0](x) # -> x = (b, 128, h/4, w/4) ( ^*4 )
        x = self.encoder[1](x) # -> x = (b, 128, h/4, w/4)
        enc_skip_1 = x

        x = self.encoder[2](x) # -> x = (b, 256, h/8, w/8) ( ^*2 )
        x = self.encoder[3](x) # -> x = (b, 256, h/8, w/8)
        enc_skip_2 = x


        x = self.encoder[4](x) # -> x = (b, 512, h/16, w/16) ( ^*2 ) 
        x = self.encoder[5](x) # -> x = (b, 512, h/16, w/16)
        enc_skip_3 = x

        x = self.encoder[6](x) # -> x = (b, 1024, h/32, w/32) ( ^*2 )
        x = self.encoder[7](x)

        decoder_input = x



        return x

    def compute_loss(self, _y, y):
        "Computes the autoencoding loss"
        return self.task.loss(_y, y)
