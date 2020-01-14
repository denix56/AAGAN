import torch
from torch.nn import LeakyReLU, Module, Sequential, Dropout
from modules import Conv2d


# Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class NLayerDiscriminator(Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw,),
            LeakyReLU(0.2, True)
        ]

        sequence += [Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = Sequential(*sequence)

    def forward(self, img_a, img_b):
        """Standard forward."""
        merged = torch.cat([img_a, img_b], 1)
        return self.model(merged)


class Discriminator(Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels * 2, 64, kernel_size=3, stride=2),
            LeakyReLU(0.2, True),
            Conv2d(64, 128, kernel_size=3, stride=2),
            LeakyReLU(0.2, True),
            Conv2d(128, 256, kernel_size=3, stride=2),
            LeakyReLU(0.2, True),
            Conv2d(256, 512, kernel_size=3, stride=2),
            LeakyReLU(0.2, True),
            Conv2d(512, 1, kernel_size=3, stride=1)
        )

    def forward(self, img_a, img_b):
        merged = torch.cat([img_a, img_b], 1)
        out = self.layers(merged)

        return out
