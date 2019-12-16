import torch
from torch.nn import LeakyReLU, Module, Sequential
from modules import Conv2d


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

    def forward(self, real_x, fake_x):
        merged = torch.cat([real_x, fake_x], 1)
        out = self.layers(merged)

        return out
