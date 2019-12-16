from torch import nn
from torch.nn.utils import spectral_norm


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super(Conv2d, self).__init__()
        self.layer = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                             stride=stride, padding=padding))

    def forward(self, x):
        return self.layer(x)
