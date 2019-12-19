import torch
from torch.nn import ReLU, Module, Sequential
from torch.nn.functional import relu, interpolate

from modules import Conv2d
from attention_models import DCAM, MCAM


class Generator(Module):
    def __init__(self, in_channels, use_log_softmax):
        super(Generator, self).__init__()
        self.use_log_softmax = use_log_softmax

        # Encoder
        self.en_conv = Conv2d(in_channels, 32, kernel_size=3, stride=1)

        self.en_block1 = ENBlock(32, True)

        self.en_block2 = ENBlock(64)
        self.en_block3 = ENBlock(128)
        self.en_block4 = ENBlock(256)
        self.en_block5 = ENBlock(512)

        self.res_block1 = ResBlock(1024)

        # Decoder
        self.res_block2 = ResBlock(1024)

        self.de_block1 = DEBlock(1024)
        self.de_block2 = DEBlock(512)
        self.de_block3 = DEBlock(256)
        self.de_block4 = DEBlock(128)

        self.de_block5 = DEBlock(64, True)

        self.de_conv = Conv2d(32, 3, kernel_size=3, stride=1)

    def forward(self, x):
        # Encoder
        out0 = self.en_conv(x)
        out1, out_dcam = self.en_block1(out0)
        out2 = self.en_block2(out1)
        out3 = self.en_block3(out2)
        out4 = self.en_block4(out3)
        out5 = self.en_block5(out4)

        res1 = self.res_block1(out5)

        # Decoder
        res2 = self.res_block2(res1)
        de_out1 = self.de_block1(res2)
        de_out1 += out4
        de_out2 = self.de_block2(de_out1)
        de_out2 += out3
        de_out3 = self.de_block3(de_out2)
        de_out3 += out2
        de_out4 = self.de_block4(de_out3)
        de_out4 += out1

        de_out5, out_mcam = self.de_block5(de_out4, out_dcam)

        de_out5 += out0

        de_out = self.de_conv(de_out5)
        de_out += x

        return torch.tanh(de_out)

    @torch.jit.ignore
    def feat_map_layers(self):
        return [self.en_block1.dcam.softmax, self.de_block5.mcam.softmax, self.de_block5.mcam.interpolation]


class ResBlock(Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.layer =  Sequential(
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1),
            ReLU(inplace=True),
            Conv2d(in_channels, in_channels, kernel_size=3, stride=1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out + x


class ENBlock(Module):
    def __init__(self, in_channels, with_dcam=False, use_log_softmax=True):
        super(ENBlock, self).__init__()

        self.conv1 = Conv2d(in_channels, in_channels, kernel_size=3, stride=2)
        self.conv2 = Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1)

        if with_dcam:
            self.dcam = DCAM(in_channels, use_log_softmax)
        else:
            self.dcam = None

    def forward(self, x):
        out = self.conv1(relu(x))

        if self.dcam:
            dcam_out = self.dcam(out)
            out = self.conv2(relu(dcam_out))

            return out, dcam_out
        else:
            return self.conv2(relu(out, inplace=True))


class DEBlock(Module):
    def __init__(self, in_channels, with_mcam=False, use_log_softmax=True):
        super(DEBlock, self).__init__()
        out_channels = in_channels // 2

        if with_mcam:
            self.mcam = MCAM(out_channels, use_log_softmax)
        else:
            self.mcam = None

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x, dcam=None):
        out = self.conv1(relu(x))
        if self.mcam:
            mcam_out = self.mcam(out, dcam)
            out = interpolate(mcam_out, scale_factor=2, mode='nearest')
            out = self.conv2(out)
            return out, mcam_out
        else:
            out = interpolate(out, scale_factor=2, mode='nearest')
            out = self.conv2(out)
            return out
