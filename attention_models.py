from torch.nn import Module, Parameter
from torch.nn.functional import relu, softmax, log_softmax

from modules import Conv2d
import torch

from utils import slerp


class ReshapeSoftmax(Module):
    def __init__(self, use_log=True):
        super(ReshapeSoftmax, self).__init__()
        self.use_log = use_log

    def forward(self, x):
        if self.use_log:
            return log_softmax(x.flatten(start_dim=2), dim=2).view_as(x)
        else:
            return softmax(x.flatten(start_dim=2), dim=2).view_as(x)


class DCAM(Module):
    def __init__(self, in_channels, use_log_softmax):
        super(DCAM, self).__init__()

        out_channels = in_channels // 4

        self.conv0 = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
        self.conv3 = Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1)
        self.conv4 = Conv2d(out_channels * 4, in_channels, kernel_size=1, stride=1, padding=0)

        self.feat_conv = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.softmax = ReshapeSoftmax(use_log_softmax)

        self.eps = Parameter(torch.zeros(1))

    def forward(self, x):
        out0 = relu(self.conv0(x))
        out1 = relu(self.conv1(out0))
        out2 = relu(self.conv2(torch.cat([out0, out1], 1)))
        out3 = relu(self.conv3(torch.cat([out0, out1, out2], 1)))
        out4 = relu(self.conv4(torch.cat([out0, out1, out2, out3], 1)))

        alpha = self.softmax(out4)

        out = self.feat_conv(x)
        # return feature map for visualization
        return self.eps * (alpha * out) + x


class InterpolationLayer(Module):
    def __init__(self, interpolation_mode, out_channels):
        super(InterpolationLayer, self).__init__()
        self.tau = Parameter(torch.tensor([0.5]))
        self.interpolation_mode = interpolation_mode
        self.layer = Conv2d(out_channels * 2, out_channels, 1, stride=1, padding=0)

    def forward(self, alpha, beta):
        if self.interpolation_mode == 0:
            tau = self.tau.clamp(0, 1)
            return tau * alpha + (1 - tau) * beta
        elif self.interpolation_mode == 1:
            tau = self.tau.clamp(0, 1)
            return slerp(tau, beta.flatten(start_dim=1), alpha.flatten(start_dim=1)).view_as(alpha)
        elif self.interpolation_mode == 2:
            merged = torch.cat([alpha, beta], 1)
            return relu(self.layer(merged))
        else:
            assert 'Unsupported interpolation mode'


class MCAM(Module):
    def __init__(self, in_channels, use_log_softmax, interpolation_mode):
        super(MCAM, self).__init__()

        out_channels = in_channels // 4

        self.conv0 = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.conv2 = Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv3 = Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.conv4 = Conv2d(out_channels * 4, in_channels, kernel_size=1, stride=1, padding=0)

        self.softmax = ReshapeSoftmax(use_log_softmax)
        self.interpolation = InterpolationLayer(interpolation_mode, out_channels=in_channels)

        self.feat_conv = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.zeta = Parameter(torch.zeros(1))

    def forward(self, x, alpha):
        out0 = relu(self.conv0(x))
        out1 = relu(self.conv1(x))
        out2 = relu(self.conv2(x))
        out3 = relu(self.conv3(x))
        out4 = relu(self.conv4(torch.cat([out0, out1, out2, out3], 1)))

        beta = self.softmax(out4)

        beta_ = self.interpolation(alpha, beta)

        out = self.feat_conv(x)

        # return feature map for visualization
        return self.zeta * (beta_ * out) + x
