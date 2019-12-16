import torch
from torch.nn import Module


class RaLSGANLoss(Module):
    def __init__(self):
        super(RaLSGANLoss, self).__init__()

    def forward(self, C_ij, C_ik):
        return torch.mean((C_ij - C_ik.mean() - 1) ** 2) + torch.mean((C_ik - C_ij.mean() + 1) ** 2)
