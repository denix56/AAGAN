import torch


def haze_images(imgs, depths, beta_min=0.6, beta_max=1.8, eta_min=0.5, eta_max=1):
    batch_size = depths.size(0)

    beta = torch.rand(batch_size, 1, 1, 1, device=imgs.device) * (beta_max - beta_min) + beta_min
    t = torch.exp(-beta * depths)

    eta = torch.rand(batch_size, 1, 1, 1, device=imgs.device) * (eta_max - eta_min) + eta_min
    A = eta.expand(batch_size, 3, t.size(2), t.size(3))
    I = imgs * t + A * (1 - t)
    return I