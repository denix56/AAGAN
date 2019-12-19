import torch


def haze_images(img, depth, beta_min=0.6, beta_max=1.8, eta_min=0.5, eta_max=1):
    depths_min = depth.min()
    depths_max = depth.max()

    depths_min = depths_min.view(1, 1, 1)
    depths_max = depths_max.view(1, 1, 1)

    depth = (depth - depths_min) / (depths_max - depths_min + 1e-10)

    beta = torch.rand(1, 1, 1) * (beta_max - beta_min) + beta_min
    t = torch.exp(-beta * depth)

    eta = torch.rand(1, 1, 1) * (eta_max - eta_min) + eta_min
    A = eta.expand(3, t.size(1), t.size(2))
    I = img * t + A * (1 - t)
    return I
