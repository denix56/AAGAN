from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import crop, to_pil_image, hflip, to_tensor
from scipy.io import loadmat
import h5py
import torch
import random
import numpy as np


class NYUv2Dataset(Dataset):
    def __init__(self, args, is_training, crop_size=(256, 256)):
        super(NYUv2Dataset, self).__init__()
        self.is_training = is_training
        self.crop_size = crop_size

        with h5py.File(args.dataset_path, 'r') as data:
            splits = loadmat(args.splits_path)
            self.images = np.array(data['images'])
            self.depths = np.array(data['depths'])
            if is_training:
                idx = np.array(splits['trainNdxs']).reshape(-1) - 1
            else:
                idx = np.array(splits['testNdxs']).reshape(-1) - 1

            self.images = np.transpose(self.images[idx], (0, 3, 2, 1))
            self.depths = np.expand_dims(np.transpose(self.depths[idx], (0, 2, 1)), axis=-1)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        depth = self.depths[idx]

        if self.is_training:
            image = to_pil_image(image)
            depth = to_pil_image(depth)

            params = RandomCrop.get_params(image, self.crop_size)
            image = crop(image, *params)
            depth = crop(depth, *params)

            if random.random() < 0.5:
                image = hflip(image)
                depth = hflip(depth)

        image = to_tensor(image)
        depth = to_tensor(depth)

        return image, depth
