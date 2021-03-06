from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop, Normalize, ColorJitter
from torchvision.transforms.functional import resized_crop, to_pil_image, hflip, to_tensor
from scipy.io import loadmat
import h5py
import torch
import random
import numpy as np

from utils import haze_images


class NYUv2Dataset(Dataset):
    def __init__(self, args, indices, is_training, crop_size=(256, 256)):
        super(NYUv2Dataset, self).__init__()
        self.crop_size = crop_size
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        self.is_training = is_training

        with h5py.File(args.dataset_path, 'r') as data:
            #splits = loadmat(args.splits_path)
            self.images = np.array(data['images'])
            self.depths = np.array(data['depths'])

            self.images = np.transpose(self.images[indices], (0, 3, 2, 1))
            self.depths = np.expand_dims(np.transpose(self.depths[indices], (0, 2, 1)), axis=-1)

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

            params = RandomResizedCrop.get_params(image, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333))
            image = resized_crop(image, *params, self.crop_size)
            depth = resized_crop(depth, *params, self.crop_size)

            if random.random() < 0.5:
                image = hflip(image)
                depth = hflip(depth)

        image = to_tensor(image)
        depth = to_tensor(depth)

        hazy_image = haze_images(image, depth)

        if self.is_training:
            hazy_image += 0.01 * torch.randn_like(hazy_image)
            hazy_image.clamp_(0, 1)

        self.normalize(image)
        self.normalize(hazy_image)

        if self.is_training:
            return image, hazy_image
        else:
            return image, hazy_image, depth
