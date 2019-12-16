import argparse
import os

from PIL import Image

from math import log10

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from multiprocessing import cpu_count
import tqdm

from generator import Generator

from utils import haze_images

from NYUv2Dataset import NYUv2Dataset


def load_checkpoint(args, generator):
    ckpt = torch.load(args.resume)
    generator.load_state_dict(ckpt['generator'])


def main(args):
    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    generator = Generator(3)

    generator.to(device)

    load_checkpoint(args, generator)

    if args.jit:
        generator = torch.jit.script(generator)
        torch.jit.save(generator, os.path.join(args.output, 'jit_' + os.path.basename(args.resume)))

    with torch.no_grad():
        if not args.img_path:
            dataset = NYUv2Dataset(args, False)
            loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers,
                                shuffle=False, pin_memory=True)

            avg_psnr = 0
            criterion = MSELoss()

            for sample, depth in tqdm(loader):
                sample = sample.to(device)
                depth = depth.to(device)

                haze_sample = haze_images(sample, depth)

                g_out = generator(haze_sample)

                mse = criterion(g_out, sample)
                psnr = 10 * log10(1 / mse.item())

                avg_psnr += psnr

            print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(loader)))
        else:
            im = Image.open(args.img_path)
            im = to_tensor(im).to(device)
            g_out = generator(im)

            save_image(g_out, os.path.join(args.output, 'dehazed_' + os.path.basename(args.img_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='Path to .mat file of dataset')
    parser.add_argument('--splits_path', type=str, help='Path to train_test_split.mat')
    parser.add_argument('--img_path', type=str, help='Path to img')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('--resume', type=str, help='Load checkpoint')
    parser.add_argument('--jit', action='store_true', help='Jit compile model and save to output dir')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Number of workers')

    args = parser.parse_args()
    main(args)