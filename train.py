from multiprocessing import cpu_count
import time
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss, L1Loss
import torchvision.models as models
from torchvision.utils import make_grid
from torchvision.transforms import Normalize

from generator import Generator
from discriminator import Discriminator
from loss import *
from NYUv2Dataset import NYUv2Dataset


def save_checkpoint(args, generator, discriminator, g_optimizer, d_optimizer, g_step_lr, d_step_lr, step):
    ckpt_path = os.path.join(args.logdir, 'model.ckpt-{}.pt'.format(step))

    torch.save({'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'g_step_lr': g_step_lr.state_dict(),
                'd_step_lr': d_step_lr.state_dict(),
                'step': step}, ckpt_path)


def load_checkpoint(args, generator, discriminator, g_optimizer, d_optimizer, g_step_lr, d_step_lr):
    ckpt = torch.load(args.resume)
    generator.load_state_dict(ckpt['generator'])
    discriminator.load_state_dict(ckpt['discriminator'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_step_lr.load_state_dict(ckpt['g_step_lr'])
    d_step_lr.load_state_dict(ckpt['d_step_lr'])

    print('Checkpoint loaded')

    return ckpt['step']


def train(args):
    os.makedirs(args.logdir, exist_ok=True)

    num_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    torch.backends.cudnn.benchmark = args.use_cudnn_benchmark

    dataset = NYUv2Dataset(args, True)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=True, pin_memory=True)

    test_dataset = NYUv2Dataset(args, False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0,
                        shuffle=True, pin_memory=True)

    generator = Generator(3, args.use_log_softmax)
    discriminator = Discriminator(3)

    print(generator)
    print(discriminator)

    generator.to(device)
    discriminator.to(device)

    vgg19 = models.vgg19(pretrained=True, progress=True).features
    vgg19.to(device)

    g_optimizer = Adam(generator.parameters(), lr=args.generator_learning_rate)
    d_optimizer = Adam(discriminator.parameters(), lr=args.discriminator_learning_rate)

    content_loss = L1Loss()
    perceptual_loss = MSELoss()
    ralsgan_loss = RaLSGANLoss()

    g_step_lr = StepLR(g_optimizer, args.lr_decay_step, gamma=0.1)
    d_step_lr = StepLR(d_optimizer, args.lr_decay_step, gamma=0.1)

    global_step = 0

    if args.resume:
        global_step = load_checkpoint(args, generator, discriminator, g_optimizer, d_optimizer, g_step_lr, d_step_lr)

    writer = SummaryWriter(log_dir=args.logdir)

    img, depth = next(iter(loader))
    img = img.to(device)

    writer.add_graph(generator, img)
    #writer.add_graph(discriminator, (img, generator(img)))

    del img, depth

    inv_normalize = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                              std=[1/0.229, 1/0.224, 1/0.225], inplace=True)

    feat_maps = {}

    def get_hook(name):
        def hook(module, input, output):
            feat_maps[name] = output.detach().view(-1, 1, output.size(2), output.size(3))

        return hook

    for epoch in range(args.epochs):
        print('Epoch {}'.format(epoch))

        for samples, hazy_samples in loader:
            start = time.time()

            samples = samples.to(device)
            hazy_samples = hazy_samples.to(device)

            g_outputs = generator(hazy_samples)

            global_step += 1

            out_str = 'Step {}'.format(global_step)

            if global_step >= args.gen_train_start_step:
                g_optimizer.zero_grad()

                g_loss_c = content_loss(samples, g_outputs)
                g_loss_p = perceptual_loss(vgg19(samples), vgg19(g_outputs))

                C_ij = discriminator(hazy_samples, g_outputs)
                C_ik = discriminator(hazy_samples, samples)

                g_loss_r = ralsgan_loss(C_ij, C_ik)

                g_loss = args.lambda_c * g_loss_c + args.lambda_p * g_loss_p + args.lambda_r * g_loss_r
                g_loss.backward()
                g_optimizer.step()
                g_step_lr.step()

                out_str += ', g_loss {:.4f}, content_loss {:.4f}, perceptual_loss {:.4f}, ralsgan_loss {:.4f}'.format(
                    g_loss.item(), g_loss_c.item(), g_loss_p.item(), g_loss_r.item())

            if global_step >= args.discr_train_start_step:
                d_optimizer.zero_grad()

                C_ij = discriminator(hazy_samples, samples)
                C_ik = discriminator(hazy_samples, g_outputs.detach())

                d_loss = ralsgan_loss(C_ij, C_ik)
                d_loss.backward()
                d_optimizer.step()
                d_step_lr.step()

                out_str += ', d_loss {:.4f}'.format(d_loss.item())

            print(out_str + ', time: {:.2f} s'.format(time.time() - start))

            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, generator, discriminator, g_optimizer, d_optimizer,
                                g_step_lr, d_step_lr, global_step)

            if global_step % args.summary_step == 0:
                with torch.no_grad():
                    img, hazed_img, depth = next(iter(test_loader))
                    img = img.to(device)
                    hazed_img = hazed_img.to(device)
                    depth = depth.to(device)

                    writer.add_scalar('generator/generator_loss', g_loss, global_step)
                    writer.add_scalar('generator/content_loss', g_loss_c, global_step)
                    writer.add_scalar('generator/perceptual_loss', g_loss_p, global_step)
                    writer.add_scalar('generator/ralsgan_loss', g_loss_r, global_step)

                    if global_step > args.discr_train_start_step:
                        writer.add_scalar('discriminator/discriminator_loss', d_loss, global_step)

                    layers = generator.feat_map_layers()
                    map_names = ['prior_feature_map', 'posterior_feature_map', 'novel_feature_map']
                    for name, layer in zip(map_names, layers):
                        layer.register_forward_hook(get_hook(name))

                    test_g_out = generator(hazed_img)

                    img.squeeze_(0)
                    hazed_img.squeeze_(0)
                    depth.squeeze_(0)
                    test_g_out.squeeze_(0)

                    inv_normalize(img)
                    inv_normalize(hazed_img)
                    inv_normalize(test_g_out)

                    writer.add_image('GT', img, global_step)
                    writer.add_image('Depth_map', depth, global_step)
                    writer.add_image('Hazed', hazed_img, global_step)
                    writer.add_image('Dehazed', test_g_out, global_step)

                    for name, feat_map in feat_maps.items():
                        writer.add_image(name, make_grid(feat_map, normalize=True, scale_each=True), global_step)

                    for name, value in generator.named_parameters():
                        writer.add_histogram('generator/' + name.replace('.', '/'), value, global_step)

                    for name, value in discriminator.named_parameters():
                        writer.add_histogram('discriminator/' + name.replace('.', '/'), value, global_step)

                    del img, depth, hazed_img, test_g_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='Path to .mat file of dataset')
    parser.add_argument('--splits_path', type=str, help='Path to train_test_split.mat')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('--use_cuda', type=bool, default=True, help= 'Use CUDA')
    parser.add_argument('--use_cudnn_benchmark', action='store_true', help='Use CuDNN benchmark')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Number of workers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--logdir', type=str, default='logdir', help='Directory to save checkpoints and '
                                                                     'tensorboard output')
    parser.add_argument('--gen_train_start_step', type=int, default=0, help='Step to start training generator')
    parser.add_argument('--discr_train_start_step', type=int, default=0, help='Step to start training discriminator')
    parser.add_argument('-glr', '--generator_learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('-dlr', '--discriminator_learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--checkpoint_step', type=int, default=5000, help='Checkpoint save interval')
    parser.add_argument('--summary_step', type=int, default=100, help='Save info to Tensorboard')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    parser.add_argument('--lambda_c', type=float, default=100, help='Content loss coef')
    parser.add_argument('--lambda_p', type=float, default=100, help='Perceptual loss coef')
    parser.add_argument('--lambda_r', type=float, default=0.1, help='RaLSGAN loss coef')
    parser.add_argument('--lr_decay_step', type=int, default=100000, help='Decrease LR every nth step')
    parser.add_argument('--use_log_softmax', type=bool, default=True, help='Use log_softmax activation in '
                                                                           'attention models instead of softmax')
    args = parser.parse_args()
    train(args)
