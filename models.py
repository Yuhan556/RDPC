from copy import deepcopy
from turtle import forward

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable

from utils import *


class EncoderMNIST(nn.Module):
    def __init__(self, n_channel, latent_dim, quantize_latents, stochastic,
                 ls, input_size, L, q_limits):
        super(EncoderMNIST, self).__init__()

        self.n_channel = n_channel
        self.latent_dim = latent_dim
        self.quantize_latents = quantize_latents
        self.stochastic = stochastic
        self.ls = ls # layer scale: integer factor
        self.input_size = input_size # specified by dataset

        if self.quantize_latents:
            # Quantize to L uniformly spaced points between limits
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers)
        if self.stochastic:
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)
            else:
                raise ValueError('Quant. disabled')

        final_layer_width = int(self.ls*128)
        self.main = nn.Sequential(
            nn.Linear(self.input_size, 4*final_layer_width),
            nn.BatchNorm1d(4*final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(4*final_layer_width, 2*final_layer_width),
            nn.BatchNorm1d(2*final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(2*final_layer_width, final_layer_width),
            nn.BatchNorm1d(final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(final_layer_width, final_layer_width),
            nn.BatchNorm1d(final_layer_width),
            nn.LeakyReLU(),
            nn.Linear(final_layer_width, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Tanh()
        )
        self.final_layer_width = final_layer_width

    def encode(self, x):
        """
        Forward pass without quantizing or adding noise
        """
        x = x.view(-1, self.input_size)
        x = self.main(x)

        return x

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x

    def add_stochasticity(self, x, u):
        assert self.stochastic, f'Stochasticity disabled'

        return x + u

    def forward(self, x, u):
        x = x.view(-1, self.input_size)
        x = self.main(x)

        # in universal quantization, add noise then quantize
        if self.stochastic:
            x = x + u
        if self.quantize_latents:
            x = self.q(x)
        return x


class DecoderMNIST(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(DecoderMNIST, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.expand = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        if self.output_size == 784:
            self.deconvolve = nn.Sequential(
                nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 128, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 1, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f'No deconvolution defined for output size of {self.output_size}.')

    def forward(self, x, u):
        x = self.expand(x - u)
        x = x.view(-1, 32, 4, 4)
        x = self.deconvolve(x)

        return x



class DiscriminatorMNIST(nn.Module):
    def __init__(self, n_channel):
        super(DiscriminatorMNIST, self).__init__()
        self.n_channel = n_channel

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4096)
        x_recon = self.fc(x)
        return x_recon



class Q(nn.Module):
    def __init__(self, n_channel):
        super(Q, self).__init__()
        self.n_channel = n_channel

        self.main = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.main(x)
        return x




# ----------------------------

class EncoderSVHN(nn.Module):
    def __init__(self, n_channel, latent_dim, quantize_latents, stochastic,
                 ls, input_size, L, q_limits):
        super(EncoderSVHN, self).__init__()

        self.n_channel = n_channel
        self.latent_dim = latent_dim
        self.quantize_latents = quantize_latents
        self.stochastic = stochastic
        self.ls = ls # layer scale: integer factor
        self.input_size = input_size # specified by dataset

        if self.quantize_latents:
            centers = generate_centers(L, q_limits)
            self.q = Quantizer(centers=centers, sigma=2/L)
        if self.stochastic:
            if self.quantize_latents:
                self.alpha = (q_limits[1] - q_limits[0])/(L-1)
            else:
                raise ValueError('Quant. disabled')

        ilw = int(self.ls*64) # initial layer width
        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(ilw, 2*ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(2*ilw, 4*ilw, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
        )
        self.conv_flat_dim = ilw*4*4*4
        self.final = nn.Sequential(
            nn.Linear(self.conv_flat_dim, self.latent_dim),
            nn.Tanh(),
        )

    def encode(self, x):
        """
        Forward pass without quantizing or adding noise
        """
        x = self.main(x)
        x = x.view(-1, self.conv_flat_dim)
        x = self.final(x)

        return x

    def quantize(self, x):
        assert self.quantize_latents, f'Quantization disabled'
        x = self.q(x)

        return x

    def add_stochasticity(self, x, u):
        assert self.stochastic, f'Stochasticity disabled'

        return x + u

    def forward(self, x, u):
        x = self.main(x)
        x = x.view(-1, self.conv_flat_dim)
        x = self.final(x)

        # in universal quantization, add noise then quantize
        if self.stochastic:
            x = x + u
        if self.quantize_latents:
            x = self.q(x)

        return x


class DecoderSVHN(nn.Module):
    def __init__(self, latent_dim, output_size):
        super(DecoderSVHN, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size

        self.expand = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        if self.output_size == 3*32*32:
            self.deconvolve = nn.Sequential(
                nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 108, kernel_size=5, stride=2, padding=0),
                nn.BatchNorm2d(108),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(108, 128, kernel_size=5, stride=1, padding=0),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 3, kernel_size=4, stride=1, padding=0),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f'No deconvolution defined for output size of {self.output_size}.')

    def forward(self, x, u):
        x = self.expand(x - u)
        x = x.view(-1, 32, 4, 4)
        x = self.deconvolve(x)

        return x


class DiscriminatorSVHN(nn.Module):
    def __init__(self, args):
        super(DiscriminatorSVHN, self).__init__()

        self.n_channel = args.n_channel

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 4096)
        x = self.fc(x)
        return x

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, 
                     stride=stride, padding=2, bias=False)

class Q_SVHN(nn.Module):
    def __init__(self, n_channel):
        super(Q_SVHN, self).__init__()
        self.n_channel = n_channel

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        out = self.network(x)
        return out



# ----------------------------
def NetQ(args):
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        baseclass = Q
    elif args.dataset == 'svhn':
        baseclass = Q_SVHN

    class NetQ(baseclass):
        def __init__(self, n_channel):
            super().__init__(n_channel)

        def __str__(self):
            return f'Q for {self.__class__}\n' + super().__str__()
    return NetQ(args.n_channel)


def Encoder1(args):
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        baseclass = EncoderMNIST
    elif args.dataset == 'svhn':
        baseclass = EncoderSVHN
    else:
        raise ValueError('Unknown dataset')

    class Encoder1(baseclass):
        def __init__(self, args):
            super().__init__(args.n_channel, args.latent_dim_1, args.quantize,
                             args.stochastic, args.enc_layer_scale, args.input_size,
                             args.L_1, args.limits)

        def __str__(self):
            return f'Encoder1: latent_dim_1 = {self.latent_dim}, L_1 = {self.L}'

    return Encoder1(args)


def Decoder1(args):
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        baseclass = DecoderMNIST
    elif args.dataset == 'svhn':
        baseclass = DecoderSVHN
    elif args.dataset == 'lsun_bedrooms':
        baseclass = DecoderLSUN
    else:
        raise ValueError('Unknown dataset')

    class Decoder1(baseclass):
        def __init__(self, args):
            super().__init__(args.latent_dim_1, args.input_size)

        def __str__(self):
            return f'Decoder1: latent_dim_1 = {self.latent_dim}'

    return Decoder1(args)



def Discriminator(args):
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        baseclass = DiscriminatorMNIST
    elif args.dataset == 'svhn':
        baseclass = DiscriminatorSVHN
    elif args.dataset == 'lsun_bedrooms':
        baseclass = DiscriminatorLSUN
    else:
        raise ValueError('Unknown dataset')

    class Discriminator(baseclass):
        def __init__(self, n_channel):
            super().__init__(n_channel)

        def __str__(self):
            return f'Discriminator for {self.__class__}\n' + super().__str__()

    if baseclass == DiscriminatorMNIST:
        return Discriminator(args.n_channel)
    elif baseclass == DiscriminatorSVHN:
        return Discriminator(args)



class Quantizer(nn.Module):
    """
    Scalar Quantizer module
    Source: https://github.com/mitscha/dplc
    """
    def __init__(self, centers=[-1.0, 1.0], sigma=1.0):
        super(Quantizer, self).__init__()
        self.centers = centers
        self.sigma = sigma

    def forward(self, x):
        centers = x.data.new(self.centers)
        xsize = list(x.size())

        # Compute differentiable soft quantized version
        x = x.view(*(xsize + [1]))
        level_var = Variable(centers, requires_grad=False)
        dist = torch.abs(x-level_var)
        output = torch.sum(level_var * nn.functional.softmax(-self.sigma*dist, dim=-1), dim=-1)
        _, symbols = torch.min(dist.data, dim=-1, keepdim=True)
        for _ in range(len(xsize)): centers.unsqueeze(0) # in-place error
        centers = centers.expand(*(xsize + [len(self.centers)]))

        quant = centers.gather(-1, symbols.long()).squeeze_(dim=-1)

        # Replace activations in soft variable with hard quantized version
        output.data = quant

        return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')
    parser.add_argument('-input_size', type=int, default=784, help='hidden dimension (default: 128)')
    parser.add_argument('-latent_dim_1', type=int, default=6, help='hidden dimension of z (default: 8)')
    parser.add_argument('-L_1', type=int, default=3, help='number of quantization levels (default: 4')
    parser.add_argument('-n_channel', type=int, default=1, help='input channels (default: 1)')
    parser.add_argument('-quantize', type=bool, default=True, help='do quantization (default: True)')
    parser.add_argument('-stochastic', type=bool, default=True, help='add noise below quantization threshold (default: True)')
    parser.add_argument('-limits', nargs=2, type=float, default=(-1,1), help='quanitzation limits (default: (-1,1))')
    parser.add_argument('-enc_layer_scale', type=float, default=1., help='layer factor for encoder')

    args = parser.parse_args()
    disc = Discriminator(args)
    enc = Encoder1(args)
    dec = Decoder1(args)

    args_joint = deepcopy(args)
    vars(args_joint)['enc_layer_scale'] = 1
    enc_joint = Encoder1(args_joint)

    quantizer = Quantizer(centers=[-2, -1, 1, 2])
    print('--- Summary for Discriminator ---')
    summary(disc.cuda(), (1, 28, 28))
    print('--- Summary for Decoder ---')
    summary(dec.cuda(), (args.latent_dim_1,))
    print('--- Summary for Base Encoder ---')
    summary(enc.cuda(), (1, 28, 28))
