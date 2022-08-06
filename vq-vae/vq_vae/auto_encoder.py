from __future__ import print_function
import abc

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import pdb

from .nearest_embed import NearestEmbed, NearestEmbedEMA


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class AutoEncoder(nn.Module):
    """
    Standard CNN based autoencoder. 
    """


    def __init__(self, model_d, num_channels=3):
        super(AutoEncoder, self).__init__()

        self.model_d = model_d
        self.num_channels = num_channels
        self.model_type = 'ae'

        d = 64

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, 2 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * d, 4 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * d, 8 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * d, 16 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16 * d),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16 * d, 8 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8 * d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8 * d, 4 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * d, 2 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )

        # FC layer to get features down to size. 
        self.fc = nn.Sequential(
                nn.Linear(16 * d * 4 * 4, 4096), 
                nn.ReLU(inplace=True),
                nn.Linear(4096, model_d),
                nn.ReLU()
        )
    
        # To get back starting vector for decoder. 
        self.decoder_fc = nn.Sequential(
                nn.Linear(model_d, 4096), 
                nn.ReLU(inplace=True), 
                nn.Linear(4096, 16 * d * 4 * 4), 
                nn.ReLU()
        )

        self.mse = 0

    def encode(self, x):
        h1 = self.encoder(x).view(x.shape[0], -1)
        return self.fc(h1)

    def decode(self, z):
        h1 = self.decoder_fc(z).view(z.shape[0], 16 * 64, 4, 4)
        return torch.tanh(self.decoder(h1))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def loss_function(self, x, recon_x, z):
        self.mse = F.mse_loss(recon_x, x)

        return self.mse

    def latest_losses(self):
        return {'mse': self.mse}

class VAE(nn.Module):
    """Variational AutoEncoder for MNIST
       Taken from pytorch/examples: https://github.com/pytorch/examples/tree/master/vae"""

    def __init__(self, model_d, kl_coef=1, num_channels=3):
        super(VAE, self).__init__()

        self.model_d = model_d
        self.kl_coef = kl_coef
        self.num_channels = num_channels
        self.model_type = 'vae'

        d = 64

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, 2 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * d, 4 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * d, 8 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8 * d),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * d, 16 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16 * d),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16 * d, 8 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8 * d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8 * d, 4 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * d, 2 * d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1),
        )

        # FC layer to get features down to size. 
        self.fc = nn.Sequential(
                nn.Linear(16 * d * 4 * 4, 4096), 
                nn.ReLU(inplace=True), 
        )

        # To predict latent code. 
        self.fc_mu = nn.Linear(4096, model_d)
        self.fc_std = nn.Linear(4096, model_d)
    
        # To get back starting vector for decoder. 
        self.decoder_fc = nn.Sequential(
                nn.Linear(model_d, 4096), 
                nn.ReLU(inplace=True), 
                nn.Linear(4096, 16 * d * 4 * 4), 
                nn.ReLU()
        )

        self.sigmoid = nn.Sigmoid()
        self.kl_coef = kl_coef
        self.mse = 0
        self.kl = 0

    def encode(self, x):
        h1 = self.encoder(x).view(x.shape[0], -1)
        h1 = self.fc(h1)
        return self.fc_mu(h1), self.fc_std(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h1 = self.decoder_fc(z).view(z.shape[0], 16 * 64, 4, 4)
        return torch.tanh(self.decoder(h1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def sample(self, size):
        sample = torch.randn(size, 20)
        if self.cuda():
            sample = sample.cuda()
        sample = self.decode(sample).cpu()
        return sample

    def loss_function(self, x, recon_x, mu, logvar, z):
        self.mse = F.mse_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return self.mse + self.kl_coef*self.kl

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl}


class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""

    def __init__(self, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.emb_size = k
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, hidden)
        self.fc3 = nn.Linear(hidden, 400)
        self.fc4 = nn.Linear(400, 784)

        self.emb = NearestEmbed(k, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, 784))
        z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size,
                             int(self.hidden / self.emb_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class CVAE(AbstractAutoEncoder):
    def __init__(self, d, kl_coef=0.1, **kwargs):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4,
                               stride=2, padding=1, bias=False),
        )
        self.f = 8
        self.d = d
        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, self.d * self.f ** 2, requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.mse + self.kl_coef * self.kl_loss

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}


class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()

        self.model_type = 'vqvae'

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                d, num_channels, kernel_size=4, stride=2, padding=1),
        )
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)


class VQ_CVAE2(nn.Module):
    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_CVAE2, self).__init__()
