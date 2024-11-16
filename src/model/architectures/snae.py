import torch
from torch import nn

import src.modules.network as network
from src.model.modules.lightning import Module as LightningModule
from src.model.modules.ae import Module as Autoencoder


class Model(LightningModule, Autoencoder):
    def __init__(self, encoder, decoder, model_config, optimizer_config, metrics=None):
        super().__init__(encoder, decoder)

        self.optimizer_config = optimizer_config
        self.metrics = metrics

        self.latent_dim = model_config["latent_dim"]
        self.mc_samples = 1
        self.sigma = 0

        self.distance = model_config["distance"]
        self.encoder_transform = model_config["reparameterization"]


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.nonlinear_transform = None
        self.linear_mixture_inv = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)
        self.linear_mixture_inv = network.LinearPositive(
            torch.rand(latent_dim, observed_dim), **self.config
        )

    def forward(self, x):
        y = self.nonlinear_transform.forward(x)
        z = self.linear_mixture_inv(y)
        return z


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = network.LinearPositive(
            torch.rand(observed_dim, latent_dim), **self.config
        )
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

    def forward(self, z):
        y = self.linear_mixture(z)
        x = self.nonlinear_transform(y)
        return x
