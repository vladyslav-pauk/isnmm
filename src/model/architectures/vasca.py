import torch
import torch.nn as nn

import src.modules.network as network
from src.model.modules.lightning import Module as LightningModule
from src.model.modules.vae import Module as VariationalAutoencoder


class Model(LightningModule, VariationalAutoencoder):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.latent_dim = model_config["latent_dim"]

        self.optimizer_config = optimizer_config
        self.prior_config = model_config["prior"]
        self.posterior_config = model_config["posterior"]
        self.encoder_transform = model_config["reparameterization"]

        self.mc_samples = model_config["mc_samples"]
        self.sigma = model_config["sigma"]

        self.distance = model_config["distance"]
        self.experiment_metrics = model_config["experiment_name"]


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])

        self.loc_network = None
        self.scale_network = None

        # self.save_hyperparameters({
        #     'config': config
        # })

    def construct(self, latent_dim, observed_dim):
        self.loc_network = self.constructor(observed_dim, latent_dim - 1, **self.config)
        self.scale_network = self.constructor(observed_dim, latent_dim - 1, **self.config)

    def forward(self, x):
        loc = self.loc_network(x)
        scale = self.scale_network(x)
        return loc, scale.exp() # clamp(min=1e-12)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.linear_mixture = None

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = network.LinearPositive(torch.rand(observed_dim, latent_dim), **self.config)
        return self

    def forward(self, z):
        x = self.linear_mixture(z)
        return x
