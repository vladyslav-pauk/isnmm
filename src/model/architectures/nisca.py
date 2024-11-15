import torch
import torch.nn as nn

import src.modules.network as network
from src.model.modules.lightning import Module as LightningModule
from src.model.modules.vae import Module as VariationalAutoencoder


class Model(LightningModule, VariationalAutoencoder):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None, metrics=None):
        super().__init__(encoder, decoder)

        self.encoder = encoder
        self.decoder = decoder

        self.metrics = metrics

        self.latent_dim = model_config["latent_dim"]

        self.optimizer_config = optimizer_config
        self.prior_config = model_config["prior"]
        self.posterior_config = model_config["posterior"]
        self.encoder_transform = model_config["reparameterization"]

        self.mc_samples = model_config["mc_samples"]
        self.sigma = model_config["sigma"]

        self.distance = model_config["distance"]
        self.unmixing = model_config["unmixing"]


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])

        self.loc_network = None
        self.scale_network = None
        self.linear_mixture_inv_loc = nn.Identity()
        self.linear_mixture_inv_scale = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        self.loc_network = self.constructor(observed_dim, latent_dim - 1, **self.config)
        self.scale_network = self.constructor(observed_dim, latent_dim - 1, **self.config)

        self.linear_mixture_inv_loc = network.LinearPositive(
            torch.rand(latent_dim - 1, observed_dim), **self.config
        )
        self.linear_mixture_inv_scale = network.LinearPositive(
            torch.rand(latent_dim - 1, observed_dim), **self.config
        )
        # task: change order of observed, latent arguments in constructor
        # task: make it a part of CNN? final linear layer

    def forward(self, x):
        loc = self.loc_network(x)
        scale = self.scale_network(x)

        # loc = self.linear_mixture_inv_loc(loc)
        # scale = self.linear_mixture_inv_scale(scale)

        return loc, scale.exp().clamp(1e-12, 1e12)


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

    def forward(self, x):
        x = self.linear_mixture(x)
        x = self.nonlinear_transform(x)
        return x

# todo: CNN and CFCN automatically with linear mixture, FCN without
# task: SVMAX initialization
# task: clean and squash github commits
# task: use yaml for config
# task: move to experiment, make a folder for each experiment (move configs too?)
# task: check numerical stability of the model, if there are nans or unusual values
# todo: look into cnae and vasca papers and see what else has to be implemented
# todo: instead of h1, h2, ... use depth and width, make both, if not depth, width read h1, h2...
