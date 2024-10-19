import torch
import torch.nn as nn

import src.modules.network as network
from src.model.modules.lightning import Module as LightningModule
from src.model.modules.vae import Module as VariationalAutoencoder


# todo: use only simplex_recovery run config with sweep
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
        # self.linear_mixture_inv = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        self.loc_network = self.constructor(observed_dim, latent_dim - 1, **self.config)
        self.scale_network = self.constructor(observed_dim, latent_dim - 1, **self.config)

        # self.linear_mixture_inv_mu = network.LinearPositive(
        #     torch.rand(latent_dim - 1, observed_dim), **self.config
        # )
        # self.linear_mixture_inv_var = network.LinearPositive(
        #     torch.rand(latent_dim - 1, observed_dim), **self.config
        # )
        # todo: change order of observed, latent arguments in constructor

    def forward(self, x):
        loc = self.loc_network(x)
        # mu = self.linear_mixture_inv_mu(mu)
        scale = self.scale_network(x)
        # std = std.abs().pow(0.5)
        # # std = torch.exp(0.5 * std)
        # std = self.linear_mixture_inv_var(std)
        # std = torch.zeros_like(mu)
        return loc, scale


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

# todo: sometimes during training get nan, right now skipped
# todo: mc and batch in fcnconstructor, activation argument (to config?)
# todo: check the networks once again, make sure everything is consistent and implemented right, ask gpt to improve
# todo: clean up and test nisca model.
# todo: SVMAX initialization
# todo: experiment parent class setup and update metric, common for all models in an experiment
# todo: proper cnn with linear layer and proper postnonlinearity (make a separate class PNLConstructor for FCN or CNN)
