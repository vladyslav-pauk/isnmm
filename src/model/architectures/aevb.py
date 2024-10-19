import torch
import torch.nn as nn

import src.modules.network as network
from src.model.modules.lightning import Module as LightningModule
from src.model.modules.vae import Module as VariationalAutoencoder


class Model(LightningModule, VariationalAutoencoder):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None, **kwargs):
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

    def _regularization_loss(self, model_output, data, idxes):
        latent_sample = model_output["latent_sample"]

        # prior, posterior = self._model(*model_output["posterior_parameterization"])
        # neg_entropy_posterior = posterior.log_prob(latent_sample).mean()
        # log_prior = prior.log_prob(latent_sample).mean()

        import torch
        mu, std = model_output["posterior_parameterization"]
        neg_entropy_posterior = -0.5 * torch.mean(1 + 2 * torch.log(std + 1e-12) + torch.log(torch.tensor(2 * torch.pi))) * latent_sample.size(-1)
        log_prior = -0.5 * torch.mean(mu.pow(2) + std.pow(2) + torch.log(torch.tensor(2 * torch.pi))) * latent_sample.size(-1)

        kl_posterior_prior = neg_entropy_posterior - log_prior
        return {"kl_posterior_prior": 2 * self.sigma**2 * kl_posterior_prior}


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, latent_dim, observed_dim):
        self.mu_network = network.FCN(observed_dim, latent_dim, **self.config)
        self.log_var_network = network.FCN(observed_dim, latent_dim, **self.config)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        std = torch.exp(0.5 * log_var)
        return mu, std


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, latent_dim, observed_dim):
        self.network = network.FCN(latent_dim, observed_dim, **self.config)

    def forward(self, z):
        return self.network.forward(z)
