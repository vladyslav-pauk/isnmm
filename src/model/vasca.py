import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import src.modules.network as network
from src.model.modules.autoencoder import AE
from src.model.modules.metric_linear_mixture import LMM
from src.modules.distribution.transformed_normal import TransformedNormal

from src.modules.transform.logit_transform import LogitTransform
from torch.distributions import Dirichlet


class Model(AE, LMM):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.optimizer_config = optimizer_config
        self.mc_samples = model_config["mc_samples"]
        self.sigma = model_config["sigma"]
        self.latent_dim = model_config["latent_dim"]

        self.posterior = TransformedNormal
        self.posterior_transform = LogitTransform()
        self.prior = Dirichlet(torch.ones(self.latent_dim))

    def _reparameterization(self, sample):
        # sample = torch.cat((sample, torch.zeros_like(sample[..., :1])), dim=-1)
        # return F.softmax(sample, dim=-1)
        return self.posterior_transform._inverse(sample)

    def _regularization_loss(self, model_output, data, idxes):
        latent_sample = model_output["latent_sample"]
        variational_parameters = model_output["latent_parameterization_batch"]

        posterior_distribution = self.posterior(*variational_parameters, self.posterior_transform)
        neg_entropy_posterior = posterior_distribution.log_prob(latent_sample).mean()
        log_prior = self.prior.log_prob(latent_sample).mean()
        kl_posterior_prior = neg_entropy_posterior - log_prior

        return {"kl_posterior_prior": self.sigma ** 2 * kl_posterior_prior}

    def configure_optimizers(self):
        lr = self.optimizer_config["lr"]
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr["encoder"]},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr["decoder"]}
        ], **self.optimizer_config["params"])
        return optimizer


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])

        self.mu_network = None
        self.log_var_network = None

    def construct(self, latent_dim, observed_dim):
        self.mu_network = self.constructor(observed_dim, latent_dim - 1, **self.config)
        self.log_var_network = self.constructor(observed_dim, latent_dim - 1, **self.config)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        std = torch.exp(0.5 * log_var)
        return mu, std


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

# fixme: program simplex recovery experiment: vasca / cnae+mves / nisca
