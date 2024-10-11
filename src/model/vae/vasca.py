import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim
import wandb

import src.modules.network as network
from src.model.ae_module import AE
from src.model.lmm_module import LMM
import src.modules.metric as metric

# fixme: organize models
# fixme: program simplex recovery experiment: vasca / cnae+mves / nisca


class Model(AE, LMM):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.optimizer_config = optimizer_config
        self.mc_samples = model_config["mc_samples"]
        self.sigma = model_config["sigma"]
        self.latent_dim = model_config["latent_dim"]

    @staticmethod
    def _reparameterization(z):
        z = torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1)
        return F.softmax(z, dim=-1)

    def _regularization_loss(self, model_output, data, idxes):
        latent_sample = model_output["latent_sample"]
        variational_parameters = model_output["latent_parameterization_batch"]

        neg_entropy_latent = - self._entropy(latent_sample, variational_parameters)
        kl_posterior_prior = neg_entropy_latent - torch.lgamma(torch.tensor(latent_sample.size(-1)))
        return {"kl_posterior_prior": self.sigma ** 2 * kl_posterior_prior}

    @staticmethod
    def _entropy(latent_sample, variational_parameters):
        mean, std = variational_parameters

        epsilon = 1e-12
        log_var = 2 * torch.log(std + epsilon)
        sigma_diag_inv = 1 / (std + epsilon)

        projected_latent = torch.log(latent_sample[:, :, :-1] / latent_sample[:, :, -1:]) - mean

        log_2pi = torch.log(torch.tensor(2 * torch.pi))
        entropy = 0.5 * (latent_sample.size(-1) - 1) * log_2pi
        entropy += (projected_latent ** 2 * sigma_diag_inv.unsqueeze(0)).sum(dim=-1).mean() / 2
        entropy += log_var[:, :-1].sum(dim=-1).mean() / 2
        entropy += torch.log(latent_sample).sum(dim=-1).mean()

        return entropy

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

# todo: separate universal scripts run_sweep, train, analyze_sweep, generate_data from experiments (each experiment folder has a script)