import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

import src.modules.network as network
from src.model.modules.autoencoder import AE
from src.model.modules.metric_post_nonlinear import PNL


class Model(AE, PNL):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.optimizer_config = optimizer_config
        self.mc_samples = model_config["mc_samples"]
        self.sigma = model_config["sigma"]
        self.latent_dim = model_config["latent_dim"]

    @staticmethod
    def _reparameterization(sample):
        sample = torch.cat((sample[..., :-1], torch.zeros_like(sample[..., :1])), dim=-1)
        sample = F.softmax(sample, dim=-1)
        # sample = sample / sample.sum(dim=-1).unsqueeze(-1)
        return sample

    def _regularization_loss(self, model_output, data, idxes):
        latent_sample = model_output["latent_sample"]
        variational_parameters = model_output["latent_parameterization_batch"]

        neg_entropy_latent = - self._entropy(latent_sample, variational_parameters)
        kl_posterior_prior = neg_entropy_latent - torch.lgamma(torch.tensor(latent_sample.size(-1)))

        return {"kl_posterior_prior": self.sigma ** 2 * kl_posterior_prior}

    # @staticmethod
    # def _entropy(latent_sample, variational_parameters):
    #     mean, std = variational_parameters
    #     mean = mean[..., :-1]
    #     std = std[..., :-1]
    #
    #     epsilon = 1e-12
    #     log_var = 2 * torch.log(std + epsilon)
    #     sigma_diag_inv = 1 / (std + epsilon)
    #
    #     projected_latent = torch.log(latent_sample[..., :-1] / latent_sample[..., -1].unsqueeze(-1)) - mean.unsqueeze(0)
    #     # projected_latent = torch.log(latent_sample / latent_sample.sum(dim=-1).unsqueeze(-1)) - mean
    #     projected_latent.to(latent_sample.device).squeeze(0)
    #     sigma_diag_inv.to(latent_sample.device)
    #
    #     log_2pi = torch.log(torch.tensor(2 * torch.pi)).to(latent_sample.device)
    #     entropy = 0.5 * (latent_sample.size(-1) - 1) * log_2pi
    #     entropy += (projected_latent ** 2 * sigma_diag_inv.unsqueeze(0)).sum(dim=-1).mean() / 2
    #     # entropy += log_var[:, :-1].sum(dim=-1).mean() / 2
    #     entropy += log_var.sum(dim=-1).mean() / 2
    #     entropy += torch.log(latent_sample).sum(dim=-1).mean()
    #
    #     return entropy



    @staticmethod
    def _entropy(latent_sample, variational_parameters):
        mean, std = variational_parameters
        mean = mean[..., :-1]
        std = std[..., :-1]

        epsilon = 1e-12
        log_var = 2 * torch.log(std + epsilon)
        sigma_diag_inv = 1 / (std + epsilon)

        projected_latent = (latent_sample[:, :, :-1] / latent_sample[:, :, -1:]) - mean

        log_2pi = torch.log(torch.tensor(2 * torch.pi)).to(latent_sample.device)
        entropy = 0.5 * (latent_sample.size(-1) - 1) * log_2pi
        entropy += (projected_latent ** 2 * sigma_diag_inv.unsqueeze(0)).sum(dim=-1).mean() / 2
        entropy += log_var[:, :-1].sum(dim=-1).mean() / 2
        entropy += torch.log(latent_sample).sum(dim=-1).mean()

        return entropy

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        lr = self.optimizer_config["lr"]

        optimizer = optimizer_class([
            {'params': self.encoder.mu_network.parameters(), 'lr': lr["encoder"]},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr["decoder"]["linear"]},
            {'params': self.decoder.nonlinear_transform.parameters(), 'lr': lr["decoder"]["nonlinear"]}
        ], **self.optimizer_config["params"])
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **lr["scheduler"]
            }
        }


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.mu_network = None
        self.log_var_network = None

    def construct(self, latent_dim, observed_dim):
        self.mu_network = self.constructor(observed_dim, latent_dim, **self.config)
        self.log_var_network = self.constructor(observed_dim, latent_dim, **self.config)
        self.log_var_network.eval()

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        std = torch.exp(0.5 * log_var)
        std = torch.zeros_like(mu).to(mu.device)
        return mu, std


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        # if latent_dim != observed_dim:
        #     self.linear_mixture = network.LinearPositive(
        #         torch.eye(observed_dim, latent_dim), **self.config
        #     )
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)
        # self.nonlinear_transform = self.constructor(latent_dim, observed_dim, **self.config)

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
