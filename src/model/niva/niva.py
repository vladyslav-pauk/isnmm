import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim

import src.modules.network as network
from src.modules.ae_module import AutoEncoderModule
import src.modules.metric as metric


class Model(AutoEncoderModule):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.ground_truth = ground_truth_model
        self.optimizer_config = optimizer_config
        self.mc_samples = model_config["mc_samples"]

        self.sigma = model_config["sigma"]

        self.metrics = None
        self.log_monitor = None
        self.setup_metrics()

    @staticmethod
    def reparameterization(z):
        z = torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1)
        return F.softmax(z, dim=-1)


    def loss_function(self, data, model_output, idxes):
        if not self.sigma:
            self.sigma = self.ground_truth.sigma

        reconstructed_sample = model_output["reconstructed_sample"]
        latent_sample = model_output["latent_sample"]
        variational_parameters = model_output["latent_parameterization_batch"]

        loss = {"reconstruction": self._reconstruction(data, reconstructed_sample)}

        neg_entropy_latent = - self._entropy(latent_sample, variational_parameters)
        kl_posterior_prior = neg_entropy_latent - torch.lgamma(torch.tensor(latent_sample.size(-1)))
        loss.update({"kl_posterior_prior": self.sigma ** 2 * kl_posterior_prior})

        return loss

    @staticmethod
    def _reconstruction(data, reconstructed_sample):
        recon_loss = data.size(-1) / 2 * F.mse_loss(
            reconstructed_sample, data.expand_as(reconstructed_sample), reduction='mean'
        )
        # recon_loss += self.sigma ** 2 * data.size(-1) / 2 * torch.log(torch.tensor(2 * torch.pi))
        return recon_loss

    @staticmethod
    def _entropy(latent_sample, variational_parameters):
        mean, std = variational_parameters

        log_var = 2 * torch.log(std + 1e-12)
        sigma_diag_inv = 1 / (std + 1e-12)

        projected_latent = torch.log(latent_sample[:, :, :-1] / latent_sample[:, :, -1:]) - mean

        log_2pi = torch.log(torch.tensor(2 * torch.pi))
        entropy = 0.5 * (latent_sample.size(-1) - 1) * log_2pi
        entropy += (projected_latent ** 2 * sigma_diag_inv.unsqueeze(0)).sum(dim=-1).mean() / 2
        entropy += log_var[:, :-1].sum(dim=-1).mean() / 2
        entropy += torch.log(latent_sample).sum(dim=-1).mean()

        return entropy

    def setup_metrics(self):
        self.metrics = torchmetrics.MetricCollection({
            'subspace_distance': metric.SubspaceDistance(),
            'h_r_square': metric.ResidualNonlinearity()
        })
        self.metrics.eval()
        self.log_monitor = {
            "monitor": "mixture_mse_db",
            "mode": "min"
        }

    def update_metrics(self, data, model_output, labels, idxes):
        reconstructed_sample = model_output["reconstructed_sample"].mean(0)
        latent_sample = model_output["latent_sample"]
        linearly_mixed_sample = labels["linearly_mixed_sample"]
        latent_sample_qr = labels["latent_sample_qr"]

        self.metrics['subspace_distance'].update(
            idxes, latent_sample.mean(0), latent_sample_qr
        )
        self.metrics['h_r_square'].update(
            data, reconstructed_sample, linearly_mixed_sample
        )

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
