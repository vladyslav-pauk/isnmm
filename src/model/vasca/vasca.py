import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim

import src.modules.network as network
from src.model.ae_module import AutoEncoderModule
import src.modules.metric as metric

# fixme: organize models


class Model(AutoEncoderModule):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.optimizer_config = optimizer_config
        self.mc_samples = model_config["mc_samples"]
        self.sigma = model_config["sigma"]
        self.latent_dim = model_config["latent_dim"]
        # fixme: if ground_truth sigma and latent_dim are not provided in the dataset, fix this case
        self.linear_mixture_true = None
        self.metrics = None

    @staticmethod
    def reparameterization(z):
        z = torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1)
        return F.softmax(z, dim=-1)

    def loss_function(self, data, model_output, idxes):
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

    def setup_metrics(self):
        if self.ground_truth:
            self.latent_dim = self.ground_truth.latent_dim
            self.sigma = self.ground_truth.sigma
            self.linear_mixture_true = self.ground_truth.linear_mixture

        self.metrics = torchmetrics.MetricCollection({
            'mixture_mse_db': metric.MatrixMse(),
            'mixture_sam': metric.SpectralAngle(),
            'mixture_log_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange(),
            'subspace_distance': metric.SubspaceDistance(),
            'latent_mse': metric.MatrixMse()
        })
        self.metrics.eval()
        self.log_monitor = {
            "monitor": "mixture_mse_db",
            "mode": "min"
        }

    def update_metrics(self, data, model_output, labels, idxes):
        latent_sample = model_output["latent_sample"].mean(0)
        latent_sample_true = labels["latent_sample"]
        latent_sample_qr = labels["latent_sample_qr"]
        linear_mixture_true = self.linear_mixture_true
        linear_mixture = self.decoder.linear_mixture.matrix

        self.metrics['mixture_mse_db'].update(linear_mixture_true, linear_mixture)
        self.metrics['mixture_sam'].update(linear_mixture_true, linear_mixture)
        self.metrics['mixture_log_volume'].update(linear_mixture)
        self.metrics['mixture_matrix_change'].update(linear_mixture)
        self.metrics['subspace_distance'].update(idxes, latent_sample, latent_sample_qr)
        self.metrics['latent_mse'].update(latent_sample, latent_sample_true)

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