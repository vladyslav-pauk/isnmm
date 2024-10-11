import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

import src.modules.network as network
from src.model.modules.autoencoder import AE
from src.model.modules.post_nonlinear import PNL


class Model(AE, PNL):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.optimizer_config = optimizer_config
        self.mc_samples = model_config["mc_samples"]
        # self.sigma = model_config["sigma"]
        # self.latent_dim = model_config["latent_dim"]

    @staticmethod
    def _reparameterization(z):
        # z = torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1)
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
        # mean = mean[..., :-1]
        # std = std[..., :-1]

        epsilon = 1e-12
        log_var = 2 * torch.log(std + epsilon)
        sigma_diag_inv = 1 / (std + epsilon)

        # projected_latent = torch.log(latent_sample[..., :-1] / latent_sample[..., -1]) - mean
        projected_latent = torch.log(latent_sample / latent_sample.sum(dim=-1).unsqueeze(-1)) - mean

        log_2pi = torch.log(torch.tensor(2 * torch.pi))
        entropy = 0.5 * (latent_sample.size(-1) - 1) * log_2pi
        entropy += (projected_latent ** 2 * sigma_diag_inv.unsqueeze(0)).sum(dim=-1).mean() / 2
        # entropy += log_var[:, :-1].sum(dim=-1).mean() / 2
        entropy += log_var.sum(dim=-1).mean() / 2
        entropy += torch.log(latent_sample).sum(dim=-1).mean()

        return entropy * 0

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        lr = self.optimizer_config["lr"]

        optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr["encoder"]},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr["decoder"]["linear"]},
            {'params': self.decoder.nonlinear_transform.parameters(), 'lr': lr["decoder"]["nonlinear"]}
        ], **self.optimizer_config["params"])
        scheduler = ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                **lr["scheduler"]
            }
        }

    # def _setup_metrics(self, ground_truth=None):
    #     metrics = {}
    #     if ground_truth:
    #         self.latent_dim = ground_truth.latent_dim
    #         self.sigma = ground_truth.sigma
    #         metrics.update({
    #             'subspace_distance': metric.SubspaceDistance(),
    #             'r_square': metric.ResidualNonlinearity()
    #         })
    #
    #     self.metrics = torchmetrics.MetricCollection(metrics)
    #     self.metrics.eval()
    #
    #     wandb.define_metric(name="r_square", summary='max')
    #
    # def _update_metrics(self, observed_sample, model_output, labels, idxes):
    #     if labels:
    #         latent_sample = model_output["latent_sample"].mean(0)
    #         latent_sample_qr = labels["latent_sample_qr"]
    #         linearly_mixed_sample = self.decoder.linear_mixture(latent_sample)
    #
    #         self.metrics['subspace_distance'].update(
    #             idxes, latent_sample, latent_sample_qr
    #         )
    #         self.metrics['r_square'].update(
    #             model_output, labels, linearly_mixed_sample, observed_sample
    #         )


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

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        std = torch.exp(0.5 * log_var)
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

    def forward(self, x):
        x = self.linear_mixture(x)
        x = self.nonlinear_transform(x)
        return x

# class Decoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.constructor = getattr(network, config["constructor"])
#
#     def construct(self, latent_dim, observed_dim):
#         self.linear_mixture = network.LinearPositive(
#             torch.rand(observed_dim, latent_dim), **self.config
#         )
#
#         self.nonlinearity = nn.ModuleList([self.constructor(
#             input_dim=1, output_dim=1, **self.config
#         ) for _ in range(observed_dim)])
#
#     def nonlinear_transform(self, x):
#         x = torch.cat([
#             self.nonlinearity[i](x[..., i:i + 1].view(-1, 1)).view_as(x[..., i:i + 1])
#             for i in range(x.shape[-1])
#         ], dim=-1)
#         return x
#
#     def forward(self, z):
#         y = self.linear_mixture(z)
#         x = self.nonlinear_transform(y)
#         return x


# todo: sometimes during training get nan, right now skipped
# todo: mc and batch in fcnconstructor, activation argument (to config?)
# todo: check the networks once again, make sure everything is consistent and implemented right, ask gpt to improve
# todo: clean up and test nisca model.

# todo: SVMAX initialization
# todo: experiment parent class setup and update metric, common for all models in an experiment
