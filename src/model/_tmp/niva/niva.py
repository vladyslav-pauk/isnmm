import torch.nn as nn
import torchmetrics
import torch.optim as optim
import torch
import torch.nn.functional as F

import src.modules.network as network
from src.model.vasca.vasca import Model as VASCA
from src.model.vasca.vasca import Encoder
import src.modules.metric as metric


class Model(VASCA):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(ground_truth_model, encoder, decoder, model_config, optimizer_config)

        self.ground_truth = ground_truth_model
        self.metrics = None
        self.log_monitor = None
        self.optimizer = None

        self.setup_metrics()


    @staticmethod
    def reparameterization(z):
        z = torch.cat((z, torch.zeros_like(z[..., :1])), dim=-1)
        return F.softmax(z, dim=-1)


    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        lr = self.optimizer_config["lr"]
        self.optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr["encoder"]},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr["decoder"]["linear"]},
            {'params': self.decoder.nonlinear_transform.parameters(), 'lr': lr["decoder"]["nonlinear"]}
        ], **self.optimizer_config["params"])
        return self.optimizer

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
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = None
        self.nonlinear_transform = None

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = network.LinearPositive(
            torch.eye(observed_dim, latent_dim), **self.config
        )
        # self.linear_mixture.eval()

        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

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
