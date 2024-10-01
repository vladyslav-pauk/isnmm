import xxsubtype

import torch
import torch.nn as nn
import torchmetrics
import torch.optim as optim

import src.modules.network as network
from src.model.vasca import Model as VASCA
from src.model.vasca import Encoder
import src.modules.metric as metric


class Model(VASCA):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(ground_truth_model, encoder, decoder, model_config, optimizer_config)

        self.ground_truth = ground_truth_model

        self.setup_metrics()

    def configure_optimizers(self):
        lr = self.optimizer_config["lr"]
        lr_th_nl = lr["th"]["nl"]
        lr_th_l = lr["th"]["l"]
        lr_ph = lr["ph"]
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr_ph},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr_th_l},
            {'params': self.decoder.nonlinear_transform.parameters(), 'lr': lr_th_nl}
        ], **self.optimizer_config["params"])
        return optimizer

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
        reconstructed_sample = model_output["reconstructed_sample"]
        latent_sample = model_output["latent_sample"]
        true_latent_sample = labels["latent_sample"]
        linearly_mixed_sample = labels["linearly_mixed_sample"]

        self.metrics['subspace_distance'].update(
            idxes, reconstructed_sample.mean(0).squeeze(), true_latent_sample
        )

        self.metrics['h_r_square'].update(
            # lambda x: self.decoder(x.unsqueeze(1)).squeeze(1),
            self.decoder.nonlinear_transform,
            self.ground_truth.linearly_mixed_sample,
            self.ground_truth.noiseless_sample
        )


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = None
        self.nonlinear_transform = None

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = self.constructor(latent_dim, observed_dim, **self.config)

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
