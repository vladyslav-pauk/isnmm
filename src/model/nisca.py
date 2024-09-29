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

        self.metrics = torchmetrics.MetricCollection({
            'mixture_log_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange(),
            'z_subspace': metric.SubspaceDistance(),
            'h_r_square': metric.ResidualNonlinearity()
        })
        self.metrics.eval()

    def configure_optimizers(self):
        lr = self.optimizer_config["lr"]
        lr_th_nl = lr["th"]["nl"]
        lr_th_l = lr["th"]["l"]
        lr_ph = lr["ph"]
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr_ph},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr_th_l},
            {'params': self.decoder.nonlinearity.parameters(), 'lr': lr_th_nl}
        ], **self.optimizer_config["params"])
        return optimizer

    def update_metrics(self, data, model_output, labels):
        self.metrics['mixture_log_volume'].update(
            self.decoder.linear_mixture.matrix
        )

        self.metrics['mixture_matrix_change'].update(
            self.decoder.linear_mixture.matrix
        )

        self.metrics['z_subspace'].update(
            labels[0], model_output[1].mean(dim=0)
        )

        self.metrics['h_r_square'].update(
            self.decoder.nonlinear_transform,
            self.ground_truth.linearly_mixed_sample,
            self.ground_truth.noiseless_sample
        )


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = network.LinearPositive(
            torch.rand(observed_dim, latent_dim), **self.config
        )

        self.nonlinearity = nn.ModuleList([self.constructor(
            input_dim=1, output_dim=1, **self.config
        ) for _ in range(observed_dim)])

    def nonlinear_transform(self, x):
        x = torch.cat([
            self.nonlinearity[i](x[..., i:i + 1].view(-1, 1)).view_as(x[..., i:i + 1])
            for i in range(x.shape[-1])
        ], dim=-1)
        return x

    def forward(self, z):
        x = self.linear_mixture(z)
        x = self.nonlinear_transform(x)
        return x

# fixme: fix decoder: sometimes get error, not at initialization (loss infinity, nans)
# fixme: h_metric slow!!!
# fixme: test to get good results, neural network output is horizontal!!! (nearly constant)  something is wrong

# todo: mc and batch in fcnconstructor, activation argument (to config?)
# todo: check the networks once again, make sure everything is consistent and implemented right, ask gpt to improve
# todo: clean up and test nisca model.

# class Network(nn.Module):
#     def __init__(self, output_dim, hidden_layers, activation=None, output_activation=None, weight_initialization=None, **kwargs):
#         super(Network, self).__init__()
#         self.component_wise_nets = nn.ModuleList([
#             self._build_component_wise_net(hidden_layers, activation)
#             for _ in range(output_dim)
#         ])
#         self._initialize_weights(activation, weight_initialization) if len(hidden_layers) != 0 else None
#
#     def _build_component_wise_net(self, hidden_layers, activation):
#         if len(hidden_layers) == 0:
#             return nn.Identity()
#         layers = []
#         input_size = 1
#         for hidden_size in hidden_layers.values():
#             layers.append(nn.Linear(input_size, hidden_size))
#             if activation:
#                 layers.append(getattr(nn, activation)())  # Adding activation
#             input_size = hidden_size
#
#         layers.append(nn.Linear(input_size, 1))
#         return nn.Sequential(*layers)
#
#     def _initialize_weights(self, activation, init_weights):
#         for net in self.component_wise_nets:
#             for layer in net:
#                 if isinstance(layer, nn.Linear):
#                     if init_weights:
#                         getattr(nn.init, init_weights)(layer.weight, nonlinearity=activation.lower())
#                     if layer.bias is not None:
#                         nn.init.zeros_(layer.bias)
#
#     def forward(self, x):
#         transformed_components = [
#             self.component_wise_nets[i](x[..., i:i + 1]) for i in range(x.shape[-1])
#         ]
#         return torch.cat(transformed_components, dim=-1).abs()