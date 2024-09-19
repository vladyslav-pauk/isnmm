import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.modules.network import LinearPositive, FCNConstructor
# from src.modules.vae_module import VAEModule
from src.model.vasca import Model as VASCA
from src.model.vasca import Encoder
import src.modules.metric as metric


class Model(VASCA):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, train_config=None):
        super().__init__(ground_truth_model=ground_truth_model, encoder=encoder, decoder=decoder, train_config=train_config)

        self.ground_truth = ground_truth_model

        self.metrics = torchmetrics.MetricCollection({
            'mixture_mse_db': metric.MatrixMse(),
            'mixture_sam': metric.SpectralAngle(),
            'mixture_log_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange(),
            'z_subspace': metric.SubspaceDistance()
            # 'h_r_square': metric.ResidualNonlinearity()
        })
        self.metrics.eval()

    def update_metrics(self, data, model_output, labels):
        self.metrics['mixture_mse_db'].update(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_sam'].update(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_log_volume'].update(
            self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_matrix_change'].update(
            self.decoder.linear_mixture.matrix
        )
        self.metrics['z_subspace'].update(
            labels[0], model_output[1].mean(dim=0)
        )
        # self.metrics['h_r_square'].update(
        #     labels[0],
        #     self.ground_truth.data_model.linear_mixture.matrix,
        #     self.ground_truth.data_model.nonlinear_transform, self.decoder.nonlinear_transform
        # )

# fixme: fix slow training, initialization (loss infinity), mc and batch in fcnconstructor, activation argument (to
#  config?)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, config_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.linear_mixture = LinearPositive(torch.ones(output_dim, latent_dim), **config_decoder)

        # self.nonlinear_transform = ComponentWiseNonlinear(output_dim, **config_decoder)

        self.nonlinear_transform = nn.ModuleList([
            FCNConstructor(
                input_dim=1, output_dim=1, **config_decoder
            ) for _ in range(output_dim)
        ])
        # todo: no need for a class, compose from constructors here

    def forward(self, z):
        x = self.linear_mixture(z)
        # x = self.nonlinear_transform(y)

        x = torch.cat([
            self.nonlinear_transform[i](x[..., i:i + 1].view(-1, 1)).view_as(x[..., i:i + 1])
            for i in range(x.shape[-1])
        ], dim=-1)

        return x
        # todo: check the networks once again, make sure everything is consistent and implemented right, ask gpt to improve

# fixme: clean up and test nisca model, initialize on top of vasca super. just modify the metric and decoder
#  neural network output is horizontal!!! (nearly constant)  something is wrong


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