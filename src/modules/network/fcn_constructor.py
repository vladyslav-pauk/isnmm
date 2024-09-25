import torch.nn as nn
from torch.nn import functional as F

import torch
import torch.nn as nn
from torch.nn import functional as F


class FCNConstructor(nn.Module):
    def __init__(self,
                 input_dim, output_dim,
                 hidden_layers=None,
                 hidden_activation=None,
                 output_activation=None,
                 weight_initialization=None,
                 bn_eps=None,
                 bn_momentum=None,
                 dropout_rate=None,
                 use_convolution=False,
                 conv_layer_dims=[150, 140, 130],
                 conv_groups=1,
                 **kwargs):
        super().__init__()

        hidden_dims = list(hidden_layers.values()) if hidden_layers else []
        conv_dims = conv_layer_dims if conv_layer_dims else []

        # Weight initialization strategy
        self.init_weights = getattr(nn.init, weight_initialization, nn.init.xavier_uniform_)

        # Handle both string-based and callable activations
        self.hidden_activation = hidden_activation if callable(hidden_activation) else getattr(F, hidden_activation,
                                                                                               F.relu)
        self.output_activation = output_activation if callable(output_activation) else getattr(F, output_activation,
                                                                                               None)

        # Select the network type based on use_convolution
        if use_convolution:
            self.layers = self.build_conv_network(input_dim, output_dim, conv_dims, conv_groups, bn_eps, bn_momentum,
                                                  dropout_rate)
        else:
            self.layers = self.build_fcn_network(input_dim, output_dim, hidden_dims, bn_eps, bn_momentum, dropout_rate)

    def build_conv_network(self, input_dim, output_dim, conv_dims, conv_groups, bn_eps, bn_momentum, dropout_rate):
        layers = []

        # Input layer for Conv network
        layers.append(nn.Conv1d(input_dim, conv_dims[0] * input_dim, kernel_size=1, groups=conv_groups))
        layers.append(nn.SiLU())

        # Hidden layers for Conv network
        for i in range(1, len(conv_dims)):
            layers.append(
                nn.Conv1d(conv_dims[i - 1] * input_dim, conv_dims[i] * input_dim, kernel_size=1, groups=conv_groups))
            layers.append(nn.ReLU())

        # Output layer for Conv network
        layers.append(nn.Conv1d(conv_dims[-1] * input_dim, output_dim, kernel_size=1, groups=conv_groups))

        return nn.Sequential(*layers)

    def build_fcn_network(self, input_dim, output_dim, hidden_dims, bn_eps, bn_momentum, dropout_rate):
        # Initialize hidden layers with optional batch normalization and dropout for fully connected network
        hidden_layers = nn.ModuleList([
            nn.Sequential(
                self.init_layer(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], h)),
                nn.BatchNorm1d(h, eps=bn_eps, momentum=bn_momentum) if bn_eps else nn.Identity(),
                nn.Dropout(dropout_rate if dropout_rate else 0) if dropout_rate else nn.Identity()
            ) for i, h in enumerate(hidden_dims)
        ])

        # Initialize the output layer
        output_layer = self.init_layer(nn.Linear(hidden_dims[-1], output_dim))

        return nn.Sequential(*hidden_layers, output_layer)

    def init_layer(self, layer):
        try:
            self.init_weights(layer.weight, nonlinearity=self.hidden_activation.__name__)
        except TypeError:
            self.init_weights(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        if isinstance(self.layers[0], nn.Conv1d):
            x = x.unsqueeze(-1)  # Prepare input for Conv1d layers (add channel dimension)

        for layer in self.layers:
            x = self.hidden_activation(layer(x)) if isinstance(layer, nn.Sequential) else layer(x)

        if isinstance(self.layers[0], nn.Conv1d):
            x = x.squeeze(-1)  # Remove channel dimension after Conv1d layers

        return x if not self.output_activation else self.output_activation(x)

# class FCNConstructor(nn.Module):
#     def __init__(self,
#                  input_dim, output_dim,
#                  hidden_layers=None,
#                  hidden_activation=None,
#                  output_activation=None,
#                  weight_initialization=None,
#                  bn_eps=None,
#                  bn_momentum=None,
#                  **kwargs):
#         super().__init__()
#
#         hidden_dims = list(hidden_layers.values())
#         self.init_weights = getattr(nn.init, weight_initialization, lambda x: x)
#
#         # self.dropout_rate = kwargs.get('dropout_rate', None)
#         # if self.dropout_rate:
#         #     layers.append(nn.Dropout(self.dropout_rate))
#
#         self.hidden_activation = getattr(F, hidden_activation, lambda x: x)
#         self.hidden_layers = nn.ModuleList([
#             nn.Sequential(
#                 self.init_layer(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], h)),
#                 nn.BatchNorm1d(h, eps=bn_eps, momentum=bn_momentum) if bn_eps else nn.Identity(),
#             ) for i, h in enumerate(hidden_dims)
#         ])
#
#         self.output_activation = getattr(F, output_activation, lambda x: x)
#         self.output_layer = self.init_layer(nn.Linear(hidden_dims[-1], output_dim))
#
#     def init_layer(self, layer):
#         try:
#             self.init_weights(layer.weight, nonlinearity=self.hidden_activation.__name__)
#         except TypeError:
#             self.init_weights(layer.weight)
#         nn.init.zeros_(layer.bias)
#         return layer
#
#     def forward(self, x):
#         for layer in self.hidden_layers:
#             x = self.hidden_activation(layer(x))
#         return self.output_activation(self.output_layer(x))
# fixme: batchnorm in evaluation mode
# fixme: on the residual plot y-axis is multiplied by 100, why?
