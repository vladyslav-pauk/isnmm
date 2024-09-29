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
                 **kwargs):
        super().__init__()

        hidden_dims = list(hidden_layers.values()) if hidden_layers else []

        self.init_weights = getattr(nn.init, weight_initialization) if weight_initialization else lambda x: x

        self.hidden_activation = getattr(F, hidden_activation) if hidden_activation else nn.Identity()
        self.output_activation = getattr(F, output_activation) if output_activation else nn.Identity()

        self.layers = self.build_fcn_network(input_dim, output_dim, hidden_dims, bn_eps, bn_momentum, dropout_rate)

    def build_fcn_network(self, input_dim, output_dim, hidden_dims, bn_eps, bn_momentum, dropout_rate):
        hidden_layers = nn.ModuleList([
            self._layer_stack(i, h, input_dim, hidden_dims, bn_eps, bn_momentum, dropout_rate)
            for i, h in enumerate(hidden_dims)
        ])
        output_layer = self.init_layer(nn.Linear(hidden_dims[-1], output_dim))
        return nn.Sequential(*hidden_layers, output_layer)

    def _layer_stack(self, i, h, input_dim, hidden_dims, bn_eps, bn_momentum, dropout_rate):
        layer_stack = nn.Sequential(
                self.init_layer(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], h)),
                nn.BatchNorm1d(h, eps=bn_eps, momentum=bn_momentum) if bn_eps else nn.Identity(),
                nn.Dropout(dropout_rate if dropout_rate else 0) if dropout_rate else nn.Identity()
            )
        return layer_stack

    def init_layer(self, layer):
        try:
            self.init_weights(layer.weight, nonlinearity=self.hidden_activation.__name__)
        except TypeError:
            self.init_weights(layer.weight) if self.init_weights else lambda x: x
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        return self.output_activation(self.layers[-1](x))

# fixme: batchnorm in evaluation mode
