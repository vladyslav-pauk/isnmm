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
                 **kwargs):
        super().__init__()

        hidden_dims = list(hidden_layers.values())
        self.init_weights = getattr(nn.init, weight_initialization, lambda x: x)

        self.hidden_activation = getattr(F, hidden_activation, lambda x: x)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                self.init_layer(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], h)),
                nn.BatchNorm1d(h, eps=bn_eps, momentum=bn_momentum) if bn_eps else nn.Identity(),
            ) for i, h in enumerate(hidden_dims)
        ])

        self.output_activation = getattr(F, output_activation, lambda x: x)
        self.output_layer = self.init_layer(nn.Linear(hidden_dims[-1], output_dim))

    def init_layer(self, layer):
        try:
            self.init_weights(layer.weight, nonlinearity=self.hidden_activation.__name__)
        except TypeError:
            self.init_weights(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.hidden_activation(layer(x))
        return self.output_activation(self.output_layer(x))
