import torch.nn as nn
from torch.nn import functional as F


class FCNConstructor(nn.Module):
    def __init__(self,
                 input_dim, output_dim,
                 hidden_layers=None, activation=None, output_activation=None, weight_initialization=None,
                 **kwargs):
        super().__init__()

        self.activation = activation
        self.output_activation = getattr(F, output_activation, lambda x: x)
        self.init_weights = getattr(nn.init, weight_initialization, lambda x: x)

        hidden_dims = list(hidden_layers.values())
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                self.init_layer(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], h)),
                nn.BatchNorm1d(h)
            ) for i, h in enumerate(hidden_dims)
        ])
        self.output_layer = self.init_layer(nn.Linear(hidden_dims[-1], output_dim))

    def init_layer(self, layer):
        try:
            self.init_weights(layer.weight, nonlinearity=self.activation)
        except TypeError:
            self.init_weights(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        for layer in self.hidden_layers:
            activation = getattr(F, self.activation, lambda x: x)
            x = activation(layer(x))
        return self.output_activation(self.output_layer(x))

# todo: use in-place activation functions where possible, e.g., F.relu_(h)? or use nn.ReLU(inplace=True) layers
