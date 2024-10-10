import torch
import torch.nn as nn


class CFCNConstructor(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers=None,
                 hidden_activation=None,
                 output_activation=None,
                 weight_initialization=None,
                 bn_eps=None,
                 bn_momentum=None,
                 dropout_rate=None,
                 **kwargs):
        super().__init__()

        self.input_dim = input_dim

        self.hidden_activation = getattr(nn, hidden_activation) if hidden_activation else nn.Identity
        self.output_activation = getattr(nn, output_activation) if output_activation else nn.Identity
        self.init_weights = getattr(nn.init, weight_initialization) if weight_initialization else lambda x: x

        self.fcn_per_component = nn.ModuleList([
            self.build_fcn_network(1, 1, hidden_layers, bn_eps, bn_momentum, dropout_rate)
            for _ in range(input_dim)
        ])

    def build_fcn_network(self, input_dim, output_dim, hidden_layers, bn_eps, bn_momentum, dropout_rate):
        hidden_dims = list(hidden_layers.values()) if hidden_layers else []

        layers = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Sequential(
                self.init_weights(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dim)),
                nn.BatchNorm1d(hidden_dim, eps=bn_eps, momentum=bn_momentum) if bn_eps else nn.Identity(),
                self.hidden_activation(),
                nn.Dropout(dropout_rate if dropout_rate else 0) if dropout_rate else nn.Identity()
            ))

        output_layer = nn.Sequential(
            self.init_weights(nn.Linear(hidden_dims[-1], output_dim)),
            self.output_activation()
        )
        layers.append(output_layer)

        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = []
        for i in range(self.input_dim):
            x_i = x[..., i:i+1]
            f_i = self.fcn_per_component[i](x_i)
            outputs.append(f_i)

        return torch.cat(outputs, dim=-1)
