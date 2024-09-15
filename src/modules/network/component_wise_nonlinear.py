import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, output_dim, hidden_layers, activation=None, init_weights=None):
        super(Network, self).__init__()
        self.component_wise_nets = nn.ModuleList([
            self._build_component_wise_net(hidden_layers, activation)
            for _ in range(output_dim)
        ])
        self._initialize_weights(activation, init_weights) if len(hidden_layers) != 0 else None

    def _build_component_wise_net(self, hidden_layers, activation):
        if len(hidden_layers) == 0:
            return nn.Identity()
        layers = []
        input_size = 1
        for hidden_size in hidden_layers.values():
            layers.append(nn.Linear(input_size, hidden_size))
            if activation:
                layers.append(getattr(nn, activation)())  # Adding activation
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self, activation, init_weights):
        for net in self.component_wise_nets:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    if init_weights:
                        getattr(nn.init, init_weights)(layer.weight, nonlinearity=activation.lower())
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        transformed_components = [
            self.component_wise_nets[i](x[..., i:i + 1]) for i in range(x.shape[-1])
        ]
        return torch.cat(transformed_components, dim=-1).abs()
