import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = latent_dim

        hidden_dims = list(hidden_layers.values())

        self.mean_layers = nn.ModuleList()
        self.var_layers = nn.ModuleList()
        self.mean_bn_layers = nn.ModuleList()
        self.var_bn_layers = nn.ModuleList()

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.mean_layers.append(nn.Linear(prev_dim, h_dim))
            self.var_layers.append(nn.Linear(prev_dim, h_dim))

            self.mean_bn_layers.append(nn.BatchNorm1d(h_dim))
            self.var_bn_layers.append(nn.BatchNorm1d(h_dim))

            prev_dim = h_dim

        self.fc_mean = nn.Linear(prev_dim, latent_dim - 1)
        self.fc_var = nn.Linear(prev_dim, latent_dim - 1)

    def forward(self, x):
        h_mean = x
        for fc, bn in zip(self.mean_layers, self.mean_bn_layers):
            h_mean = F.relu(bn(fc(h_mean)))
        mean = self.fc_mean(h_mean)

        h_var = x
        for fc, bn in zip(self.var_layers, self.var_bn_layers):
            h_var = F.relu(bn(fc(h_var)))
        log_var = self.fc_var(h_var)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_layers, activation=None, sigma=None):
        super(Decoder, self).__init__()
        self.sigma = sigma

        self.lin_transform = LinearP(latent_dim, output_dim)
        self.nonlinear_transform = NonlinearTransform(output_dim, hidden_layers, activation)

    def forward(self, z):
        y = self.lin_transform(z)
        x = self.nonlinear_transform(y)
        return x


import torch
import torch.nn as nn


class NonlinearTransform(nn.Module):
    def __init__(self, output_dim, hidden_layers, activation=None):
        super(NonlinearTransform, self).__init__()
        self.component_wise_nets = nn.ModuleList([
            self._build_component_wise_net(hidden_layers, activation)
            for _ in range(output_dim)
        ])
        self._initialize_weights(activation) if len(hidden_layers) != 0 else None

    def _build_component_wise_net(self, hidden_layers, activation):
        if len(hidden_layers) == 0:
            return nn.Identity()
        layers = []
        input_size = 1  # Each component receives a scalar value
        for hidden_size in hidden_layers.values():
            layers.append(nn.Linear(input_size, hidden_size))
            if activation:
                layers.append(getattr(nn, activation)())  # Adding activation
            input_size = hidden_size
        # Final layer
        layers.append(nn.Linear(input_size, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self, activation):
        """ Initialize weights for the linear layers using normal distribution """
        for net in self.component_wise_nets:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity=activation.lower())
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Apply component-wise transformations to each part of x
        transformed_components = [
            self.component_wise_nets[i](x[..., i:i + 1]) for i in range(x.shape[-1])
        ]
        return torch.cat(transformed_components, dim=-1).abs()


class LinearP(nn.Linear):
    def __init__(self, input_dim, output_dim):
        super(LinearP, self).__init__(input_dim, output_dim, bias=False)
        nn.init.normal_(self.weight)

    @property
    def matrix(self):
        return self.weight.abs()

    def forward(self, input):
        return F.linear(input, self.matrix, self.bias)
