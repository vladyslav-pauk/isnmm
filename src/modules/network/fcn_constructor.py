import torch.nn as nn
from torch.nn import functional as F


class FCNConstructor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, activation=None, output_activation=None, weight_initialization=None):
        super().__init__()

        self.activation = getattr(F, activation, lambda x: x)
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
        self.init_weights(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_activation(self.output_layer(x))

# todo: use in-place activation functions where possible, e.g., F.relu_(h)? or use nn.ReLU(inplace=True) layers

# class EncoderConstructor(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_layers, activation=None, init_weights=None):
#         super().__init__()
#
#         self.activation = getattr(F, activation, lambda x: x)
#         self.init_weights = getattr(nn.init, init_weights, lambda x: x)
#
#         hidden_dims = list(hidden_layers.values())
#         self.mu_layers = nn.ModuleList(
#             [self.init_layer(
#                 nn.Linear(input_dim if i == 0 else hidden_dims[i-1], h)
#             ) for i, h in enumerate(hidden_dims)]
#         )
#         self.log_var_layers = nn.ModuleList(
#             [self.init_layer(
#                 nn.Linear(input_dim if i == 0 else hidden_dims[i-1], h)
#             ) for i, h in enumerate(hidden_dims)]
#         )
#
#         self.output_mu = self.init_layer(nn.Linear(hidden_dims[-1], latent_dim))
#         self.output_log_var = self.init_layer(nn.Linear(hidden_dims[-1], latent_dim))
#
#     def init_layer(self, layer):
#         self.init_weights(layer.weight)
#         nn.init.zeros_(layer.bias)
#         return nn.Sequential(layer, nn.BatchNorm1d(layer.out_features))
#
#     def forward(self, x):
#         h_mu = x
#         for layer in self.mu_layers:
#             h_mu = self.activation(layer(h_mu))
#
#         h_log_var = x
#         for layer in self.log_var_layers:
#             h_log_var = self.activation(layer(h_log_var))
#
#         return self.output_mu(h_mu), self.output_log_var(h_log_var)


#
# class EncoderConstructor(nn.Module):
#     def __init__(self, input_dim=None, latent_dim=None, hidden_layers=None, activation=None, init_weights=None):
#         super().__init__()
#         if activation:
#             self.activation = getattr(F, activation)
#         else:
#             self.activation = lambda x: x
#
#         hidden_dims = list(hidden_layers.values())
#
#         self.mu_layers = self.build_layers(input_dim, hidden_dims, init_weights)
#         self.log_var_layers = self.build_layers(input_dim, hidden_dims, init_weights)
#
#         self.output_mu = None
#         self.output_log_var = None
#
#     def build_layers(self, input_dim, hidden_dims, init_weights):
#         layers = nn.ModuleList()
#         prev_dim = input_dim
#         for h_dim in hidden_dims:
#             layer = nn.Linear(prev_dim, h_dim)
#             if init_weights:
#                 getattr(nn.init, init_weights)(layer.weight)
#                 nn.init.zeros_(layer.bias)
#                 # getattr(nn.init, init_weights)(layer.bias)
#             layers.append(nn.Sequential(
#                 layer,
#                 nn.BatchNorm1d(h_dim)
#             ))
#             prev_dim = h_dim
#         return layers
#
#     def forward(self, x):
#         h_mu = x
#         for layer in self.mu_layers:
#             h_mu = self.activation(layer(h_mu))
#
#         h_log_var = x
#         for layer in self.log_var_layers:
#             h_log_var = self.activation(layer(h_log_var))
#
#         return self.output_mu(h_mu), self.output_log_var(h_log_var)

# class Encoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_layers):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = latent_dim
#
#         hidden_dims = list(hidden_layers.values())
#
#         self.mean_layers = nn.ModuleList()
#         self.var_layers = nn.ModuleList()
#         self.mean_bn_layers = nn.ModuleList()
#         self.var_bn_layers = nn.ModuleList()
#
#         prev_dim = input_dim
#         for h_dim in hidden_dims:
#             self.mean_layers.append(nn.Linear(prev_dim, h_dim))
#             self.var_layers.append(nn.Linear(prev_dim, h_dim))
#
#             self.mean_bn_layers.append(nn.BatchNorm1d(h_dim))
#             self.var_bn_layers.append(nn.BatchNorm1d(h_dim))
#
#             prev_dim = h_dim
#
#         self.fc_mean = nn.Linear(prev_dim, latent_dim - 1)
#         self.fc_var = nn.Linear(prev_dim, latent_dim - 1)
#
#     def forward(self, x):
#         h_mean = x
#         for fc, bn in zip(self.mean_layers, self.mean_bn_layers):
#             h_mean = F.relu(bn(fc(h_mean)))
#         mean = self.fc_mean(h_mean)
#
#         h_var = x
#         for fc, bn in zip(self.var_layers, self.var_bn_layers):
#             h_var = F.relu(bn(fc(h_var)))
#         log_var = self.fc_var(h_var)
#
#         return mean, log_var