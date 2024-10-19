import torch.nn as nn


class CNNConstructor(nn.Module):
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

        conv_dims = list(hidden_layers.values())
        self.init_weights = getattr(nn.init, weight_initialization) if weight_initialization else lambda x: x
        self.layers = self.build_network(input_dim, output_dim, conv_dims, hidden_activation, output_activation, bn_eps, bn_momentum, dropout_rate)

    def build_network(self, input_dim, output_dim, conv_dims, hidden_activation, output_activation, bn_eps, bn_momentum, dropout_rate):

        layers = nn.ModuleList([])
        hidden_activation = getattr(nn, hidden_activation) if hidden_activation else nn.Identity
        output_activation = getattr(nn, output_activation) if output_activation else nn.Identity

        layers.append(nn.Conv1d(input_dim, conv_dims[0] * input_dim, kernel_size=1, groups=input_dim))
        layers.append(hidden_activation())

        for i in range(1, len(conv_dims) - 1):
            layers.append(
                self.init_weights(nn.Conv1d(conv_dims[i - 1] * input_dim, conv_dims[i] * input_dim, kernel_size=1, groups=input_dim))
            )
            layers.append(
                nn.BatchNorm1d(conv_dims[i], eps=bn_eps, momentum=bn_momentum) if bn_eps else nn.Identity()
            )
            layers.append(
                hidden_activation()
            )
            layers.append(
                nn.Dropout(dropout_rate if dropout_rate else 0) if dropout_rate else nn.Identity()
            )

        layers.append(nn.Conv1d(conv_dims[-1] * input_dim, output_dim, kernel_size=1, groups=input_dim))
        layers.append(output_activation())

        return nn.Sequential(*layers)

    def forward(self, x):
        monte_carlo = len(x.size()) == 3
        batch_size, *dims = x.size()
        if monte_carlo:
            x = x.view(-1, *dims[1:])
        x = self.layers(x.unsqueeze(-1)).squeeze(-1)
        return x.view(batch_size, *dims) if monte_carlo else x


# class Decoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#
#     def construct(self, input_dim, output_dim):
#         layers = []
#         in_channels = input_dim
#         hidden_sizes = list(self.config.get('hidden_layers').values())
#
#         for h in hidden_sizes:
#             layers.append(nn.Conv1d(in_channels, h * input_dim, kernel_size=1, groups=input_dim))
#             layers.append(nn.ReLU())
#             in_channels = h * input_dim
# task: use this method in cnn build
#
#         layers.append(nn.Conv1d(in_channels, output_dim, kernel_size=1, groups=input_dim))
#         self.d_net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         num_samples, monte_carlo_samples, components = x.shape
#         x = x.view(-1, components).unsqueeze(-1)
#         x = self.d_net(x)
#         x = x.squeeze().view(num_samples, monte_carlo_samples, -1)
#         return x