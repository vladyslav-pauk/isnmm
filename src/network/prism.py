import torch
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
    def __init__(self, latent_dim, output_dim, hidden_layers, output_activation=None, sigma=None):
        super(Decoder, self).__init__()

        self.sigma = sigma
        self.lin_transform = LinearP(latent_dim, output_dim)
        # nn.init.eye_(self.linear_A.weight)

        # output_activation = getattr(F, output_activation, None)
        # self.component_wise_nets = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(1, hidden_layers["h1"]),
        #         output_activation(),
        #         nn.Linear(hidden_layers["h2"], 1)
        #     )
        #     for _ in range(output_dim)
        # ])

    def forward(self, z):
        x = self.lin_transform(z)
        sigma = self.sigma
        # x = torch.cat([
        #     self.component_wise_nets[i](Az[:, i:i + 1]) for i in range(Az.shape[1])
        # ], dim=1)
        return x, sigma


class LinearP(nn.Linear):
    def __init__(self, input_dim, output_dim):
        super(LinearP, self).__init__(input_dim, output_dim, bias=False)
        nn.init.normal_(self.weight)

    @property
    def matrix(self):
        return self.weight.abs()

    def forward(self, input):
        return F.linear(input, self.matrix, self.bias)
