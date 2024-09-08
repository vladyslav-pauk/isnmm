import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = latent_dim

        hidden_dim = hidden_layers["h1"]
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_layers, output_activation=None):
        super().__init__()

        hidden_dim = hidden_layers["h1"]
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.output_activation = getattr(F, output_activation, None)

    def forward(self, z):
        h = self.LeakyReLU(self.FC_hidden(z))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.output_activation(self.FC_output(h))
        return x_hat
