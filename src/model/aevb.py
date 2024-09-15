import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.vae_module import VAEModule
from torch.distributions import MultivariateNormal


class Model(VAEModule):
    def __init__(self, encoder=None, decoder=None, data_model=None):
        super().__init__(encoder=encoder, decoder=decoder)

        self.observed_dim = encoder.input_dim
        self.latent_dim = encoder.output_dim

        self.latent_prior_distribution = MultivariateNormal
        self.noise_distribution = MultivariateNormal
        self.variational_posterior_distribution = MultivariateNormal

    def generative_model(self, latent, noise):
        observed = self.decoder(latent) + noise
        return observed

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def inference_model(self, observed):
        mean, log_var = self.encoder(observed)
        covariance = log_var.exp().diag_embed()
        distribution = self.variational_posterior_distribution(loc=mean, covariance_matrix=covariance)
        return distribution

    def likelihood(self, latent):
        decoded_latent = self.decoder(latent)
        noise_covariance = 0.1 * torch.eye(decoded_latent.shape[1])
        distribution = self.noise_distribution(loc=decoded_latent, covariance_matrix=noise_covariance)
        return distribution

    def prior(self):
        distribution = self.latent_prior_distribution(
            loc=torch.zeros(self.latent_dim), covariance_matrix=torch.eye(self.latent_dim)
        )
        return distribution

    def marginal_likelihood(self):
        pass

    @staticmethod
    def loss_function(x, recon_x, mu, log_var):
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return {"cross-entropy": bce, "kl_divergence": kld}

    def metric(self):
        return {}


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
