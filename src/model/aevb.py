import torch
import torch.nn.functional as F

from src.modules.training_module import VAE
from torch.distributions import MultivariateNormal


class Model(VAE):
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
