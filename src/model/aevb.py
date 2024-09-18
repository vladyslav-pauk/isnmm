import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

from src.modules.vae_module import VAEModule
from src.modules.network import FCNConstructor


class Model(VAEModule):
    def __init__(self, encoder=None, decoder=None, train_config=None, **kwargs):
        super().__init__(encoder=encoder, decoder=decoder, **train_config)

        self.metrics = torchmetrics.MetricCollection({})

    def reparameterize(self, params, mc_samples):
        mean, log_var = params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    @staticmethod
    def loss_function(data, model_output):
        x = data
        recon_x, _, (mu, log_var) = model_output
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return {"cross-entropy": bce, "kl_divergence": kld}

    def update_metrics(self, data, model_output, labels):
        pass


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, config_encoder):
        super().__init__()

        self.mu_network = FCNConstructor(input_dim, latent_dim, **config_encoder)
        self.log_var_network = FCNConstructor(input_dim, latent_dim, **config_encoder)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, config_decoder):
        super().__init__()

        self.network = FCNConstructor(latent_dim, output_dim, **config_decoder)

    def forward(self, z):
        return self.network.forward(z)
