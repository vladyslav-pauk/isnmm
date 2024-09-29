import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim

from src.modules.vae_module import VAEModule
from src.modules.network import FCNConstructor


class Model(VAEModule):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None, **kwargs):
        super().__init__(encoder=encoder, decoder=decoder, **model_config)

        self.metrics = torchmetrics.MetricCollection({})
        self.optimizer_config = optimizer_config

        self.log_monitor = {
            "monitor": "validation_loss",
            "mode": "min"
        }

    def reparameterize(self, params, mc_samples):
        mean, log_var = params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent_mc_sample = eps.mul(std).add_(mean)
        return latent_mc_sample

    @staticmethod
    def loss_function(data, model_output):
        x = data
        recon_x, _, (mu, log_var) = model_output
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return {"cross-entropy": bce, "kl_divergence": kld}

    def configure_optimizers(self):
        lr = self.optimizer_config["lr"]
        lr_th = lr["th"]
        lr_ph = lr["ph"]
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr_ph},
            {'params': self.decoder.parameters(), 'lr': lr_th}
        ], **self.optimizer_config["params"])
        return optimizer

    def update_metrics(self, data, model_output, labels):
        pass


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, latent_dim, observed_dim):
        self.mu_network = FCNConstructor(observed_dim, latent_dim, **self.config)
        self.log_var_network = FCNConstructor(observed_dim, latent_dim, **self.config)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, latent_dim, observed_dim):
        self.network = FCNConstructor(latent_dim, observed_dim, **self.config)

    def forward(self, z):
        return self.network.forward(z)
