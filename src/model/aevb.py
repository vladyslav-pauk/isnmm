import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim

from src.modules.ae_module import AutoEncoderModule
from src.modules.network import FCN


class Model(AutoEncoderModule):
    def __init__(self, encoder=None, decoder=None, model_config=None, optimizer_config=None, **kwargs):
        super().__init__(encoder=encoder, decoder=decoder)

        self.metrics = torchmetrics.MetricCollection({})
        self.optimizer_config = optimizer_config

        self.log_monitor = {
            "monitor": "validation_loss",
            "mode": "min"
        }
        self.latent_dim = model_config["latent_dim"]
        self.mc_samples = model_config["mc_samples"]

    @staticmethod
    def reparameterization(sample):
        return sample

    @staticmethod
    def loss_function(data, model_output, idxes):
        reconstructed_sample = model_output["reconstructed_sample"]
        mu, std = model_output["latent_parameterization_batch"]

        bce = F.binary_cross_entropy(reconstructed_sample, data.expand_as(reconstructed_sample), reduction='sum') / data.size(0)
        kld = -0.5 * torch.sum(1 + 2 * torch.log(std + 1e-12) - mu.pow(2) - std.pow(2)) / data.size(0)
        return {"cross-entropy": bce, "kl_divergence": kld}

    def configure_optimizers(self):
        lr = self.optimizer_config["lr"]
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr["encoder"]},
            {'params': self.decoder.parameters(), 'lr': lr["decoder"]}
        ], **self.optimizer_config["params"])
        return optimizer

    def update_metrics(self, data, model_output, labels, idxes):
        pass


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, latent_dim, observed_dim):
        self.mu_network = FCN(observed_dim, latent_dim, **self.config)
        self.log_var_network = FCN(observed_dim, latent_dim, **self.config)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        std = torch.exp(0.5 * log_var)
        return mu, std


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, latent_dim, observed_dim):
        self.network = FCN(latent_dim, observed_dim, **self.config)

    def forward(self, z):
        return self.network.forward(z)
