import torch
from torch import nn
from torch import optim

from src.model.modules.metric_post_nonlinear import PNL
from src.model.modules.autoencoder import AE
import src.modules.network as network


class Model(AE, PNL):
    def __init__(self, encoder, decoder, model_config, optimizer_config):
        super().__init__(encoder, decoder)

        self.optimizer_config = optimizer_config
        self.latent_dim = model_config["latent_dim"]
        self.mc_samples = 1
        self.sigma = 0

    @staticmethod
    def _reparameterization(sample):
        sample = sample / sample.sum(dim=-1).unsqueeze(-1)
        return sample

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        lr = self.optimizer_config["lr"]
        self.optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr["encoder"]},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr["decoder"]["linear"]},
            {'params': self.decoder.nonlinear_transform.parameters(), 'lr': lr["decoder"]["nonlinear"]}
        ], **self.optimizer_config["params"])
        return self.optimizer


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.nonlinear_transform = None
        self.linear_mixture_inv = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)
        self.linear_mixture_inv = network.LinearPositive(
            torch.rand(latent_dim, observed_dim), **self.config
        )

    def forward(self, x):
        y = self.nonlinear_transform.forward(x)
        z = self.linear_mixture_inv(y)
        return z, torch.zeros_like(z).to(z.device)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = network.LinearPositive(
            torch.rand(observed_dim, latent_dim), **self.config
        )
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

    def forward(self, z):
        y = self.linear_mixture(z)
        x = self.nonlinear_transform(y)
        return x

# todo: unequal dimensions (as in paper with blocks)
