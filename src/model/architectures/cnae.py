import torch
from torch import nn

from src.modules.optimizer.augmented_lagrange import AugmentedLagrangeMultiplier
import src.modules.network as network
import src.experiments as exp
from src.model.modules.lightning_module import Module as LightningModule
from src.model.modules.ae import Module as Autoencoder


class Model(LightningModule, Autoencoder):
    def __init__(self, encoder, decoder, model_config, optimizer_config):
        super().__init__(encoder, decoder)

        self.optimizer = None
        self.optimizer_config = optimizer_config

        self.latent_dim = model_config["latent_dim"]
        self.mc_samples = 1
        self.sigma = 0

        self.distance = model_config["distance"]
        self.experiment_metrics = model_config["experiment_name"]
        self.encoder_transform = model_config["reparameterization"]

    def _regularization_loss(self, model_output, observed_batch, idxes):
        latent_sample = model_output["latent_sample"]
        return self.optimizer.compute_regularization_loss(latent_sample.mean(dim=0), observed_batch.mean(dim=0), idxes)

    def configure_optimizers(self):
        self.optimizer = AugmentedLagrangeMultiplier(
            params=list(self.parameters()),
            constraint_fn=self._constraint,
            optimizer_config=self.optimizer_config
        )
        return self.optimizer

    @staticmethod
    def _constraint(latent_sample):
        return torch.sum(latent_sample, dim=-1) - 1.0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.optimizer.update_buffers(batch["idxes"], self(batch["data"])["latent_sample"])


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture_inv = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        if latent_dim != observed_dim:
            self.linear_mixture_inv = network.LinearPositive(
                torch.eye(latent_dim, observed_dim), **self.config
            )
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

    def forward(self, x):
        x = self.nonlinear_transform(x)
        x = self.linear_mixture_inv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        if latent_dim != observed_dim:
            self.linear_mixture = network.LinearPositive(
                torch.eye(observed_dim, latent_dim), **self.config
            )
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

    def forward(self, x):
        x = self.linear_mixture(x)
        x = self.nonlinear_transform(x)
        return x
