import torch
from torch import nn
from torch.nn import functional as F
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
        # sample = torch.cat((sample, torch.zeros_like(sample[..., :1])), dim=-1)
        # F.softmax(sample, dim=-1)
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
        self.network = None

    def construct(self, latent_dim, observed_dim):
        self.network = self.constructor(observed_dim, latent_dim, **self.config)

    def forward(self, x):
        z = self.network.forward(x)
        return z, torch.zeros_like(z).to(z.device)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        # self.linear_mixture = network.LinearPositive(
        #     torch.eye(observed_dim, latent_dim), **self.config
        # )
        # self.linear_mixture.eval()
        self.nonlinear_transform = self.constructor(latent_dim, observed_dim, **self.config)

    def forward(self, z):
        y = self.linear_mixture(z)
        x = self.nonlinear_transform(y)
        return x

# class Decoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.constructor = getattr(network, config["constructor"])
#
#     def construct(self, latent_dim, observed_dim):
#         self.linear_mixture = network.LinearPositive(
#             torch.rand(observed_dim, latent_dim), **self.config
#         )
#
#         self.nonlinearity = nn.ModuleList([self.constructor(
#             input_dim=1, output_dim=1, **self.config
#         ) for _ in range(observed_dim)])
#
#     def nonlinear_transform(self, x):
#         x = torch.cat([
#             self.nonlinearity[i](x[..., i:i + 1].view(-1, 1)).view_as(x[..., i:i + 1])
#             for i in range(x.shape[-1])
#         ], dim=-1)
#         return x
#
#     def forward(self, z):
#         y = self.linear_mixture(z)
#         x = self.nonlinear_transform(y)
#         return x

# fixme: run for 3->4 dims and latent_dim-1 so it has smae structure as vasca
# todo: proper cnn with linear layer and proper postnonlinearity (make a separate class PNLConstructor for FCN or CNN)
# todo: clean up and readme
