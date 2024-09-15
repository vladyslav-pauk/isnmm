from numpy import random

import torch
import torch.nn as nn
from torch.distributions import Distribution

from src.modules.network.nonlinear_transform import Network as NonlinearTransform
from src.modules.network.linear_positive import Network as LinearPositive


# todo: implement usgs semi-real data
class GenerativeModel(Distribution):
    def __init__(self, mixing_matrix_path=None, model="linear", degree=None, snr=None, seed=None):
        super().__init__()
        # random.seed(config['data_model']["seed"])
        #
        # self.model = config['data_model']["model"]
        # self.degree = config['data_model']["degree"]
        # self.snr_db = torch.tensor(10.0).pow(config['data_model']["SNR"] / 10)

        random.seed(seed)

        linear_transform_matrix = torch.eye(self.observed_dim, self.latent_dim)
        self.observed_dim = self.config_data_model["observed_dim"]
        self.latent_dim = self.config_data_model["latent_dim"]

        self.lin_transform_matrix = linear_transform_matrix
        self.observed_dim, self.latent_dim = linear_transform_matrix.shape

        self.model = model
        self.degree = degree
        self.snr_db = torch.tensor(10.0).pow(snr / 10)

        self.sigma = None

        # observed_dim = config['data_model']["observed_dim"]
        # latent_dim = config['data_model']["latent_dim"]
        self.latent_dist, self.noise_dist, self.lin_transform, self.nonlinear_transform = self._init_model(self.observed_dim,
                                                                                                           self.latent_dim)

    def sample(self, sample_shape=torch.Size()):
        latent_sample = self.latent_dist.sample(sample_shape)
        noise_sample = self.noise_dist.sample(sample_shape)

        transformed_sample = self.lin_transform(latent_sample)  # latent_sample @ self.lin_transform.matrix.T
        noiseless_sample = self.nonlinear_transform(transformed_sample)

        self.sigma = torch.sqrt(
            noiseless_sample.pow(2).sum() / noiseless_sample.numel() / self.snr_db
        )
        observed_sample = noiseless_sample + self.sigma * noise_sample

        return observed_sample, (latent_sample, noiseless_sample, transformed_sample)

    def _init_model(self, observed_dim, latent_dim):

        # joint = self.joint_log_prob()

        # lin_transform = torch.tensor(random.rand(observed_dim, latent_dim)).float()
        lin_transform = LinearPositive(latent_dim, observed_dim).requires_grad_(False)

        if self.model == "linear":
            nonlin_transform = nn.Identity()
        else:
            nonlin_transform = NonlinearTransform(observed_dim, self.degree).requires_grad_(False)

        latent_vec = torch.ones(latent_dim)
        latent_dist = torch.distributions.Dirichlet(concentration=latent_vec)

        noise_vec = torch.zeros(observed_dim)
        noise_mat = torch.eye(observed_dim)
        noise_dist = torch.distributions.MultivariateNormal(noise_vec, noise_mat)

        return latent_dist, noise_dist, lin_transform, nonlin_transform


if __name__ == "__main__":

    matrix = torch.tensor(random.rand(4, 3))
    model = GenerativeModel(linear_transform_matrix=matrix, model="linear", degree=3, snr=10, seed=0)

    x, (z, z_hat, z_tilde) = model.sample()