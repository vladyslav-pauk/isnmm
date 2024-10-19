import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt

from ..utils import dict_to_str


class GenerativeModel:
    def __init__(self, linear_mixture_matrix=None, data_model_name=None, **config):
        self.config = config
        self.seed = config.get("seed", None)
        if self.seed:
            seed_everything(self.seed, workers=True)

        self.observed_dim = config["observed_dim"]
        self.latent_dim = config["latent_dim"]
        self.num_samples = config["dataset_size"]
        self.mixing_scale_factors = config.get("mixing_scale_factors", [5.0, 4.0, 1.0])

        # Initialize random mixing matrix for linear mixtures
        self.mixing_matrix = torch.randn(self.observed_dim, self.latent_dim)

        # Generate latent variables on the simplex
        latent_data = torch.rand(self.num_samples, self.latent_dim)
        self.latent_sample = latent_data / latent_data.sum(dim=1, keepdim=True)

        # QR decomposition for later use (evaluation purposes)
        self.q, _ = torch.linalg.qr(self.latent_sample)

        # Generate linear and nonlinear mixtures
        self.linear_mixture = self.create_linear_mixture()
        self.nonlinear_mixture = self.create_nonlinear_mixture()

    def create_linear_mixture(self):
        # Linear mixture scaled by given factors
        return (self.latent_sample @ self.mixing_matrix.T) * torch.tensor(self.mixing_scale_factors)

    def create_nonlinear_mixture(self):
        # Nonlinear transformations for each component of the mixture
        nonlinear_mixture = torch.zeros_like(self.linear_mixture)

        # Apply the nonlinearities (same as in SyntheticCNAE)
        nonlinear_mixture[:, 0] = 5 * torch.sigmoid(self.linear_mixture[:, 0]) + 0.3 * self.linear_mixture[:, 0]
        nonlinear_mixture[:, 1] = -3 * torch.tanh(self.linear_mixture[:, 1]) - 0.2 * self.linear_mixture[:, 1]
        # nonlinear_mixture[:, 2] = 0.4 * torch.exp(self.linear_mixture[:, 2])

        return nonlinear_mixture

    def model(self):
        return self.nonlinear_mixture, (self.latent_sample, self.linear_mixture, self.q)

    def sample(self, sample_shape=torch.Size([1])):
        return self.model()

    def plot_sample(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(self.linear_mixture[:, 0], self.linear_mixture[:, 1], c='b', label='latent')
        ax[1].scatter(self.nonlinear_mixture[:, 0], self.nonlinear_mixture[:, 1], c='r', label='observed')
        plt.show()

    def plot_nonlinearities(self):
        for i in range(self.observed_dim):
            plt.scatter(self.linear_mixture[:, i], self.nonlinear_mixture[:, i])
            plt.title(f'Nonlinearity in dimension {i+1}')
            plt.show()

    def save_data(self):

        dataset_name = dict_to_str(self.config)
        filename = f'../datasets/synthetic/{dataset_name}.mat'
        sio.savemat(filename, {
            'observed_sample': self.nonlinear_mixture.cpu().numpy(),
            'noiseless_sample': self.nonlinear_mixture.cpu().numpy(),
            'latent_sample': self.latent_sample.cpu().numpy(),
            'latent_sample_qr': self.q.cpu().numpy(),
            'linearly_mixed_sample': self.linear_mixture.cpu().numpy(),
            'linear_mixture': self.mixing_matrix.cpu().numpy(),
            'sigma': 1.0  # Can be adjusted if noise is added
        })


if __name__ == "__main__":
    config = {
        "observed_dim": 2,
        "latent_dim": 3,
        "dataset_size": 1000,
        "mixing_scale_factors": [5.0, 4.0], #, 1.0],
        "seed": 42
    }
    model = GenerativeModel(**config)
    model.plot_sample()
    model.plot_nonlinearities()