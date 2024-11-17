import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions import Distribution, constraints
from pytorch_lightning import seed_everything

from src.modules.transform import NonlinearComponentWise as NonlinearTransform
# from src.modules.transform import NonlinearDimensionReduction as NonlinearTransform
from src.modules.network.linear_positive import Network as LinearPositive
from src.modules.utils import dict_to_str


class GenerativeModel:  # (Distribution)
    # arg_constraints = {"degree": constraints.positive, "linear_mixing": constraints.positive}
    # support = constraints.simplex

    def __init__(self, linear_mixture_matrix=None, **config):
        super().__init__()
        # observed_dim = None,
        # latent_dim = None,
        # model_name = None,
        # mixing_matrix_init = "none",
        # nonlinear_transform_init = "none",
        # nonlinearity = "linear",
        # degree = None,
        # snr = None,
        # seed = None,
        self.config = config
        self.seed = config["data_seed"]
        if self.seed:
            seed_everything(self.seed, workers=True)

        self.nonlinearity = config["data_model"]
        self.degree = config["degree"]
        self.snr_db = config["snr"]
        self.sigma = None
        self.mixing_matrix_init = config["mixing_matrix_init"]
        self.nonlinear_transform_init = config["nonlinear_transform_init"]

        if linear_mixture_matrix is not None:
            self.observed_dim, self.latent_dim = linear_mixture_matrix.shape
        else:
            self.observed_dim, self.latent_dim = config["observed_dim"], config["latent_dim"]
            linear_mixture_matrix = torch.randn(self.observed_dim, self.latent_dim)

        self.latent_dist, self.noise_dist, self.linear_mixture, self.nonlinear_transform = self._init_model(linear_mixture_matrix)

    def _init_model(self, linear_mixture_matrix):
        linear_mixture = LinearPositive(
            linear_mixture_matrix, self.mixing_matrix_init, scale=self.config["scale"]
        ).requires_grad_(False)

        nonlinear_transform = NonlinearTransform(
            self.latent_dim,
            self.observed_dim,
            self.degree,
            self.nonlinearity,
            init_weights=self.nonlinear_transform_init
        ).requires_grad_(False)

        latent_vec = torch.ones(self.latent_dim)
        latent_dist = torch.distributions.Dirichlet(concentration=latent_vec)

        noise_vec = torch.zeros(self.observed_dim)
        noise_mat = torch.eye(self.observed_dim)
        noise_dist = torch.distributions.MultivariateNormal(noise_vec, noise_mat)

        return latent_dist, noise_dist, linear_mixture, nonlinear_transform
        # task: refactor like the model, so i can generate different models with config
        # task: refactor data_model so it has a forward method so i can run inference like on model
        # task: inherit from Distribution and Dataset and add __len__ and __getitem__ methods

    def model(self, latent_sample, noise_sample):
        self.linearly_mixed_sample = self.linear_mixture(latent_sample)  # latent_sample @ self.lin_transform.matrix.T
        self.noiseless_sample = self.nonlinear_transform(self.linearly_mixed_sample)

        snr = torch.tensor(10.0).pow(self.snr_db / 10)
        self.sigma = torch.sqrt(
            self.noiseless_sample.pow(2).sum() / self.noiseless_sample.numel() / snr
        )
        if self.snr_db is None:
            self.sigma = torch.tensor(0.0)
            
        self.observed_sample = self.noiseless_sample + self.sigma * noise_sample
        return self.observed_sample, (self.latent_sample, self.linearly_mixed_sample, self.noiseless_sample)
        # task: use dicts instead of tuples for forward method

    def sample(self):
        sample_shape = torch.Size([self.config["dataset_size"]])
        self.latent_sample = self.latent_dist.sample(sample_shape)
        self.noise_sample = self.noise_dist.sample(sample_shape)
        self.latent_sample_qr, _ = np.linalg.qr(self.latent_sample)
        return self.model(self.latent_sample, self.noise_sample)

    def plot_sample(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(self.linearly_mixed_sample[:, 0], self.linearly_mixed_sample[:, 1], c='b', label='latent')
        ax[1].scatter(self.observed_sample[:, 0], self.observed_sample[:, 1], c='r', label='observed')
        plt.show()

    def plot_nonlinearities(self):
        for i in range(self.observed_dim):
            plt.scatter(self.linearly_mixed_sample[:, i], self.noiseless_sample[:, i])
            plt.show()

    def save_data(self):

        dataset_name = dict_to_str(self.config)
        filename = f'../datasets/synthetic/{dataset_name}.mat'

        sio.savemat(filename, {
            'observed_sample': self.observed_sample.cpu().numpy(),
            'latent_sample': self.latent_sample.cpu().numpy(),
            'noiseless_sample': self.noiseless_sample.cpu().numpy(),
            'linearly_mixed_sample': self.linearly_mixed_sample.cpu().numpy(),
            'latent_sample_qr': self.latent_sample_qr.cpu().numpy(),
            'linear_mixture': self.linear_mixture.matrix.cpu().numpy(),
            'sigma': self.sigma.cpu().numpy()
        })


if __name__ == "__main__":
    from src.helpers.generate_data import initialize_data_model

    config = {
        # "data_model": "cnae",
        # "experiment_name": "nonlinearity_removal",
        "nonlinearity": "cnae",
        "observed_dim": 2,
        "latent_dim": 3,
        "dataset_size": 1000,
        "mixing_matrix_init": "none",
        "nonlinear_transform_init": "none",
        "degree": None,
        "snr": 25,
        "seed": 1,
        "data_seed": 1,
        "scale": 5.0
    }

    # model = GenerativeModel(**config)
    data_model = initialize_data_model(experiment_name="synthetic_data", **config)
    data_model.sample()
    data_model.plot_sample()
    data_model.plot_nonlinearities()

# todo: implement usgs semi-real data and realistic nonlinearity (physics model)
