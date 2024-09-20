import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Distribution, constraints


from src.modules.transform import NonlinearComponentWise as NonlinearTransform
from src.modules.network.linear_positive import Network as LinearPositive


class GenerativeModel:  # (Distribution)
    # arg_constraints = {"degree": constraints.positive, "linear_mixing": constraints.positive}
    # support = constraints.simplex

    def __init__(self, linear_mixture_matrix, model_name=None, mixing_matrix_init=None, nonlinear_transform_init=None, model="linear", degree=None, snr=None):
        super().__init__()

        self.model = model
        self.degree = degree
        self.snr_db = torch.tensor(10.0).pow(snr / 10)
        self.sigma = None
        self.mixing_matrix_init = mixing_matrix_init
        self.nonlinear_transform_init = nonlinear_transform_init

        self.observed_dim, self.latent_dim = linear_mixture_matrix.shape
        self.latent_dist, self.noise_dist, self.linear_mixture, self.nonlinear_transform = self._init_model(linear_mixture_matrix)
        # todo: remove unnecessary attributes or arguments, e.g. model_name, or pass directly to _model_init functions

    def _init_model(self, linear_mixture_matrix):
        linear_mixture = LinearPositive(
            linear_mixture_matrix, self.mixing_matrix_init
        ).requires_grad_(False)

        nonlinear_transform = NonlinearTransform(
            self.observed_dim,
            self.degree,
            self.model,
            init_weights=self.nonlinear_transform_init
        ).requires_grad_(False)

        latent_vec = torch.ones(self.latent_dim)
        latent_dist = torch.distributions.Dirichlet(concentration=latent_vec)

        noise_vec = torch.zeros(self.observed_dim)
        noise_mat = torch.eye(self.observed_dim)
        noise_dist = torch.distributions.MultivariateNormal(noise_vec, noise_mat)

        return latent_dist, noise_dist, linear_mixture, nonlinear_transform
        # todo: define separate subclasses for each model type, inheriting from a base class.
        #  separate it into lmm, nmm, pnlmm subclasses (or just make Identity a part of nonlinear transform)
        #  make a generative_model for mixture models (like vae), then lmm, nmm, pnlmm are in folder generative_models
        #  use them for synthetic data
        # todo: inherit from Distribution and Dataset and add __len__ and __getitem__ methods

    def sample(self, sample_shape=None):
        latent_sample = self.latent_dist.sample(torch.Size(sample_shape))
        noise_sample = self.noise_dist.sample(torch.Size(sample_shape))

        linearly_mixed_sample = self.linear_mixture(latent_sample)  # latent_sample @ self.lin_transform.matrix.T
        noiseless_sample = self.nonlinear_transform(linearly_mixed_sample)

        self.sigma = torch.sqrt(
            noiseless_sample.pow(2).sum() / noiseless_sample.numel() / self.snr_db
        )
        observed_sample = noiseless_sample + self.sigma * noise_sample

        return observed_sample, (latent_sample, linearly_mixed_sample, noiseless_sample)
