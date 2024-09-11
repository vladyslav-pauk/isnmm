from numpy import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NonlinearTransform(nn.Module):
    def __init__(self, observed_dim, degree):
        super(NonlinearTransform, self).__init__()
        self.degree = degree
        self.coefficients = torch.tensor(random.rand(observed_dim, degree + 1))

    def forward(self, x):
        """ Apply the transformation component-wise: tanh(x), tanh(2x), ..., tanh(nx) """
        transformed_components = [
            torch.stack([coeff * torch.tanh((power + 1) * x[..., i:i + 1]**power) for power, coeff in enumerate(self.coefficients[i])]).sum(dim=0)
            for i in range(x.shape[-1])
        ]
        return torch.cat(transformed_components, dim=-1)

    # def inverse(self, y, num_iterations=1000):
    #     """ Numerically approximate the inverse transformation using Newton's method """
    #     x_approx = y.clone()
    #     for _ in range(num_iterations):
    #         f_x = self.forward(x_approx) - y
    #         f_x_prime = self.derivative(x_approx)
    #         x_approx -= f_x / (f_x_prime + 1e-6)
    #     return x_approx

    def inverse(self, y, tol=1e-6, max_iter=10000):
        # Initialize x with y as an approximation (you can improve this)
        x = y.clone()

        for _ in range(max_iter):
            f_x = self.forward(x)
            diff = y - f_x
            if torch.norm(diff) < tol:
                break

            # Update x using a numerical method, like Newton's method
            # Here you would compute the Jacobian or use a simpler gradient-based update
            # For simplicity, we just do a fixed-step gradient descent update
            x = x + diff * 0.1  # Step size 0.1; could be adaptive

        return x


class DataModel:
    def __init__(self, config):
        super().__init__()
        random.seed(config['data']["seed"])

        observed_dim = config['data']["observed_dim"]
        latent_dim = config['data']["latent_dim"]

        self.model = config['data']["model"]
        self.degree = config['data']["degree"]

        self.snr_db = torch.tensor(10.0).pow(config['data']["SNR"] / 10)
        self.sigma = None

        self.latent_dist, self.noise_dist, self.lin_transform, self.nonlinear_transform = self._init_model(observed_dim,
                                                                                                           latent_dim)

    def sample(self, sample_shape=torch.Size()):
        latent_sample = self.latent_dist.sample(sample_shape)
        noise_sample = self.noise_dist.sample(sample_shape)

        transformed_sample = latent_sample @ self.lin_transform.T
        noiseless_sample = self.nonlinear_transform(transformed_sample)

        self.sigma = torch.sqrt(
            noiseless_sample.pow(2).sum() / noiseless_sample.numel() / self.snr_db
        )
        observed_sample = noiseless_sample + self.sigma * noise_sample

        return (observed_sample, latent_sample), self.sigma

    def _init_model(self, observed_dim, latent_dim):
        lin_transform = torch.tensor(random.rand(observed_dim, latent_dim)).float()

        if self.model == "linear":
            nonlin_transform = nn.Identity()
        else:
            nonlin_transform = NonlinearTransform(observed_dim, self.degree)

        latent_vec = torch.ones(latent_dim)
        latent_dist = torch.distributions.Dirichlet(concentration=latent_vec)

        noise_vec = torch.zeros(observed_dim)
        noise_mat = torch.eye(observed_dim)
        noise_dist = torch.distributions.MultivariateNormal(noise_vec, noise_mat)

        return latent_dist, noise_dist, lin_transform, nonlin_transform


class SyntheticDataset(Dataset):
    def __init__(self, config):
        self.data_size = config['data']["size"]
        data_model = DataModel(config)
        self.lin_transform = data_model.lin_transform
        self.nonlinear_transform = data_model.nonlinear_transform
        self.data, self.sigma = data_model.sample(sample_shape=torch.Size([config['data']["size"]]))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)
