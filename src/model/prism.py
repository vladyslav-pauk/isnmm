import sys

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.network import ComponentWiseNonlinear, LinearPositive
from torch.distributions import Dirichlet, LogNormal, Normal
from src.modules.vae_module import VAEModule


class Model(VAEModule):
    def __init__(self, encoder=None, decoder=None, data_model=None, mc_samples=1, lr=None, metrics=None, monitor=None, config=None, data_config=None):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr)

        self.observed_dim = encoder.input_dim
        self.latent_dim = encoder.output_dim

        self.latent_prior_distribution = Dirichlet(torch.ones(self.latent_dim))
        self.noise_distribution = Normal
        self.variational_posterior_distribution = LogNormal

        self.data_model = data_model

        self.mc_samples = mc_samples
        self.metrics = metrics
        self.monitor = monitor

    # def reparameterize(self, mean, log_var):
    #     std = torch.exp(0.5 * log_var)
    #     eps = torch.randn_like(std)
    #     z_log = mean + std * eps
    #     z = F.softmax(z_log, dim=-1)
    #     return z

    def reparameterize(self, params):
        mean, log_var = params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(self.mc_samples, *std.shape)
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        samples = torch.cat((samples, torch.zeros(samples.shape[0], samples.shape[1], 1)), dim=2)
        z = F.softmax(samples, dim=-1)
        return z

    # def generative_model(self, z, noise):
    #     noiseless = self.decoder(z)
    #     return noiseless + noise
    # def inference_model(self, observed):
    #     mean, log_var = self.encoder(observed)
    #     std = torch.exp(0.5 * log_var)
    #     return self.variational_posterior_distribution(mean, std)
    #
    # def likelihood(self, z):
    #     noiseless_sample = self.decoder(z)
    #     noise_amp = torch.sqrt(
    #         noiseless_sample.pow(2).sum() / noiseless_sample.shape[0] / noiseless_sample.shape[1] / self.snr
    #     )
    #     noise = self.noise_distribution(loc=torch.zeros_like(noiseless_sample), scale=noise_amp)
    #     noise_sample = noise.sample()
    #     return self.generative_model(z, noise_sample)

    # def prior(self):
    #     return self.latent_prior_dis

    # def loss_function(self, x, recon_x, mu, log_var):
    #     recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    #     KL_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    #     return {"reconstruction": recon_loss, "kl_divergence": KL_divergence}

    # def loss_function(self, x, recon_x_samples, z_samples, mu, log_var):
    #     recon_x_samples = recon_x_samples.view(recon_x_samples.size(0), *x.size())
    #     recon_loss = F.mse_loss(recon_x_samples, x.expand_as(recon_x_samples), reduction='sum') / x.size(
    #         0) / recon_x_samples.size(0)
    #     KL_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
    #     return {"reconstruction": recon_loss, "kl_divergence": KL_divergence}

    def loss_function(self, x, recon_x_samples, z_samples, encoder_params):
        mu, log_var = encoder_params
        sigma = self.decoder.sigma
        R = recon_x_samples.size(0)

        recon_loss = (recon_x_samples - x.unsqueeze(0).expand_as(recon_x_samples)).pow(2)
        recon_loss = recon_loss.sum(dim=-1).mean() / 2 / sigma ** 2

        # recon_loss = recon_loss.sum(dim=-1).mean() \
        #     if recon_x_samples.size() == x.size() \
        #     else recon_loss.sum(dim=2).mean()

        tilde_z = torch.log(z_samples[:, :, :-1] / z_samples[:, :, -1:]) - mu.unsqueeze(0)
        sigma_diag_inv = torch.diag_embed(1.0 / torch.exp(0.5 * log_var)).unsqueeze(0).expand(R, -1, -1, -1)
        h_z = torch.sum(tilde_z.unsqueeze(-1).transpose(-1, -2) @ sigma_diag_inv @ tilde_z.unsqueeze(-1), dim=-1).mean() / 2
        h_z += log_var[:, :-1].sum(dim=-1).mean() / 2 + torch.log(z_samples).sum(dim=-1).mean()

        return {"reconstruction": recon_loss, "entropy": -h_z}

    # def loss_function(self, x, recon_x_samples, z_samples, encoder_params, decoder_params):
    #     mu, log_var = encoder_params
    #     sigma = decoder_params
    #     R = recon_x_samples.size(0)
    #
    #     # Handle reconstruction loss with proper dimension handling
    #     if recon_x_samples.size() != x.size():
    #         recon_loss = (recon_x_samples - x).pow(2).sum(dim=2).mean(dim=1).mean(dim=0) / sigma ** 2
    #     else:
    #         recon_loss = (recon_x_samples - x).pow(2).sum(dim=-1).mean(dim=0) / sigma ** 2
    #
    #     # Calculate KL divergence
    #     tilde_z = torch.log(z_samples[:, :, :-1] / z_samples[:, :, -1:]) - mu.unsqueeze(0)
    #     sigma_diag_inv = torch.diag_embed(1.0 / torch.exp(0.5 * log_var)).unsqueeze(0).expand(R, -1, -1, -1)
    #     h_z = 0.5 * torch.sum(tilde_z.unsqueeze(-1).transpose(-1, -2) @ sigma_diag_inv @ tilde_z.unsqueeze(-1))
    #
    #     # Add the constant term from the reference code
    #     h_z += 0.5 * log_var[:, :-1].sum() + torch.log(z_samples).sum()
    #     h_z += mu.shape[1] * torch.log(torch.tensor(2 * np.pi))
    #
    #     # Final loss (negative ELBO)
    #     return {"recon_loss": (recon_loss / 2), "h_z": - (h_z / 2)}

    def metric(self, x, z, x_recon_samples, z_samples, posterior_params):
        A_hat = self.decoder.linear_mixing.matrix
        A0 = self.data_model.dataset.linear_mixing

        min_mse = float('inf')
        perms = itertools.permutations(range(A0.shape[0]))

        for perm in perms:
            A_hat_permuted = A_hat[list(perm), :]
            mse = torch.mean(torch.sum((A0 - A_hat_permuted) ** 2, dim=1))
            if mse < min_mse:
                min_mse = mse

        mse_dB = 10 * torch.log10(min_mse)

        # W = torch.linalg.pinv(A0) @ A_hat
        # print(A0 @ torch.linalg.pinv(A0))
        # print(A0 @ W, A_hat)

        # print(torch.linalg.det(A0), torch.linalg.det(A_hat))
        Rr = torch.rand(self.latent_dim, self.latent_dim)
        R = torch.linalg.pinv(A0) @ A_hat
        # R = A_hat @ A_hat.T
        # print(W.T @ A0.T - A_hat.T)
        # z_true = (A0 @ z.T).T
        # z_recon = torch.linalg.pinv(A0) @ (A_hat @ z.T)
        z_recon = (R @ z.T).T
        z_reconr = (Rr @ z.T).T

        ssd = subspace_distance(z, z_recon)
        ssdr = subspace_distance(z, z_reconr)
        # print(ssd, ssdr)
        # print(R)
        # print(torch.linalg.det(R))
        sys.exit()
        # ssd = subspace_distance(z @ A0.T, z @ A0.T @ R.T)

        # import numpy as np
        #
        # t = 100
        # n = 2
        # non_linearity = 1e-15
        #
        # S = np.random.rand(t, n)
        # W = np.random.rand(n, n)
        #
        # S = torch.tensor(S)
        # W = torch.tensor(W)
        #
        # U = S @ W + non_linearity * torch.randn(*S.shape)
        #
        # print(subspace_distance(S, U))

        return {"mse_A_dB": mse_dB, "ssd": ssd}


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = latent_dim

        hidden_dims = list(hidden_layers.values())

        self.mean_layers = nn.ModuleList()
        self.var_layers = nn.ModuleList()
        self.mean_bn_layers = nn.ModuleList()
        self.var_bn_layers = nn.ModuleList()

        prev_dim = input_dim
        for h_dim in hidden_dims:
            self.mean_layers.append(nn.Linear(prev_dim, h_dim))
            self.var_layers.append(nn.Linear(prev_dim, h_dim))

            self.mean_bn_layers.append(nn.BatchNorm1d(h_dim))
            self.var_bn_layers.append(nn.BatchNorm1d(h_dim))

            prev_dim = h_dim

        self.fc_mean = nn.Linear(prev_dim, latent_dim - 1)
        self.fc_var = nn.Linear(prev_dim, latent_dim - 1)

    def forward(self, x):
        h_mean = x
        for fc, bn in zip(self.mean_layers, self.mean_bn_layers):
            h_mean = F.relu(bn(fc(h_mean)))
        mean = self.fc_mean(h_mean)

        h_var = x
        for fc, bn in zip(self.var_layers, self.var_bn_layers):
            h_var = F.relu(bn(fc(h_var)))
        log_var = self.fc_var(h_var)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_layers, activation=None, sigma=None):
        super(Decoder, self).__init__()
        self.sigma = sigma

        self.lin_transform = LinearPositive(latent_dim, output_dim)
        self.nonlinear_transform = NonlinearTransform(output_dim, hidden_layers, activation)

    def forward(self, z):
        y = self.lin_transform(z)
        x = self.nonlinear_transform(y)
        return x
