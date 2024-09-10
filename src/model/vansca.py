import sys
import itertools
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Dirichlet, LogNormal, Normal

from src.training_module import VAE
from src.utils import subspace_distance, plot_components, LineFitter, plot_components_wrapped, mse_matrix_db


class Model(VAE):
    def __init__(self, encoder=None, decoder=None, data_model=None, mc_samples=1, lr=None):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr)

        self.observed_dim = encoder.input_dim
        self.latent_dim = encoder.output_dim

        self.latent_prior_distribution = Dirichlet(torch.ones(self.latent_dim))
        self.noise_distribution = Normal
        self.variational_posterior_distribution = LogNormal

        self.data_model = data_model

        self.mc_samples = mc_samples

    def reparameterize(self, params):
        mean, log_var = params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(self.mc_samples, *std.shape, device=std.device)
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        samples = torch.cat((samples, torch.zeros(samples.shape[0], samples.shape[1], 1, device=std.device)), dim=2)
        z = F.softmax(samples, dim=-1)
        return z

    def loss_function(self, x, mc_samples, variational_parameters):
        mu, log_var = variational_parameters
        sigma = self.decoder.sigma

        recon_x_samples, z_samples = mc_samples
        R = recon_x_samples.size(0)

        recon_loss = (recon_x_samples - x.unsqueeze(0).expand_as(recon_x_samples)).pow(2)
        recon_loss = recon_loss.sum(dim=-1).mean() / 2 / sigma ** 2

        tilde_z = torch.log(z_samples[:, :, :-1] / z_samples[:, :, -1:]) - mu.unsqueeze(0)
        sigma_diag_inv = torch.diag_embed(1.0 / torch.exp(0.5 * log_var)).unsqueeze(0).expand(R, -1, -1, -1)
        h_z = torch.sum(tilde_z.unsqueeze(-1).transpose(-1, -2) @ sigma_diag_inv @ tilde_z.unsqueeze(-1),
                        dim=-1).mean() / 2
        h_z += log_var[:, :-1].sum(dim=-1).mean() / 2 + torch.log(z_samples).sum(dim=-1).mean()

        return {"reconstruction": recon_loss, "entropy": -h_z}

    def metric(self, x, z, mc_sample, variational_parameters=None):

        # x_recon_samples, z_samples = mc_sample

        true_mixing_A = self.data_model.dataset.lin_transform.to(x.device)
        model_mixing_A = self.decoder.lin_transform.matrix.to(x.device)

        A_mse_dB = mse_matrix_db(true_mixing_A, model_mixing_A)

        true_nonlinearity = self.data_model.dataset.nonlinear_transform
        model_nonlinearity = self.decoder.nonlinear_transform

        residual_nonlinearity = lambda x: true_nonlinearity.inverse(model_nonlinearity(x))
        y_true = z @ true_mixing_A.T
        fitter = LineFitter()
        fitter.fit(residual_nonlinearity, y_true).unsqueeze(0)

        nonlinearity_plot = plot_components_wrapped(
            y_true,
            true_nonlinearity=true_nonlinearity,
            inferred_nonlinearity=model_nonlinearity,
        )
        residual_nonlinearity_plot = plot_components_wrapped(
            y_true,
            residual_nonlinearity=residual_nonlinearity,
            fitter=fitter,
            labels=fitter.rsquared
        )
        nonlinearity_plot.show()
        residual_nonlinearity_plot.show()
        # wandb.log({"nonlinearity": nonlinearity_plot, "residual_nonlinearity": residual_nonlinearity_plot})
        # nonlinearity_plot.close()
        residual_nonlinearity_plot.close()
        # fixme: neural network output is horizontal!!! (nearly constant)  something is wrong
        # sys.exit()

        wandb.define_metric("h_r-squared", summary="max")
        return {
            "h_r-squared": fitter.rsquared.mean(),
            # "z_subspace_distance": reconstruction_subspace_distance,
            # "A_mse_dB": A_mse_dB,
        }
        # wandb.define_metric("subspace_distance", summary="min")
        # r_squared = torch.tensor([0.0])

        # z_recon = (torch.linalg.pinv(true_mixing_A) @ residual_nonlinearity((model_mixing_A @ z.T).T).T).T
        # reconstruction_subspace_distance = subspace_distance(z, z_recon)
        # reconstruction_subspace_distance = torch.norm(z - z_recon, dim=-1).mean()

        # print(residual_nonlinearity(y_true), fitter(y_true))

        # nonlinearity_plot = plot_components(
        #     y_true,
        #     # true_nonlinearity=true_nonlinearity,
        #     # inferred_nonlinearity=model_nonlinearity,
        #     residual_nonlinearity=residual_nonlinearity,
        #     linear_fit=fitter
        # )
        # nonlinearity_plot.show()
        # nonlinearity_plot = plot_components(
        #     y_true,
        #     residual_nonlinearity=residual_nonlinearity,
        #     # Create a component-wise lambda for the linear fit
        #     linear_fit=lambda x: torch.cat(
        #         [fitter.slopes[i] * x[:, i:i + 1] + fitter.intercepts[i] for i in range(x.shape[-1])], dim=-1
        #     )
        # )

        # print(reconstruction_subspace_distance)

        # reconstruction_subspace_distance = torch.norm(z - z_recon, dim=-1).mean()

        # data_nonlinearity = lambda x: self.data_model.dataset.nonlinear_transform(x)
        # data_nonlinearity_i = lambda x: self.data_model.dataset.nonlinear_transform.inverse(x)
        # model_nonlinearity = lambda x: self.decoder.nonlinear_transform(x)

        # W = torch.linalg.pinv(A0) @ A_hat
        # print(A0 @ torch.linalg.pinv(A0))
        # print(A0 @ W, A_hat)

        # print(torch.linalg.det(A0), torch.linalg.det(A_hat))
        # Rr = torch.rand(self.latent_dim, self.latent_dim)
        # R = torch.linalg.pinv(A0) @ A_hat
        # R = A_hat @ A_hat.T
        # print(W.T @ A0.T - A_hat.T)
        # z_true = (A0 @ z.T).T
        # print((z @ A0.T).shape)
        # print(remainining_nonlinearity(A0 @ z[0]))
        # sys.exit()
        # print(model_nonlinearity((A_hat @ z.T).T).shape)

        # z_recon = (R @ z.T).T
        # z_reconr = (Rr @ z.T).T

        # print(ssd)

        # ssdr = subspace_distance(z, z_reconr)
        # print(ssd, ssdr)
        # print(R)
        # print(torch.linalg.det(R))

        # ssd = subspace_distance(z @ A0.T, z @ A0.T @ R.T)

# todo: test independently subspace distance and find similar probabilistic measure and use IS expectation.
