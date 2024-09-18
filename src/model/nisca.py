import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helpers.network import ComponentWiseNonlinear, LinearPositive
from torch.distributions import Dirichlet, LogNormal, Normal
from src.modules.vae_module import VAE


class Model(VAE):
    def __init__(self, encoder=None, decoder=None, mc_samples=1, lr=None, metrics=None, monitor=None, config=None, data_config=None):
        super().__init__(encoder=encoder, decoder=decoder, lr=lr)

        self.observed_dim = encoder.input_dim
        self.latent_dim = encoder.output_dim

        self.latent_prior_distribution = Dirichlet  # (torch.ones(self.latent_dim))
        self.noise_distribution = Normal
        self.variational_posterior_distribution = LogNormal
        self.mc_samples = mc_samples

        self.metrics = metrics
        self.monitor = monitor

    def reparameterize(self, params):
        mean, log_var = params
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(self.mc_samples, *std.shape, device=std.device)
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        samples = torch.cat((samples, torch.zeros(samples.shape[0], samples.shape[1], 1, device=std.device)), dim=2)
        z = F.softmax(samples, dim=-1)
        return z

    def loss_function(self, x, model_ouptut):
        x_mc_sample, z_mc_sample, variational_parameters = model_ouptut
        mu, log_var = variational_parameters
        sigma = self.decoder.sigma

        R = x_mc_sample.size(0)

        recon_loss = (x_mc_sample - x.unsqueeze(0).expand_as(x_mc_sample)).pow(2)
        recon_loss = recon_loss.sum(dim=-1).mean() / 2 / sigma ** 2

        tilde_z = torch.log(z_mc_sample[:, :, :-1] / z_mc_sample[:, :, -1:]) - mu.unsqueeze(0)
        sigma_diag_inv = torch.diag_embed(1.0 / torch.exp(0.5 * log_var)).unsqueeze(0).expand(R, -1, -1, -1)
        h_z = torch.sum(tilde_z.unsqueeze(-1).transpose(-1, -2) @ sigma_diag_inv @ tilde_z.unsqueeze(-1),
                        dim=-1).mean() / 2
        h_z += log_var[:, :-1].sum(dim=-1).mean() / 2 + torch.log(z_mc_sample).sum(dim=-1).mean()

        return {"reconstruction": recon_loss, "entropy": -h_z}

    def metrics(self):
        z_subspace = self.metrics.subspace_distance(self.data_model.nonlinear_transform,
                                                    self.decoder.nonlinear_transform)
        h_r_square = self.metrics.residual_nonlinearity(self.data_model.nonlinear_transform,
                                                        self.decoder.nonlinear_transform)
        return {"z_subspace": z_subspace, "h_r_square": h_r_square}

    # def metrics(self, x, z, x_mc_sample, z_mc_sample, variational_parameters, decoder):
    #
    #     return self.metrics.compute_metrics(x, z, x_mc_sample, z_mc_sample, variational_parameters, decoder)

    # todo: create metric class and initialize on datamodule and pass into model class.
    #  method metric is then calling the metric class methodss
    # def metrics(self, x, z, x_mc_sample, z_mc_sample, variational_parameters=None):
    #
    #     # true_mixing_A = self.data_model.lin_transform.to(x.device)
    #     # model_mixing_A = self.decoder.lin_transform.matrix.to(x.device)
    #     #
    #     # true_nonlinearity = self.data_model.nonlinear_transform
    #     # model_nonlinearity = self.decoder.nonlinear_transform
    #     #
    #     # metrics = {}
    #     # for metric in self.metrics:
    #     #
    #     #     if metric == "A_MSE_dB":
    #     #         A_mse_dB = mse_matrix_db(true_mixing_A, model_mixing_A)
    #     #         metrics["A_MSE_dB"] = A_mse_dB
    #     #
    #     #     elif metric == "A_SAM":
    #     #         A_sam = spectral_angle_distance(true_mixing_A, model_mixing_A)
    #     #         metrics["A_SAM"] = A_sam
    #     #
    #     #     elif metric == "z_MSE_dB":
    #     #         z_mse_dB = mse_matrix_db(z, z_mc_sample.mean(dim=0))
    #     #         metrics["z_mse_dB"] = z_mse_dB
    #     #
    #     #     elif metric == "z_subspace":
    #     #         z_subspace = subspace_distance(z, z_mc_sample.mean(dim=0))
    #     #         metrics["z_subspace"] = z_subspace
    #     #
    #     #     elif metric == "h_R-squared":
    #     #         h_rsquared = residual_nonlinearity(
    #     #             z @ true_mixing_A.T,
    #     #             true_nonlinearity,
    #     #             model_nonlinearity,
    #     #             show_plot=False
    #     #         ).rsquared.mean()
    #     #         metrics["h_R-squared"] = h_rsquared
    #     # metrics = self.metrics_module.compute_metrics(x, z, x_mc_sample, z_mc_sample, variational_parameters, decoder=self.decoder)
    #     self.metric_a_mse_db.update()
    #     self.metric_a_sam.update()
    #     self.metric_z_subspace.update()
    #
    #     # wandb.define_metric(self.monitor, summary="max")
    #
    #     metrics = {
    #         "a_mse_db": self.a_mse_metric.compute(),
    #         "a_sam": self.spectral_angle_metric.compute(),
    #         "subspace_distance": self.subspace_distance_metric.compute()
    #     }
    #     return metrics

    # def metric(self, x, z, x_mc_sample, z_mc_sample, variational_parameters=None):
    #
    #     true_mixing_A = self.data_model.dataset.lin_transform.to(x.device)
    #     model_mixing_A = self.decoder.lin_transform.matrix.to(x.device)
    #
    #     true_nonlinearity = self.data_model.dataset.nonlinear_transform
    #     model_nonlinearity = self.decoder.nonlinear_transform
    #
    #     A_mse_dB = mse_matrix_db(true_mixing_A, model_mixing_A)
    #     A_sam = spectral_angle_distance(true_mixing_A, model_mixing_A)
    #
    #     z_mse_dB = mse_matrix_db(z, z_mc_sample.mean(dim=0))
    #     z_subspace = subspace_distance(z, z_mc_sample.mean(dim=0))
    #     h_rsquared = self.residual_nonlinearity(
    #         z @ true_mixing_A.T,
    #         true_nonlinearity,
    #         model_nonlinearity,
    #         show_plot=False
    #     ).rsquared.mean()
    #
    #     wandb.define_metric(self.config.train.monitor, summary="max")
    #     return {
    #         "A_mse_dB": A_mse_dB,
    #         # "A_SAM": A_sam,
    #         # "z_mse_dB": z_mse_dB,
    #         # "z_subspace": z_subspace,
    #         # "h_rsquared": h_rsquared
    #     }

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
    def __init__(self, latent_dim, output_dim,
                 hidden_layers, activation, output_activation, mixing_initialization, weight_initialization):
        super(Decoder, self).__init__()

        self.linear_mixture = LinearPositive(
            torch.ones(output_dim, latent_dim), mixing_initialization
        )
        self.nonlinear_transform = ComponentWiseNonlinear(
            output_dim, hidden_layers, activation, output_activation, weight_initialization
        )
        # todo: no need for a class, compose from constructors here

    def forward(self, z):
        y = self.linear_mixture(z)
        x = self.nonlinear_transform(y)
        return x

# fixme: clean up and test nisca model, initialize on top of vasca super. just modify the metric and decoder
#  neural network output is horizontal!!! (nearly constant)  something is wrong
