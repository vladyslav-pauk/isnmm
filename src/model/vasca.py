import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.modules.network import LinearPositive, FCNConstructor
from src.modules.vae_module import VAEModule
import src.modules.metric as metric


class Model(VAEModule):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, train_config=None):
        super().__init__(encoder, decoder, **train_config)

        self.ground_truth = ground_truth_model

        self.metrics = torchmetrics.MetricCollection({
            'mixture_mse_db': metric.MatrixMse(),
            # 'mixture_sam': metric.SpectralAngle(),
            # 'mixture_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange(),
            # 'z_subspace': metric.SubspaceDistance()
        })
        self.metrics.eval()

    def reparameterize(self, variational_parameters, mc_samples):
        mean, log_var = variational_parameters
        std = torch.exp(0.5 * log_var)

        eps = torch.randn(mc_samples, *std.shape)

        z_samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        z_samples = torch.cat((z_samples, torch.zeros(*z_samples.size()[:2], 1)), dim=2)
        z_samples = F.softmax(z_samples, dim=-1)
        return z_samples

    # def loss_function(self, data, model_output):
    #     data_rec_mc_sample, latent_mc_sample, variational_parameters = model_output
    #     # print(variational_parameters)
    #     recon_loss = self.reconstruction(data, data_rec_mc_sample)
    #     neg_entropy_z = - self.entropy(latent_mc_sample, variational_parameters)
    #     kl_posterior_prior = neg_entropy_z - torch.lgamma(torch.tensor(latent_mc_sample.size(-1)))
    #     # fixme: neg_e is not training without rec_loss (why? is already initiated with the largest possible?)
    #     # print(recon_loss, kl_posterior_prior)
    #     return {"reconstruction": recon_loss, "kl_posterior_prior": kl_posterior_prior}
    #     # fixme: check all formula with the paper and thesis manuscript once again
    #
    # def reconstruction(self, data, data_rec_mc_sample):
    #     N = data.size(-1)
    #     sigma = self.ground_truth.data_model.sigma
    #     mse_loss = F.mse_loss(
    #         data_rec_mc_sample, data.expand_as(data_rec_mc_sample), reduction='mean'
    #     )
    #     recon_loss = mse_loss / (2 * sigma ** 2) * N
    #     recon_loss += N / 2 * torch.log(torch.tensor(2 * torch.pi)) # 0.92 * N
    #     return recon_loss
    #
    # def entropy(self, z_mc_sample, variational_parameters):
    #     M = z_mc_sample.size(-1)
    #     mu, log_var = variational_parameters
    #
    #     z_last = z_mc_sample[:, :, -1:]
    #     tilde_z = torch.log(z_mc_sample[:, :, :-1] / z_last) - mu
    #
    #     sigma_diag_inv = torch.exp(-0.5 * log_var)
    #
    #     log_2pi = torch.log(torch.tensor(2 * torch.pi))
    #     h_z = 0.5 * (M - 1) * log_2pi
    #     h_z += (tilde_z ** 2 * sigma_diag_inv.unsqueeze(0)).sum(dim=-1).mean() / 2
    #     h_z += log_var[:, :-1].sum() / (2 * M)
    #     h_z += torch.log(z_mc_sample).sum(dim=-1).mean()
    #     return h_z

    def loss_function(self, data, model_output):
        x_mc_sample, z_mc_sample, variational_parameters = model_output
        x = data
        mu, log_var = variational_parameters
        sigma = self.ground_truth.data_model.sigma

        R = x_mc_sample.size(0)

        recon_loss = (x_mc_sample - x.unsqueeze(0).expand_as(x_mc_sample)).pow(2)
        recon_loss = recon_loss.sum(dim=-1).mean() / 2 / sigma ** 2

        tilde_z = torch.log(z_mc_sample[:, :, :-1] / z_mc_sample[:, :, -1:]) - mu.unsqueeze(0)
        sigma_diag_inv = torch.diag_embed(1.0 / torch.exp(0.5 * log_var)).unsqueeze(0).expand(R, -1, -1, -1)
        h_z = torch.sum(tilde_z.unsqueeze(-1).transpose(-1, -2) @ sigma_diag_inv @ tilde_z.unsqueeze(-1),
                        dim=-1).mean() / 2
        h_z += log_var[:, :-1].sum(dim=-1).mean() / 2 + torch.log(z_mc_sample).sum(dim=-1).mean()

        return {"reconstruction": recon_loss, "entropy": -h_z}

    def update_metrics(self, data, model_output, labels):
        self.metrics['mixture_mse_db'].update(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        # self.metrics['mixture_sam'].update(
        #     self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        # )
        # self.metrics['mixture_volume'].update(
        #     self.decoder.linear_mixture.matrix
        # )
        self.metrics['mixture_matrix_change'].update(
            self.decoder.linear_mixture.matrix
        )
        # self.metrics['z_subspace'].update(
        #     labels[0], model_output[1].mean(dim=0)
        # )

    # def inference_model(self, observed):
    #     mean, log_var = self.encoder(observed)
    #     covariance = log_var.exp().diag_embed()
    #     distribution = self.variational_posterior_distribution(loc=mean, covariance_matrix=covariance)
    #     return distribution
    #
    # def likelihood(self, latent):
    #     decoded_latent = self.decoder(latent)
    #     noise_covariance = 0.1 * torch.eye(decoded_latent.shape[1])
    #     distribution = self.noise_distribution(loc=decoded_latent, covariance_matrix=noise_covariance)
    #     return distribution
    #
    # def prior(self):
    #     distribution = self.latent_prior_distribution(
    #         loc=torch.zeros(self.latent_dim), covariance_matrix=torch.eye(self.latent_dim)
    #     )
    #     return distribution
    #
    # def marginal_likelihood(self):
    #     pass


# class Encoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, config_encoder):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = latent_dim
#
#         hidden_dims = list(config_encoder["hidden_layers"].values())
#
#         self.mean_layers = nn.ModuleList()
#         self.var_layers = nn.ModuleList()
#         self.mean_bn_layers = nn.ModuleList()
#         self.var_bn_layers = nn.ModuleList()
#
#         prev_dim = input_dim
#         for h_dim in hidden_dims:
#             self.mean_layers.append(nn.Linear(prev_dim, h_dim))
#             self.var_layers.append(nn.Linear(prev_dim, h_dim))
#
#             self.mean_bn_layers.append(nn.BatchNorm1d(h_dim))
#             self.var_bn_layers.append(nn.BatchNorm1d(h_dim))
#
#             prev_dim = h_dim
#
#         self.fc_mean = nn.Linear(prev_dim, latent_dim - 1)
#         self.fc_var = nn.Linear(prev_dim, latent_dim - 1)
#
#     def forward(self, x):
#         h_mean = x
#         for fc, bn in zip(self.mean_layers, self.mean_bn_layers):
#             h_mean = F.relu(bn(fc(h_mean)))
#         mean = self.fc_mean(h_mean)
#
#         h_var = x
#         for fc, bn in zip(self.var_layers, self.var_bn_layers):
#             h_var = F.relu(bn(fc(h_var)))
#         log_var = self.fc_var(h_var)
#
#         return mean, log_var

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, config_encoder):
        super().__init__()

        self.mu_network = FCNConstructor(input_dim, latent_dim - 1, **config_encoder)
        self.log_var_network = FCNConstructor(input_dim, latent_dim - 1, **config_encoder)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, config_decoder):
        super(Decoder, self).__init__()
        self.linear_mixture = LinearPositive(torch.rand(output_dim, latent_dim), **config_decoder)

    def forward(self, z):
        x = self.linear_mixture(z)
        return x
