import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

from src.modules.network import LinearPositive, FCNConstructor
from src.modules.vae_module import VAEModule
import src.modules.metric as metric


class Model(VAEModule):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, train_config=None):
        super().__init__(encoder, decoder, **train_config)

        self.ground_truth = ground_truth_model

        self.metrics = torchmetrics.MetricCollection({
            'mixture_mse_db': metric.MatrixMse(),
            'mixture_sam': metric.SpectralAngle(),
            'mixture_log_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange(),
            'z_subspace': metric.SubspaceDistance()
        })
        self.metrics.eval()

    # def on_before_backward(self, loss: Tensor) -> None:
    #     print("Encoder", self.encoder.mu_network.hidden_layers[0][0].weight)
    #     print("Encoder", self.encoder.mu_network.hidden_layers[0][0].weight.grad)
    #
    # def on_after_backward(self) -> None:
    #     print("EncoderA", self.encoder.mu_network.hidden_layers[0][0].weight)
    #     print("EncoderA", self.encoder.mu_network.hidden_layers[0][0].weight.grad)

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'inf or nan gradient, skipping update.')
            self.zero_grad()

    def reparameterize(self, variational_parameters, mc_samples):
        mean, log_var = variational_parameters
        std = torch.exp(0.5 * log_var)

        eps = torch.randn(mc_samples, *std.shape)

        z_samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        z_samples = torch.cat((z_samples, torch.zeros(*z_samples.size()[:2], 1)), dim=2)
        z_samples = F.softmax(z_samples, dim=-1)
        return z_samples

    def loss_function(self, data, model_output):
        data_rec_mc_sample, latent_mc_sample, variational_parameters = model_output

        recon_loss = self._reconstruction(data, data_rec_mc_sample)
        neg_entropy_z = - self._entropy(latent_mc_sample, variational_parameters)
        kl_posterior_prior = neg_entropy_z - torch.lgamma(torch.tensor(latent_mc_sample.size(-1)))
        return {"reconstruction": recon_loss, "kl_posterior_prior": kl_posterior_prior}

        # todo: neg_e is not training without rec_loss, check constants,
        #  kl should be always positive (check sign of gamma(N))

    def _reconstruction(self, data, data_rec_mc_sample):
        N = data.size(-1)
        sigma = self.ground_truth.data_model.sigma
        mse_loss = F.mse_loss(
            data_rec_mc_sample, data.expand_as(data_rec_mc_sample), reduction='mean'
        )
        recon_loss = mse_loss / (2 * sigma ** 2) * N
        recon_loss += N / 2 * torch.log(torch.tensor(2 * torch.pi))
        return recon_loss

    def _entropy(self, z_mc_sample, variational_parameters):
        mu, log_var = variational_parameters

        z_last = z_mc_sample[:, :, -1:]
        tilde_z = torch.log(z_mc_sample[:, :, :-1] / z_last) - mu
        sigma_diag_inv = torch.exp(-0.5 * log_var)

        log_2pi = torch.log(torch.tensor(2 * torch.pi))
        h_z = 0.5 * (z_mc_sample.size(-1) - 1) * log_2pi
        h_z += (tilde_z ** 2 * sigma_diag_inv.unsqueeze(0)).sum(dim=-1).mean() / 2
        h_z += log_var[:, :-1].sum(dim=-1).mean() / 2
        h_z += torch.log(z_mc_sample).sum(dim=-1).mean()
        return h_z

    def update_metrics(self, data, model_output, labels):
        self.metrics['mixture_mse_db'].update(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_sam'].update(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_log_volume'].update(
            self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_matrix_change'].update(
            self.decoder.linear_mixture.matrix
        )
        self.metrics['z_subspace'].update(
            labels[0], model_output[1].mean(dim=0)
        )

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


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, config_encoder):
        super().__init__()

        self.mu_network = FCNConstructor(input_dim, latent_dim - 1, **config_encoder)
        self.log_var_network = FCNConstructor(input_dim, latent_dim - 1, **config_encoder)

    def forward(self, x):
        # print(x)
        # print()
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        # print("mu, logvar", mu, log_var)
        # print(self.mu_network.hidden_layers[0][0].weight)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, config_decoder):
        super(Decoder, self).__init__()
        self.linear_mixture = LinearPositive(torch.rand(output_dim, latent_dim), **config_decoder)

    def forward(self, z):
        x = self.linear_mixture(z)
        return x
