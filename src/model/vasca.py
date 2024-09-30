import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torch.optim as optim

import src.modules.network as network
from src.modules.ae_module import AutoEncoderModule
import src.modules.metric as metric


class Model(AutoEncoderModule):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, model_config=None, optimizer_config=None):
        super().__init__(encoder, decoder)

        self.ground_truth = ground_truth_model
        self.optimizer_config = optimizer_config
        self.mc_samples = model_config["mc_samples"]

        self.metrics = torchmetrics.MetricCollection({
            'mixture_mse_db': metric.MatrixMse(),
            'mixture_sam': metric.SpectralAngle(),
            'mixture_log_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange(),
            # 'subspace_distance': metric.SubspaceDistance()
        })
        self.metrics.eval()
        self.log_monitor = {
            "monitor": "mixture_mse_db",
            "mode": "min"
        }

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

    def reparameterize(self, variational_parameters):
        mean, log_var = variational_parameters
        std = torch.exp(0.5 * log_var)

        eps = torch.randn(self.mc_samples, *std.shape)

        z_samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        z_samples = torch.cat((z_samples, torch.zeros(*z_samples.size()[:2], 1)), dim=2)
        z_samples = F.softmax(z_samples, dim=-1)
        return z_samples

    def loss_function(self, data, model_output, idxes):
        data_rec_mc_sample, latent_mc_sample, variational_parameters = model_output

        recon_loss = self._reconstruction(data, data_rec_mc_sample)
        neg_entropy_z = - self._entropy(latent_mc_sample, variational_parameters)
        kl_posterior_prior = neg_entropy_z - torch.lgamma(torch.tensor(latent_mc_sample.size(-1)))
        return {"reconstruction": recon_loss, "kl_posterior_prior": kl_posterior_prior}

        # todo: neg_e is not training without rec_loss, check constants,
        #  kl should be always positive (check sign of gamma(N))

    def _reconstruction(self, data, data_rec_mc_sample):
        N = data.size(-1)
        sigma = self.ground_truth.sigma
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

    def update_metrics(self, data, model_output, labels, idxes):
        # reconstructed_sample, latent_sample, _ = model_output
        # true_latent_sample, linearly_mixed_sample, _ = labels

        self.metrics['mixture_mse_db'].update(
            self.ground_truth.linear_mixture, self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_sam'].update(
            self.ground_truth.linear_mixture, self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_log_volume'].update(
            self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_matrix_change'].update(
            self.decoder.linear_mixture.matrix
        )

        # self.metrics['subspace_distance'].update(
        #     idxes, reconstructed_sample.squeeze(0), true_latent_sample
        # )

    def configure_optimizers(self):
        lr = self.optimizer_config["lr"]
        lr_th = lr["th"]
        lr_ph = lr["ph"]
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr_ph},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr_th}
        ], **self.optimizer_config["params"])
        return optimizer

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

    # def on_before_backward(self, loss: Tensor) -> None:
    #     print("Encoder", self.encoder.mu_network.hidden_layers[0][0].weight)
    #     print("Encoder", self.encoder.mu_network.hidden_layers[0][0].weight.grad)
    #
    # def on_after_backward(self) -> None:
    #     print("EncoderA", self.encoder.mu_network.hidden_layers[0][0].weight)
    #     print("EncoderA", self.encoder.mu_network.hidden_layers[0][0].weight.grad)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])

        self.mu_network = None
        self.log_var_network = None

    def construct(self, latent_dim, observed_dim):
        self.mu_network = self.constructor(observed_dim, latent_dim - 1, **self.config)
        self.log_var_network = self.constructor(observed_dim, latent_dim - 1, **self.config)

    def forward(self, x):
        mu = self.mu_network.forward(x)
        log_var = self.log_var_network.forward(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.linear_mixture = None

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = network.LinearPositive(torch.rand(observed_dim, latent_dim), **self.config)
        return self

    def forward(self, z):
        x = self.linear_mixture(z)
        return x
