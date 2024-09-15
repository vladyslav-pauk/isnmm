import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.modules.network import ComponentWiseNonlinear, LinearPositive
from src.modules.vae_module import VAEModule
import src.modules.metric as metric


class Model(VAEModule):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, train_config=None):
        super().__init__(encoder, decoder, **train_config)

        self.observed_dim = encoder.input_dim
        self.latent_dim = encoder.output_dim

        self.ground_truth = ground_truth_model

        self.metrics = torchmetrics.MetricCollection({
            'mse_matrix_db': metric.MatrixMse(),
            'spectral_angle_distance': metric.SpectralAngle(),
            'subspace_distance': metric.SubspaceDistance()
        })
        # self.metrics.eval()

    # fixme: verify reparameterization 4, 15
    def reparameterize(self, params, mc_samples):
        mean, log_var = params
        std = torch.exp(0.5 * log_var)

        device = mean.device
        eps = torch.randn(mc_samples, *std.shape, device=device)
        samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        samples = torch.cat(
            (samples, torch.zeros(samples.shape[0], samples.shape[1], 1, device=device)),
            dim=2
        )
        z = F.softmax(samples, dim=-1)
        return z


    # def reparameterize(self, params, mc_samples):
    #
    #     mean, log_var = params
    #     std = torch.exp(0.5 * log_var)
    #
    #     eps = torch.randn(mc_samples, *std.shape, device=std.device)
    #     samples = mean.unsqueeze(0) + eps * std.unsqueeze(0)
    #     samples = torch.cat((samples, torch.zeros(samples.shape[0], samples.shape[1], 1, device=std.device)), dim=2)
    #
    #     z = F.softmax(samples, dim=-1)
    #
    #     return z

    # fixme: verify loss function 5, 11, 18, 19
    def loss_function(self, data, model_output):
        x_mc_sample, z_mc_sample, variational_parameters = model_output
        mu, log_var = variational_parameters
        sigma = self.ground_truth.data_model.sigma

        R = x_mc_sample.size(0)  # Number of MC samples

        # Reconstruction loss
        recon_loss = F.mse_loss(
            x_mc_sample, data.unsqueeze(0).expand_as(x_mc_sample), reduction='mean'
        ) / (2 * sigma ** 2)

        # KL Divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        return {"reconstruction": recon_loss, "kl_divergence": kl_div}

    # def loss_function(self, data, model_output):
    #     x_mc_sample, z_mc_sample, variational_parameters = model_output
    #     mu, log_var = variational_parameters
    #     sigma = self.ground_truth.sigma
    #
    #     R = x_mc_sample.size(0)
    #
    #     recon_loss = (x_mc_sample - data.unsqueeze(0).expand_as(x_mc_sample)).pow(2)
    #     recon_loss = recon_loss.sum(dim=-1).mean() / 2 / sigma ** 2
    #
    #     tilde_z = torch.log(z_mc_sample[:, :, :-1] / z_mc_sample[:, :, -1:]) - mu.unsqueeze(0)
    #     sigma_diag_inv = torch.diag_embed(1.0 / torch.exp(0.5 * log_var)).unsqueeze(0).expand(R, -1, -1, -1)
    #     h_z = torch.sum(tilde_z.unsqueeze(-1).transpose(-1, -2) @ sigma_diag_inv @ tilde_z.unsqueeze(-1),
    #                     dim=-1).mean() / 2
    #     h_z += log_var[:, :-1].sum(dim=-1).mean() / 2 + torch.log(z_mc_sample).sum(dim=-1).mean()
    #
    #     return {"reconstruction": recon_loss, "entropy": -h_z}

    # fixme: verify compute metrics 10
    # def compute_metrics(self, data, model_output, labels, print_results=False):
    #     outputs = {
    #         'decoder_matrix': self.decoder.lin_transform.matrix,
    #         'predicted_z': model_output[1].mean(dim=0)
    #     }
    #     targets = {
    #         'ground_truth_matrix': self.ground_truth.lin_transform.matrix,
    #         'ground_truth_z': labels[0]
    #     }
    #
    #     # Update metrics
    #     self.metrics['mse_matrix_db'].update(
    #         outputs['decoder_matrix'], targets['ground_truth_matrix']
    #     )
    #     self.metrics['spectral_angle_distance'].update(
    #         outputs['decoder_matrix'], targets['ground_truth_matrix']
    #     )
    #     self.metrics['subspace_distance'].update(
    #         outputs['predicted_z'], targets['ground_truth_z']
    #     )
    #
    #     metric_results = self.metrics.compute()
    #
    #     self.metrics.reset()
    #
    #     if print_results:
    #         print("True mixing matrix:\n", outputs['ground_truth_matrix'])
    #         print("Estimated mixing matrix:\n", outputs['decoder_matrix'])
    #
    #     return metric_results

    def compute_metrics(self, data, model_output, labels):
        # print(self.ground_truth.linear_mixing.matrix, self.decoder.linear_mixing.matrix)

        mse_a_db = self.metrics.mse_matrix_db(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        sam_a = self.metrics.spectral_angle_distance(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )

        ground_truth_z = labels[0]
        predicted_z = model_output[1].mean(dim=0)
        z_subspace = self.metrics.subspace_distance(
            ground_truth_z, predicted_z
        )

        # z_recon = (torch.linalg.pinv(true_mixing_A) @ residual_nonlinearity((model_mixing_A @ z.T).T).T).T

        return {"mse_a_db": mse_a_db, "sam_a": sam_a, "z_subspace": z_subspace}


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers, init_weights=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = latent_dim

        hidden_dims = list(hidden_layers.values())

        self.shared_layers = nn.ModuleList()
        self.shared_bn_layers = nn.ModuleList()

        prev_dim = input_dim
        for h_dim in hidden_dims:
            layer = nn.Linear(prev_dim, h_dim)
            if init_weights:
                getattr(nn.init, init_weights)(layer.weight)
                getattr(nn.init, init_weights)(layer.bias)
            self.shared_layers.append(layer)
            self.shared_bn_layers.append(nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        self.fc_mean = nn.Linear(prev_dim, latent_dim - 1)
        self.fc_var = nn.Linear(prev_dim, latent_dim - 1)

    def forward(self, x):
        h = x
        for fc, bn in zip(self.shared_layers, self.shared_bn_layers):
            h = F.relu(bn(fc(h)))

        mean = self.fc_mean(h)
        log_var = self.fc_var(h)
        return mean, log_var

    # todo: make a network class for encoder in networks

# class Encoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_layers):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = latent_dim
#
#         hidden_dims = list(hidden_layers.values())
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


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_layers, activation=None, init_weights=None):
        super(Decoder, self).__init__()

        self.linear_mixture = LinearPositive(torch.ones(output_dim, latent_dim), init_weights=init_weights)
        self.nonlinear_transform = ComponentWiseNonlinear(output_dim, hidden_layers, activation, init_weights=init_weights)

    def forward(self, z):
        y = self.linear_mixture(z)
        x = self.nonlinear_transform(y)
        return x

# todo: use in-place activation functions where possible, e.g., F.relu_(h).
