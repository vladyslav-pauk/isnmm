from typing import Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
import torchmetrics
from src.helpers.util import evaluate

from src.modules.ae_module import AutoEncoderModule


class Model(AutoEncoderModule):
    def __init__(self, ground_truth_model, encoder, decoder, model_config, optimizer_config):
        super().__init__(encoder, decoder)

        self.ground_truth = ground_truth_model

        # Initialize parameters and buffers
        n_sample = ground_truth_model.dataset_size
        input_dim = ground_truth_model.observed_dim
        self.mult = nn.Parameter(torch.randn(n_sample), requires_grad=False)
        self.register_buffer('F_buffer', torch.zeros((n_sample, input_dim)))
        self.register_buffer('count_buffer', torch.zeros(n_sample, dtype=torch.int32))

        # Set training parameters from config
        self.rho = optimizer_config.get('rho', 1e2)
        self.lr = optimizer_config.get('lr', 1e-3)
        self.inner_iters = optimizer_config.get('inner_iters', 1)
        self.model_file_name = model_config.get('model_file_name', 'model.ckpt')

        self.linear_mixture = ground_truth_model.linear_mixture

        # Metrics and constraint tracking
        self.metrics = torchmetrics.MetricCollection({'subspace_distance': torchmetrics.MeanMetric()})
        self.best_constraint_val = float('inf')
        self.subspace_dist_arr = []

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            train_loader = self.trainer.datamodule.train_dataloader()
            sample_batch = next(iter(train_loader))
            data_sample = sample_batch
            observed_dim = data_sample[0].shape[1]
            if self.latent_dim is None:
                self.latent_dim = data_sample[1][0].shape[1]

            self.encoder.construct(self.latent_dim, observed_dim)
            self.decoder.construct(self.latent_dim, observed_dim)

    def forward(self, x):
        fx, _ = self.encoder(x)
        qfx = self.decoder(fx)
        return fx, qfx

    def training_step(self, batch, batch_idx):
        data, idxes = batch
        fx, qfx = self(data)

        # Calculate loss
        total_loss, _, _, _ = self.loss_function(fx, qfx, data, idxes)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update buffers and multipliers
        self.F_buffer[idxes] = fx.detach()
        self.count_buffer[idxes] += 1
        if (self.global_step + 1) % self.inner_iters == 0:
            self.update_multipliers()

        return total_loss

    # def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
    #     self.compute_subspace_distance(F)


    def test_step(self, batch: Any, batch_idx: int):
        data, _ = batch
        data = data.float()

        # Forward pass and store predictions
        fx, _ = self(data)
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
        self.test_predictions.append(fx)

    def on_test_epoch_end(self):
        # Concatenate test predictions and evaluate
        F = torch.cat(self.test_predictions, dim=0)
        evaluate(F, self.trainer.datamodule.test_dataloader(), self.ground_truth.linear_mixture, self.ground_truth.observed_data)
        self.test_predictions = []

    def loss_function(self, fx, qfx, x, idxes):
        # Compute different components of the loss
        tmp = torch.sum(fx, dim=1) - 1.0
        mult = self.mult[idxes]
        reconstruct_err = F.mse_loss(qfx, x)
        feasible_err = torch.dot(mult, tmp) / x.shape[0]
        augmented_err = torch.norm(tmp) ** 2 / x.shape[0]

        # Total loss
        total_loss = reconstruct_err + feasible_err + (self.rho / 2) * augmented_err
        return total_loss, reconstruct_err, feasible_err, augmented_err

    def update_multipliers(self):
        idxes = self.count_buffer.nonzero(as_tuple=True)[0]
        F = self.F_buffer[idxes]
        diff = torch.sum(F, dim=1) - 1.0
        self.mult[idxes] += self.rho * diff

        squared_diff = torch.norm(diff) ** 2
        self.log('squared_diff', squared_diff, prog_bar=True)

        # Save model if constraint improves
        if squared_diff < self.best_constraint_val:
            self.best_constraint_val = squared_diff
            self.trainer.save_checkpoint(self.model_file_name)

        # Reset buffers
        self.F_buffer[idxes] = 0.0
        self.count_buffer[idxes] = 0

    def compute_subspace_distance(self, F):
        # Compute and log subspace distance
        F = self.buffer_F[self.idxs]
        F_cpu = F.to('cpu').detach().numpy()
        qf, _ = torch.linalg.qr(F_cpu)
        import scipy
        subspace_dist = torch.sin(scipy.linalg.subspace_angles(self.qs, qf)[0])
        self.subspace_dist_arr.append(subspace_dist.item())
        self.log('subspace_distance', subspace_dist.item(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr["encoder"])


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, input_dim, output_dim):
        layers = []
        in_channels = input_dim
        hidden_sizes = self.config.get('hidden_layers', [128, 128, 128])

        for h in hidden_sizes:
            layers.append(nn.Conv1d(in_channels, h * input_dim, kernel_size=1, groups=input_dim))
            layers.append(nn.ReLU())
            in_channels = h * input_dim

        layers.append(nn.Conv1d(in_channels, output_dim, kernel_size=1, groups=input_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float().unsqueeze(-1)
        x = self.e_net(x)
        return x.squeeze(-1), None


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def construct(self, input_dim, output_dim):
        layers = []
        in_channels = input_dim
        hidden_sizes = self.config.get('hidden_layers', [128, 128, 128])

        for h in hidden_sizes:
            layers.append(nn.Conv1d(in_channels, h * input_dim, kernel_size=1, groups=input_dim))
            layers.append(nn.ReLU())
            in_channels = h * input_dim

        layers.append(nn.Conv1d(in_channels, output_dim, kernel_size=1, groups=input_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.d_net(x)
        return x.squeeze(-1)


# class Decoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.constructor = getattr(network, config["constructor"])
#
#     def construct(self, latent_dim, observed_dim):
#         self.linear_mixture = network.LinearPositive(
#             torch.rand(observed_dim, latent_dim), **self.config
#         )
#
#         self.nonlinearity = nn.ModuleList([self.constructor(
#             input_dim=1, output_dim=1, **self.config
#         ) for _ in range(observed_dim)])
#
#     def nonlinear_transform(self, x):
#         x = torch.cat([
#             self.nonlinearity[i](x[..., i:i + 1].view(-1, 1)).view_as(x[..., i:i + 1])
#             for i in range(x.shape[-1])
#         ], dim=-1)
#         return x
#
#     def forward(self, z):
#         x = self.linear_mixture(z)
#         x = self.nonlinear_transform(x)
#         return x