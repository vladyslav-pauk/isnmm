import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics

from src.modules.ae_module import AutoEncoderModule
import src.modules.metric as metric
from src.modules.metric import EvaluateMetric
from src.modules.optimizer.constrained_lagrange import ConstrainedLagrangeOptimizer
import src.modules.network as network


class Model(AutoEncoderModule):
    def __init__(self, ground_truth_model, encoder, decoder, model_config, optimizer_config):
        super().__init__(encoder, decoder)

        self.ground_truth = ground_truth_model
        self.optimizer_config = optimizer_config
        self.optimizer = None
        self.metrics = None
        self.log_monitor = None
        self.observed_dim = self.ground_truth.observed_dim

        self.setup_metrics()

    def loss_function(self, observed_batch, model_output, idxes):
        reconstructed_sample = model_output["reconstructed_sample"]
        latent_sample = model_output["latent_sample"]
        reconstruct_err = F.mse_loss(reconstructed_sample, observed_batch)

        regularization_loss = self.optimizer.compute_constraint_errors(latent_sample, idxes, observed_batch)
        loss = {"reconstruction": reconstruct_err}
        loss.update(regularization_loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.optimizer.update_buffers(batch["idxes"], self(batch["data"])["latent_sample"])

    def configure_optimizers(self):
        self.optimizer = ConstrainedLagrangeOptimizer(
            params=list(self.parameters()),
            lr=self.optimizer_config['lr']["encoder"],
            rho=self.optimizer_config['rho'],
            inner_iters=self.optimizer_config['inner_iters'],
            n_sample=self.ground_truth.dataset_size,
            observed_dim=self.observed_dim,
            constraint_fn=lambda F: torch.sum(F, dim=1) - 1.0
        )
        return self.optimizer

    def setup_metrics(self):
        subspace_distance_metric = metric.SubspaceDistance()
        evaluate_metric = EvaluateMetric()
        constraint_error = metric.ConstraintError(lambda F: torch.sum(F, dim=1) - 1.0)

        self.metrics = torchmetrics.MetricCollection({
            'subspace_distance': subspace_distance_metric,
            'h_r_square': metric.ResidualNonlinearity(),
            'evaluate_metric': evaluate_metric,
            'constraint': constraint_error
        })
        self.metrics.eval()
        self.log_monitor = {
            "monitor": "validation_loss",
            "mode": "min"
        }

    def update_metrics(self, data, model_output, labels, idxes):
        reconstructed_sample = model_output["reconstructed_sample"]
        latent_sample = model_output["latent_sample"]

        true_latent_sample = labels["latent_sample"]
        linearly_mixed_sample = labels["linearly_mixed_sample"]
        latent_sample_qr = labels["latent_sample_qr"]

        self.metrics['evaluate_metric'].update(data, linearly_mixed_sample, latent_sample)
        self.metrics['subspace_distance'].update(idxes, self.optimizer.latent_sample_buffer, latent_sample_qr)
        self.metrics['constraint'].update(idxes, self.optimizer.latent_sample_buffer)
        self.metrics['h_r_square'].update(reconstructed_sample, linearly_mixed_sample, data)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.network = None

    def construct(self, latent_dim, observed_dim):
        self.network = self.constructor(observed_dim, latent_dim, **self.config)

    def forward(self, x):
        x = self.network.forward(x)
        return x, None


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = None
        self.nonlinear_transform = None

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = self.constructor(latent_dim, observed_dim, **self.config)

    def forward(self, x):
        x = self.linear_mixture(x)
        x = self.nonlinear_transform(x)
        return x


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
#         y = self.linear_mixture(z)
#         x = self.nonlinear_transform(y)
#         return x


# fixme: check neural network architecture, implement the correct module
# fixme: cnae unequal dimensions
# fixme: experiment cnae on noisy data
# fixme: clean up and readme
# fixme: make some runs and organize wandb to show Xiao
# fixme: train cnae with reparametrization
# fixme: train nisca with constrained optimization
