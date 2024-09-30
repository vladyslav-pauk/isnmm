import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics

from src.modules.ae_module import AutoEncoderModule
# from src.model.nisca import Decoder
from src.modules.network import CNN
import src.modules.metric as metric
from src.modules.metric import EvaluateMetric
from src.modules.optimizer.constrained_lagrange import ConstrainedLagrangeOptimizer


class Model(AutoEncoderModule):
    def __init__(self, ground_truth_model, encoder, decoder, model_config, optimizer_config):
        super().__init__(encoder, decoder)

        self.ground_truth = ground_truth_model
        self.optimizer_config = optimizer_config
        self.optimizer = None

        self.observed_dim = self.ground_truth.observed_dim

        self.setup_metrics()

    def loss_function(self, observed_batch, model_output, idxes):
        reconstructed_sample, latent_sample, _ = model_output
        reconstruct_err = F.mse_loss(reconstructed_sample, observed_batch)

        regularization_loss = self.optimizer.compute_constraint_errors(latent_sample, idxes, observed_batch)
        loss = {"reconstruction": reconstruct_err}
        loss.update(regularization_loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        data, labels, idxes = batch
        model_output = self(data)
        self.optimizer.update_buffers(idxes, model_output[1])

    def configure_optimizers(self):
        self.optimizer = ConstrainedLagrangeOptimizer(
            params=list(self.parameters()),
            lr=self.optimizer_config['lr']["encoder"],
            rho=self.optimizer_config['rho'],
            inner_iters=self.optimizer_config['inner_iters'],
            n_sample=self.ground_truth.dataset_size,
            input_dim=self.observed_dim,
            constraint_fn=lambda F: torch.sum(F, dim=1) - 1.0
        )
        return self.optimizer

    def setup_metrics(self):
        subspace_distance_metric = metric.SubspaceDistance()
        evaluate_metric = EvaluateMetric()
        constraint_error = metric.ConstraintError(lambda F: torch.sum(F, dim=1) - 1.0)

        self.metrics = torchmetrics.MetricCollection({
            'subspace_distance': subspace_distance_metric,
            # 'h_r_square': metric.ResidualNonlinearity(),
            'evaluate_metric': evaluate_metric,
            'constraint': constraint_error
        })
        self.metrics.eval()
        self.log_monitor = {"monitor": "validation_loss", "mode": "min"}

    def update_metrics(self, data, model_output, labels, idxes):
        self.metrics['evaluate_metric'].update(model_output[1], data, labels[1])
        self.metrics['subspace_distance'].update(idxes, self.optimizer.F_buffer, labels[0])
        self.metrics['constraint'].update(self.optimizer.F_buffer[idxes.to(self.optimizer.F_buffer.device)])


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
        self.e_net = nn.Sequential(*layers)

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
        hidden_sizes = list(self.config.get('hidden_layers').values())

        for h in hidden_sizes:
            layers.append(nn.Conv1d(in_channels, h * input_dim, kernel_size=1, groups=input_dim))
            layers.append(nn.ReLU())
            in_channels = h * input_dim

        layers.append(nn.Conv1d(in_channels, output_dim, kernel_size=1, groups=input_dim))
        self.d_net = nn.Sequential(*layers)
        # self.d_net = CNN(input_dim, output_dim, **self.config)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.d_net(x)
        return x.squeeze(-1)
