import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics

from src.modules.ae_module import AutoEncoderModule
# from src.model.nisca import Decoder
from src.modules.network import CNN
import src.modules.metric as metric
from src.modules.metric import EvaluateMetric


class Model(AutoEncoderModule):
    def __init__(self, ground_truth_model, encoder, decoder, model_config, optimizer_config):
        super().__init__(encoder, decoder)

        self.ground_truth = ground_truth_model
        self.lr = optimizer_config.get('lr', 1e-3)

        n_sample = ground_truth_model.dataset_size
        input_dim = ground_truth_model.observed_dim
        self.mult = nn.Parameter(torch.randn(n_sample), requires_grad=False)
        self.register_buffer('F_buffer', torch.zeros((n_sample, input_dim)))
        self.register_buffer('count_buffer', torch.zeros(n_sample, dtype=torch.int32))
        self.rho = optimizer_config.get('rho', 1e2)
        self.inner_iters = optimizer_config.get('inner_iters', 1)

        self.setup_metrics()

    def on_train_batch_end(self, train_step_output, batch, batch_idx):
        data, _, idxes = batch
        qfx, fx, _ = self(data)
        self.F_buffer[idxes] = fx.detach()
        self.count_buffer[idxes] += 1

        if (self.global_step + 1) % self.inner_iters == 0:
            self.update_multipliers()
        pass

    def update_multipliers(self):
        idxes = self.count_buffer.nonzero(as_tuple=True)[0]
        F = self.F_buffer[idxes]
        diff = torch.sum(F, dim=1) - 1.0
        self.mult[idxes] += self.rho * diff
        self.F_buffer[idxes] = 0.0
        self.count_buffer[idxes] = 0

    def loss_function(self, x, model_output, idxes):
        qfx, fx, _ = model_output
        tmp = torch.sum(fx, dim=1) - 1.0
        mult = self.mult[idxes]
        reconstruct_err = F.mse_loss(qfx, x)
        feasible_err = torch.dot(mult, tmp) / x.shape[0]
        augmented_err = (self.rho / 2) * torch.norm(tmp) ** 2 / x.shape[0]
        return {"reconstruction": reconstruct_err, "feasible": feasible_err, "augmented": augmented_err}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr["encoder"])

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
        self.metrics['subspace_distance'].update(idxes, self.F_buffer[idxes], labels[0])
        self.metrics['constraint'].update(self.F_buffer[idxes])


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