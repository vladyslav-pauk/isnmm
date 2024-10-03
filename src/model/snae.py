import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics
from torch import optim

from src.modules.ae_module import AutoEncoderModule
import src.modules.metric as metric
import src.modules.network as network


class Model(AutoEncoderModule):
    def __init__(self, ground_truth_model, encoder, decoder, model_config, optimizer_config):
        super().__init__(encoder, decoder)

        self.ground_truth = ground_truth_model
        self.optimizer_config = optimizer_config
        self.optimizer = None
        self.observed_dim = self.ground_truth.observed_dim
        self.sigma = 0
        self.mc_samples = 1

        self.metrics = None
        self.log_monitor = None
        self.setup_metrics()

    def loss_function(self, observed_batch, model_output, idxes):
        reconstructed_sample = model_output["reconstructed_sample"].squeeze(0)
        latent_sample = model_output["latent_sample"]

        loss = {"reconstruction": F.mse_loss(reconstructed_sample, observed_batch.squeeze(0))}

        regularization_loss = {}
        if hasattr(self.optimizer, "compute_constraint_errors"):
            regularization_loss = self.optimizer.compute_constraint_errors(latent_sample, idxes, observed_batch)
        loss.update(regularization_loss)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self.optimizer, "update_buffers"):
            self.optimizer.update_buffers(batch["idxes"], self(batch["data"])["latent_sample"])

    @staticmethod
    def reparameterization(sample):
        sample = torch.cat((sample, torch.zeros_like(sample[..., :1])), dim=-1)
        return F.softmax(sample, dim=-1)
    # todo: make it a part of encoder?

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        lr = self.optimizer_config["lr"]
        self.optimizer = optimizer_class([
            {'params': self.encoder.parameters(), 'lr': lr["encoder"]},
            {'params': self.decoder.linear_mixture.parameters(), 'lr': lr["decoder"]["linear"]},
            {'params': self.decoder.nonlinear_transform.parameters(), 'lr': lr["decoder"]["nonlinear"]}
        ], **self.optimizer_config["params"])
        return self.optimizer

    def setup_metrics(self):
        self.metrics = torchmetrics.MetricCollection({
            'subspace_distance': metric.SubspaceDistance(),
            'h_r_square': metric.ResidualNonlinearity(),
        })
        self.metrics.eval()
        self.log_monitor = {
            "monitor": "validation_loss",
            "mode": "min"
        }

    def update_metrics(self, data, model_output, labels, idxes):
        reconstructed_sample = model_output["reconstructed_sample"].mean(0)
        latent_sample = model_output["latent_sample"]
        linearly_mixed_sample = labels["linearly_mixed_sample"]
        latent_sample_qr = labels["latent_sample_qr"]

        self.metrics['subspace_distance'].update(
            idxes, latent_sample.mean(0), latent_sample_qr
        )
        self.metrics['h_r_square'].update(
            data, reconstructed_sample, linearly_mixed_sample
        )


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.network = None

    def construct(self, latent_dim, observed_dim):
        self.network = self.constructor(observed_dim, latent_dim - 1, **self.config)

    def forward(self, x):
        x = self.network.forward(x)
        return x, torch.zeros_like(x).to(x.device)


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = None
        self.nonlinear_transform = None

    def construct(self, latent_dim, observed_dim):
        self.linear_mixture = network.LinearPositive(
            torch.eye(observed_dim, latent_dim), **self.config
        )
        # self.linear_mixture.eval()

        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

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

# todo: proper cnn with linear layer and proper postnonlinearity (make a separate class PNLConstructor for FCN or CNN)
# fixme: cnae unequal dimensions
# fixme: experiment cnae on noisy data
# todo: clean up and readme
# fixme: make some runs and organize wandb to show Xiao
# fixme: train cnae with reparametrization
# fixme: train nisca with constrained optimization
