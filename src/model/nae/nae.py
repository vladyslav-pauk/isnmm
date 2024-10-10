import wandb
from torch.nn import functional as F
import torchmetrics

from src.model.ae_module import AutoEncoderModule
import src.modules.metric as metric


class NAE(AutoEncoderModule):
    def __init__(self, encoder, decoder, model_config):
        super().__init__(encoder, decoder)

        self.optimizer = None
        self.sigma = 0
        self.mc_samples = 1
        self.metrics = None

    def loss_function(self, observed_batch, model_output, idxes):
        reconstructed_sample = model_output["reconstructed_sample"].squeeze(0)
        latent_sample = model_output["latent_sample"].squeeze(0)

        loss = {"reconstruction": F.mse_loss(reconstructed_sample, observed_batch)}

        regularization_loss = {}
        if hasattr(self.optimizer, "compute_regularization_loss"):
            regularization_loss = self.optimizer.compute_regularization_loss(latent_sample, observed_batch, idxes)
        loss.update(regularization_loss)

        return loss

    def setup_metrics(self):
        self.metrics = torchmetrics.MetricCollection({
            'subspace_distance': metric.SubspaceDistance(),
            'r_square': metric.ResidualNonlinearity()
        })
        self.metrics.eval()

        wandb.define_metric(name="r_square", summary='max')

    def update_metrics(self, observed_sample, model_output, labels, idxes):
        latent_sample = model_output["latent_sample"].mean(0)
        latent_sample_qr = labels["latent_sample_qr"]
        linearly_mixed_sample = self.decoder.linear_mixture(latent_sample)

        self.metrics['subspace_distance'].update(
            idxes, latent_sample, latent_sample_qr
        )
        self.metrics['r_square'].update(
            model_output, labels, linearly_mixed_sample, observed_sample
        )
