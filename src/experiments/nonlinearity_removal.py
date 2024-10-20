import wandb
from torchmetrics import MetricCollection

import src.modules.metric as metric


class ModelMetrics(MetricCollection):
    def __init__(self, model):
        metrics = self._setup_metrics(model)
        super().__init__(metrics)

    @staticmethod
    def _setup_metrics(model=None):
        metrics = {}
        if model:
            metrics.update({
                'subspace_distance': metric.SubspaceDistance(),
                'r_square': metric.ResidualNonlinearity(),
                'latent_mse_db': metric.matrix_mse.MatrixMse()
            })

        wandb.define_metric(name="r_square", summary='max')
        return metrics

    def _update(self, observed_sample, model_output, labels, idxes, model):
        if labels:
            latent_sample = model_output["latent_sample"].mean(0)
            latent_sample_true = labels["latent_sample"]
            latent_sample_qr = labels["latent_sample_qr"]
            linearly_mixed_sample = model.decoder.linear_mixture(latent_sample)

            self['subspace_distance'].update(
                idxes, latent_sample, latent_sample_qr
            )
            self['r_square'].update(
                model_output, labels, linearly_mixed_sample, observed_sample
            )
            self['latent_mse_db'].update(
                latent_sample, latent_sample_true
            )
