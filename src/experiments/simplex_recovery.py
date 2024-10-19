import wandb
from torch import is_tensor
from torchmetrics import MetricCollection

import src.modules.metric as metric


class ModelMetrics(MetricCollection):
    def __init__(self, model):
        metrics = self._setup_metrics(model)
        super().__init__(metrics)

        self.linear_mixture_true = model.linear_mixture if model else None

    @staticmethod
    def _setup_metrics(model=None):
        metrics = {
            'mixture_log_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange()
        }
        if model:
            metrics.update({
                'mixture_mse_db': metric.MatrixMse(),
                'mixture_sam': metric.SpectralAngle(),
                'subspace_distance': metric.SubspaceDistance(),
                'latent_mse_db': metric.MatrixMse()
            })
        wandb.define_metric(name="mixture_matrix_change", summary='min')
        return metrics

    def _update(self, data, model_output, labels, idxes, model):
        linear_mixture = model.decoder.linear_mixture.matrix

        if is_tensor(self.linear_mixture_true):
            latent_sample = model_output["latent_sample"].mean(0)
            latent_sample_true = labels["latent_sample"]
            latent_sample_qr = labels["latent_sample_qr"]
            linear_mixture_true = self.linear_mixture_true

            self['mixture_mse_db'].update(linear_mixture_true, linear_mixture)
            self['mixture_sam'].update(linear_mixture_true, linear_mixture)
            self['subspace_distance'].update(idxes, latent_sample, latent_sample_qr)
            self['latent_mse_db'].update(latent_sample, latent_sample_true)

        self['mixture_log_volume'].update(linear_mixture)
        self['mixture_matrix_change'].update(linear_mixture)
