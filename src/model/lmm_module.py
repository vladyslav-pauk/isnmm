import torch
import torchmetrics
import wandb
import pytorch_lightning as pl

import src.modules.metric as metric


class LMM(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.linear_mixture_true = None
        self.metrics = None

    def _setup_metrics(self, ground_truth=None):

        metrics = {
            'mixture_log_volume': metric.MatrixVolume(),
            'mixture_matrix_change': metric.MatrixChange()
        }
        if ground_truth:
            self.latent_dim = ground_truth.latent_dim
            self.sigma = ground_truth.sigma
            self.linear_mixture_true = ground_truth.linear_mixture

            metrics.update({
                'mixture_mse_db': metric.MatrixMse(),
                'mixture_sam': metric.SpectralAngle(),
                'subspace_distance': metric.SubspaceDistance(),
                'latent_mse': metric.MatrixMse()
            })

        self.metrics = torchmetrics.MetricCollection(metrics)
        self.metrics.eval()

        wandb.define_metric(name="mixture_matrix_change", summary='min')

    def _update_metrics(self, data, model_output, labels, idxes):
        linear_mixture = self.decoder.linear_mixture.matrix

        if torch.is_tensor(self.linear_mixture_true):
            latent_sample = model_output["latent_sample"].mean(0)
            latent_sample_true = labels["latent_sample"]
            latent_sample_qr = labels["latent_sample_qr"]
            linear_mixture_true = self.linear_mixture_true

            self.metrics['mixture_mse_db'].update(linear_mixture_true, linear_mixture)
            self.metrics['mixture_sam'].update(linear_mixture_true, linear_mixture)
            self.metrics['subspace_distance'].update(idxes, latent_sample, latent_sample_qr)
            self.metrics['latent_mse'].update(latent_sample, latent_sample_true)

        self.metrics['mixture_log_volume'].update(linear_mixture)
        self.metrics['mixture_matrix_change'].update(linear_mixture)
