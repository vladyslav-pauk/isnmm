import wandb
import torchmetrics
import pytorch_lightning as pl


import src.modules.metric as metric


class PNL(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.optimizer = None
        self.sigma = 0
        self.mc_samples = 1
        self.metrics = None

    def _setup_metrics(self, ground_truth=None):
        metrics = {}
        if ground_truth:
            metrics.update({
                'subspace_distance': metric.SubspaceDistance(),
                'r_square': metric.ResidualNonlinearity()
            })

        self.metrics = torchmetrics.MetricCollection(metrics)
        self.metrics.eval()

        wandb.define_metric(name="r_square", summary='max')

    def _update_metrics(self, observed_sample, model_output, labels, idxes):
        if labels:
            latent_sample = model_output["latent_sample"].mean(0)
            latent_sample_qr = labels["latent_sample_qr"]
            linearly_mixed_sample = self.decoder.linear_mixture(latent_sample)

            self.metrics['subspace_distance'].update(
                idxes, latent_sample, latent_sample_qr
            )
            self.metrics['r_square'].update(
                model_output, labels, linearly_mixed_sample, observed_sample
            )
