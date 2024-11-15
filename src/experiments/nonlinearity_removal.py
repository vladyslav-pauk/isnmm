import wandb
from torchmetrics import MetricCollection

import src.modules.metric as metric
import src.model as model_package

class ModelMetrics(MetricCollection):
    def __init__(self, model, metrics_list=None):
        self.metrics_list = metrics_list if metrics_list is not None else []
        metrics = self._setup_metrics(model)
        self.linear_mixture_true = model.linear_mixture if model else None
        super().__init__(metrics)

    def _setup_metrics(self, model=None):
        all_metrics = {
            'mixture_mse_db': metric.MatrixMse(db=True),
            'subspace_distance': metric.SubspaceDistance(),
            'r_square': metric.ResidualNonlinearity(),
            'latent_mse': metric.data_mse.DataMse()
        }

        # Filter metrics based on metrics_list
        metrics = {name: m for name, m in all_metrics.items() if name in self.metrics_list}

        # Define metric properties in WandB for available metrics
        for metric_name in metrics:
            if metric_name == 'r_square':
                wandb.define_metric(name=metric_name, summary='max')

        return metrics

    def _update(self, observed_sample, model_output, labels, idxes, model):
        if labels:
            latent_sample_true = labels["latent_sample"]
            latent_sample_qr = labels["latent_sample_qr"]
            latent_sample_mean = model_output["latent_sample_mean"].mean(0)
            linearly_mixed_sample = model.decoder.linear_mixture(model_output["latent_sample"].mean(0))

            latent_sample_unmixed = self.unmix(latent_sample_mean, model, latent_sample_true.size(-1))

            if 'subspace_distance' in self.metrics_list:
                self['subspace_distance'].update(
                    idxes, latent_sample_unmixed, latent_sample_qr
                )
            if 'r_square' in self.metrics_list:
                self['r_square'].update(
                    model_output, labels, linearly_mixed_sample, observed_sample, latent_sample_unmixed
                )
            if 'latent_mse' in self.metrics_list:
                self['latent_mse'].update(
                    model_output["latent_sample"].mean(0), latent_sample_true
                )
            if 'mixture_mse_db' in self.metrics_list:
                linear_mixture = model.decoder.linear_mixture.matrix
                linear_mixture_true = self.linear_mixture_true
                self['mixture_mse_db'].update(linear_mixture_true, linear_mixture)

    def unmix(self, latent_sample, model, latent_dim):
        if model.unmixing:
            base_model = getattr(model_package, model.unmixing).Model
            unmixing = base_model(
                observed_dim=model.observed_dim,
                latent_dim=latent_dim,
                dataset_size=model.trainer.datamodule.dataset_size,
            )
            latent_sample = unmixing.estimate_abundances(latent_sample.squeeze().cpu().detach())
            return latent_sample
        return latent_sample