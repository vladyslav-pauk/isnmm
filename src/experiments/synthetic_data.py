import wandb
from torchmetrics import MetricCollection

import src.modules.metric as metric
import src.model as model_package


class ModelMetrics(MetricCollection):
    def __init__(self, true_model=None, monitor=None):
        self.metrics_list = [monitor]
        self.monitor = monitor
        self.show_plots = False
        self.log_plots = False
        self.true_model = true_model

        self._setup_metrics()

    def _setup_metrics(self):
        all_metrics = {
            'mixture_mse_db': metric.MatrixMse(db=True),
            'mixture_sam': metric.SpectralAngle(),
            'mixture_matrix_change': metric.MatrixChange(),
            'subspace_distance': metric.SubspaceDistance(),
            'r_square': metric.ResidualNonlinearity(show_plot=self.show_plots, log_plot=self.log_plots),
            'latent_mse': metric.data_mse.DataMse(),
            'mixture_log_volume': metric.MatrixVolume()
        }

        if not self.metrics_list:
            self.metrics_list = all_metrics.keys()

        metrics = {name: m for name, m in all_metrics.items() if name in self.metrics_list}

        for metric_name in metrics:
            if metric_name == self.monitor:
                wandb.define_metric(name=metric_name, summary='max')

        self.linear_mixture_true = self.true_model.linear_mixture if self.true_model else None

        super().__init__(metrics)
        return metrics

    def _update(self, observed_sample, model_output, labels, idxes, model):
        if labels:
            latent_sample_true = labels["latent_sample"]
            latent_sample_qr = labels["latent_sample_qr"]
            latent_sample_mean = model_output["latent_sample_mean"].mean(0)
            linearly_mixed_sample = model.decoder.linear_mixture(model_output["latent_sample"].mean(0))
            latent_sample_unmixed = self.unmix(latent_sample_mean, model, latent_sample_true.size(-1))
            linear_mixture = model.decoder.linear_mixture.matrix

            metric_updates = {
                'subspace_distance': (idxes, latent_sample_unmixed, latent_sample_qr),
                'r_square': (model_output, labels, linearly_mixed_sample, observed_sample, latent_sample_unmixed),
                'latent_mse': (model_output["latent_sample"].mean(0), latent_sample_true),
                'mixture_mse_db': (self.linear_mixture_true, linear_mixture),
                'mixture_sam': (self.linear_mixture_true, linear_mixture),
                'mixture_log_volume': (linear_mixture,),
                'mixture_matrix_change': (linear_mixture,)
            }

            for metric_name, args in metric_updates.items():
                if self.metrics_list is None or metric_name in self.metrics_list:
                    self[metric_name].update(*args)

    def unmix(self, latent_sample, model, latent_dim):
        if model.unmixing:
            unmixing_model = getattr(model_package, model.unmixing).Model
            unmixing = unmixing_model(
                observed_dim=model.observed_dim,
                latent_dim=latent_dim,
                dataset_size=model.trainer.datamodule.dataset_size,
            )
            latent_sample = unmixing.estimate_abundances(latent_sample.squeeze().cpu().detach())
            return latent_sample
        return latent_sample
