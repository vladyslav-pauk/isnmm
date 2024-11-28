from torchmetrics import MetricCollection
import torch

from src.modules.utils import save_metrics, plot_data
import src.modules.metric as metric


class ModelMetrics(MetricCollection):
    def __init__(self, show_plot=False, log_plot=False, save_plot=False, monitor=None):
        super().__init__([])
        self.metrics_list = [monitor]
        self.monitor = monitor
        self.show_plot = show_plot
        self.log_plot = log_plot
        self.save_plot = save_plot
        self.log_wandb = True
        self.true_model = None
        self.unmixing = False

        self.latent_dim = None

        self.plot_metrics = None #'latent'

    def setup_metrics(self, metrics_list=None):
        self.metrics_list = metrics_list
        self.image_dims = list(self.true_model.transform.unflatten(self.true_model.dataset.data).shape)
        self.image_dims[0] = self.latent_dim

        all_metrics = {
            'reconstruction': metric.Hyperspectral(
                latent_dim=self.latent_dim
            ),
            'psnr': metric.PSNR(),
            'error': metric.Hyperspectral(
                latent_dim=self.latent_dim
            ),
            'latent_components': metric.Hyperspectral(
                latent_dim=self.latent_dim,
                unmixing=self.unmixing
            ),
            'latent_mse': metric.data_mse.DataMse(
                unmixing=self.unmixing
            ),
            'latent_sam': metric.SpectralAngle(
                unmixing=self.unmixing
            ),
            'subspace_distance': metric.SubspaceDistance(
                unmixing=self.unmixing
            )
        }

        if not self.metrics_list:
            self.metrics_list = all_metrics.keys()

        metrics = {name: m for name, m in all_metrics.items() if name in self.metrics_list}

        super().__init__(metrics)
        return metrics

    def update(self, observed_sample, model_output, labels, idxes, model):

        metric_updates = {
            'reconstruction': {
                "noisy": observed_sample,
                "reconstructed": model_output['reconstructed_sample'].mean(dim=0),
            },
            'latent_components': {
                "latent_sample": model_output['latent_sample'].mean(dim=0),
                "noise": model.transform(model_output['posterior_parameterization'][1])
                # if model_output['latent_sample'].shape[0] == 1
                # else model_output['latent_sample'].std(dim=0),
            }
        }

        if labels:
            if labels['noiseless_data'] is not None:
                metric_updates.update({
                    'error': {
                        "noiseless": labels['noiseless_data'],
                        # "noise": model_output['reconstructed_sample'].std(dim=0) if model_output['reconstructed_sample'].shape[0] > 1 else model.sigma,
                        "rmse": ((model_output['reconstructed_sample'] - labels['noiseless_data']) ** 2).mean(dim=0).sqrt()
                    },
                    'psnr': {
                        "reconstructed": model_output['reconstructed_sample'].mean(dim=0),
                        "target": labels['noiseless_data']
                    },
                })

            if labels['latent_sample'] is not None:

                latent_sample = model_output['latent_sample'].mean(dim=0)
                latent_sample_true = labels["latent_sample"]
                latent_sample_qr = labels["latent_sample_qr"]

                metric_updates.update({
                    'latent_components': {
                        "latent_sample": latent_sample,
                        "true": latent_sample_true,
                        "noise": model.transform(model_output['posterior_parameterization'][1])
                    },
                    'latent_mse': {
                        "matrix_est": latent_sample,
                        "matrix_true": latent_sample_true
                    },
                    'latent_sam': {
                        "matrix_est": latent_sample,
                        "matrix_true": latent_sample_true
                    },
                    'subspace_distance': {
                        "sample": latent_sample,
                        "sample_qr": latent_sample_qr
                    }
                })

        for metric_name, kwargs in metric_updates.items():
            if self.metrics_list is None or metric_name in self.metrics_list:
                self[metric_name].update(**kwargs)

    def save(self, metrics, save_dir=None):
        save_metrics(metrics, save_dir)

    def compute(self):
        metrics = {}
        for metric_name in self.metrics_list:
            if metric_name in self:

                metric_value = self[metric_name].compute()

                if metric_value is not None:
                    metrics[metric_name] = metric_value

                if self.plot_metrics is None or self.plot_metrics in metric_name:
                    if self[metric_name].tensor is not None:
                        plot_data(
                            {metric_name: self[metric_name].tensor},
                            self.image_dims,
                            show_plot=self.show_plot,
                            save_plot=self.save_plot
                        )

                    if hasattr(self[metric_name], 'tensors'):
                        for key, value in self[metric_name].tensors.items():
                            plot_data(
                                {key: value},
                                self.image_dims,
                                show_plot=self.show_plot,
                                save_plot=self.save_plot
                            )

        return metrics

# todo: make a parent experiment class module with save_metrics and other universal structures
# todo: rec and kl save too
# todo: check if training history for metrics (login additional during validation) works
# todo: implement partially labeled metric:
#  (some data points are clear 1 material), find appropriate image
# todo: where is split dataset used?
