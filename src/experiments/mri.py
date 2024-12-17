import torch
from torchmetrics import MetricCollection

from src.modules.utils import save_metrics, unmix, permute
import src.modules.metric as metric


class ModelMetrics(MetricCollection):
    def __init__(self, show_plot=False, log_plot=False, save_plot=False, monitor=None):
        super().__init__([])
        self.plot_metrics = []
        self.metrics_list = [monitor]
        self.monitor = monitor

        self.show_plot = show_plot
        self.log_plot = log_plot
        self.save_plot = save_plot
        self.log_wandb = True

        self.model = None
        self.true_model = None
        self.latent_dim = None
        self.unmixing = None
        self.image_dims = None

    def setup_metrics(self, metrics_list=None):

        self.latent_dim = self.model.latent_dim
        self.unmixing = self.model.unmixing

        self.metrics_list = metrics_list

        self.image_dims = list(self.true_model.transform.unflatten(self.true_model.dataset.data).shape)
        self.image_dims[0] = self.latent_dim

        all_metrics = {
            'observed_reconstruction': metric.Tensor(),
            'observed_rmse': metric.DataMse(rmse=True),
            'observed_psnr': metric.PSNR(),
            'nonlinearity_rsquare': metric.ResidualNonlinearity(),
            'latent_components': metric.Tensor(),
            'latent_mse': metric.DataMse(db=True),
            'latent_sam': metric.SpectralAngle(),
            'latent_sd': metric.SubspaceDistance(),
            'latent_si': metric.Separation(db=False),
        }

        if not self.metrics_list:
            self.metrics_list = all_metrics.keys()
        metrics = {name: m for name, m in all_metrics.items() if name in self.metrics_list}

        super().__init__(metrics)
        return metrics

    def update(self, observed_sample, model_output, labels, idxes, model):

        latent_sample = model_output['latent_sample'].mean(dim=0)
        # latent_sample = model.transform(model_output['posterior_parameterization'][0])
        latent_variance = model.transform(model_output['posterior_parameterization'][1])

        if self.unmixing:
            latent_sample, mixing_matrix = unmix(
                latent_sample, self.latent_dim, self.unmixing
            )
            latent_variance = torch.matmul(latent_variance, torch.linalg.pinv(mixing_matrix).T)

        self.metrics_list = []
        metric_updates = {
            'observed_reconstruction': {
                "true": observed_sample,
                "reconstructed": model_output['reconstructed_sample'].mean(dim=0),
            }
        }

        if labels:
            if 'noiseless_sample' in labels:
                metric_updates.update({
                    'observed_rmse': {
                        "true": labels['noiseless_sample'],
                        "estimated": model_output['reconstructed_sample'].mean(dim=0)
                    },
                    'observed_psnr': {
                        "estimated": model_output['reconstructed_sample'].mean(dim=0),
                        "true": labels['noiseless_sample']
                    },
                })

                # metric_updates['observed_reconstruction'].update({
                #     "noiseless_true": labels['noiseless_sample']
                # })

            if 'latent_sample' in labels:
                metric_updates.update({
                    'nonlinearity_rsquare': {
                        "model_output": model_output,
                        "labels": labels,
                        "linearly_mixed_sample": model.decoder.linear_mixture(latent_sample),
                        "observed_sample": observed_sample
                    }
                })

                latent_sample, permutation = permute(latent_sample, labels["latent_sample"])
                # latent_variance = latent_variance[:, permutation]

                metric_updates.update({
                    'latent_components': {
                        "true": labels["latent_sample"],
                        "estimated": latent_sample
                        # "estimated_variance": latent_variance
                    },
                    'latent_mse': {
                        "estimated": latent_sample,
                        "true": labels["latent_sample"]
                    },
                    'latent_sam': {
                        "estimated": latent_sample,
                        "true": labels["latent_sample"]
                    },
                    'latent_sd': {
                        "estimated": latent_sample,
                        "true_qr": labels["latent_sample_qr"]
                    }
                })

        else:
            metric_updates.update({
                'latent_components': {
                    "estimated": latent_sample,
                    "estimated_variance": latent_variance
                },
                # 'nonlinearity': {
                #     "model_output": model_output,
                #     "linearly_mixed_sample": model.decoder.linear_mixture(latent_sample),
                #     "observed_sample": observed_sample
                # },
                'latent_si': {
                    "estimated": latent_sample
                }
            })
        self.metrics_list = metric_updates.keys()

        for metric_name, kwargs in metric_updates.items():
            if metric_name in self.metrics_list:
                self[metric_name].update(**kwargs)

    def compute(self):
        metrics = {}
        for metric_name in self.metrics_list:
            metric_value = self[metric_name].compute()

            if metric_value is not None:
                metrics[metric_name] = metric_value

            if any(metric in metric_name for metric in self.plot_metrics) or not self.plot_metrics:
                self[metric_name].plot(
                    image_dims=self.image_dims,
                    show_plot=self.show_plot,
                    save_plot=self.save_plot
                )

        return metrics

    def save(self, metrics, save_dir=None):
        save_metrics(metrics, save_dir)


# todo: print values of hypermarameters read from data in sweep parameter summary before training (now None)
# todo: make a parent experiment class module with save_metrics and other universal structures
# todo: rec and kl save too
# todo: check if training history for metrics (login additional during validation) works
# todo: implement partially labeled metric:
#  (some data points are clear 1 material), find appropriate image
# todo: where is split dataset used?
