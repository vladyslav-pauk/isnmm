from torchmetrics import MetricCollection
import torch

from src.modules.utils import save_metrics

import src.modules.metric as metric
import src.model as model_package
from src.modules.utils import unmix, permute


class ModelMetrics(MetricCollection):
    def __init__(self, show_plot=False, log_plot=False, save_plot=False, monitor=None):
        super().__init__([])
        self.plot_metrics = []
        monitor = 'latent_mse'
        self.metrics_list = [monitor]
        self.monitor = monitor
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.log_plot = log_plot
        self.log_wandb = True
        self.true_model = None
        self.model = None

    def setup_metrics(self, metrics_list=None):
        self.latent_dim = self.model.latent_dim
        self.unmixing = self.model.unmixing

        self.metrics_list = metrics_list
        all_metrics = {
            'latent_components': metric.Tensor(),
            'subspace_distance': metric.SubspaceDistance(),
            'r_square': metric.ResidualNonlinearity(),
            'latent_mse': metric.data_mse.DataMse(db=True),
            'latent_sam': metric.SpectralAngle(),
            # 'mixture_mse_db': metric.MatrixMse(db=True),
            # 'mixture_sam': metric.SpectralAngle(),
            # 'mixture_matrix_change': metric.MatrixChange(),
            # 'mixture_log_volume': metric.MatrixVolume()
        }
        # todo: add unmixing metric

        # base_model = 'MVES'
        # latent_sample_true = datamodule.latent_sample
        # observed_data = datamodule.observed_sample
        # unmixing_model = getattr(model_package, base_model).Model
        # unmixing = unmixing_model(latent_dim=latent_sample_true.size(-1), dataset_size=latent_sample_true.size(0))
        # with torch.no_grad():
        #     latent_sample_mixed = model(observed_data)['reconstructed_sample'].mean(0)
        #     linear_mixture = model.decoder.linear_mixture.matrix.cpu().detach()
        # latent_sample = unmixing.estimate_abundances(latent_sample_mixed)
        # print(linear_mixture)
        # unmixing.compute_metrics(linear_mixture, latent_sample, latent_sample_true)

        if not self.metrics_list:
            self.metrics_list = all_metrics.keys()
        metrics = {name: m for name, m in all_metrics.items() if name in self.metrics_list}
        # if self.log_wandb:
        #     for metric_name in metrics:
        #         if metric_name == self.monitor:
        #             wandb.define_metric(name=metric_name, summary='max')

        # self.linear_mixture_true = self.true_model.linear_mixture if self.true_model else None

        super().__init__(metrics)
        return metrics

    def update(self, observed_sample, model_output, labels, idxes, model):
        if labels:
            # latent_sample_true = labels["latent_sample"]
            # latent_sample_qr = labels["latent_sample_qr"]
            # latent_sample_averaged = model_output["latent_sample_mean"].mean(0)
            # latent_sample_mean = model_output["latent_sample"].mean(0)
            # model.unmixing = None
            # latent_sample_unmixed, linear_mixture = self.unmix(latent_sample_mean, latent_sample_true.shape[-1], model)

            # latent_sample_unmixed = latent_sample_mean
            # if self.unmixing:
            #     latent_sample_unmixed, mixing_matrix = unmix(
            #         latent_sample_mean, self.latent_dim, self.unmixing
            #     )
            #     mixing_matrix_pinv = torch.linalg.pinv(mixing_matrix)

            # latent_sample_unmixed, _ = permute(latent_sample_unmixed, latent_sample_true)

            # linearly_mixed_sample = model.decoder.linear_mixture(latent_sample_mean)

            latent_sample = model_output['latent_sample'].mean(dim=0)
            # latent_sample = model.transform(model_output['posterior_parameterization'][0])
            latent_variance = model.transform(model_output['posterior_parameterization'][1])
            linear_mixture = model.decoder.linear_mixture(latent_sample)

            if self.unmixing:
                linear_mixture = latent_sample
                latent_sample, mixing_matrix = unmix(
                    latent_sample, self.latent_dim, self.unmixing
                )
                latent_variance = torch.matmul(latent_variance, torch.linalg.pinv(mixing_matrix).T)

            latent_sample, permutation = permute(latent_sample, labels["latent_sample"])

            metric_updates = {
                # 'subspace_distance': (latent_sample_unmixed, latent_sample_qr),
                # 'r_square': (model_output, labels, linearly_mixed_sample, observed_sample, latent_sample_unmixed),
                # # 'latent_mse': (latent_sample_unmixed, latent_sample_true),
                # 'latent_mse': {
                #     "reconstructed": latent_sample_unmixed,
                #     "target": latent_sample_true
                # },
                # 'latent_sam': (latent_sample_unmixed, latent_sample_true),
                # 'r_square': model_output, labels, linearly_mixed_sample, observed_sample, latent_sample_unmixed,
                'latent_components': {
                    "true": labels["latent_sample"],
                    "estimated": latent_sample
                    # "estimated_variance": latent_variance
                },
                'r_square': {
                    "model_output": model_output,
                    "labels": labels,
                    "linearly_mixed_sample": linear_mixture,
                    "observed_sample": observed_sample
                },
                'subspace_distance': {
                    "estimated": latent_sample,
                    "true_qr": labels["latent_sample_qr"]
                },
                'latent_mse': {
                    "estimated": latent_sample,
                    "true": labels["latent_sample"]
                },
                'latent_sam': {
                    "estimated": latent_sample,
                    "true": labels["latent_sample"]
                }
                # 'mixture_mse_db': (self.linear_mixture_true, linear_mixture),
                # 'mixture_sam': (self.linear_mixture_true, linear_mixture),
                # 'mixture_log_volume': (linear_mixture,),
                # 'mixture_matrix_change': (linear_mixture,)
            }
            # todo: why i use idxes only in subspace distance

            for metric_name, kwargs in metric_updates.items():
                if self.metrics_list is None or metric_name in self.metrics_list:
                    self[metric_name].update(**kwargs)

    def compute(self):
        metrics = {}
        for metric_name in self.metrics_list:
            metric_value = self[metric_name].compute()

            if metric_value is not None:
                metrics[metric_name] = metric_value

            if any(metric in metric_name for metric in self.plot_metrics) or not self.plot_metrics:
                self[metric_name].plot(
                    image_dims=None,
                    show_plot=self.show_plot,
                    save_plot=self.save_plot
                )
        return metrics

    def save(self, metrics, save_dir=None):
        save_metrics(metrics, save_dir)

    # def save_metrics(self, metrics, save_dir=None):
    #     if wandb.run is not None and save_dir is None:
    #         base_dir = os.path.join(wandb.run.dir.split('wandb')[0], 'results')
    #         sweep_id = wandb.run.dir.split('/')[-4].split('-')[-1]
    #         output_path = os.path.join(base_dir, f'sweep-{sweep_id}', "sweep_data.json")
    #     else:
    #         if save_dir is None:
    #             save_dir = './results'
    #
    #         project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    #         output_path = os.path.join(
    #             project_root, 'experiments', os.environ["EXPERIMENT"], save_dir, os.environ["RUN_ID"],"sweep_data.json"
    #         )
    #     run_id = os.environ.get("RUN_ID", "default")
    #
    #     if os.path.exists(output_path):
    #         with open(output_path, 'r') as f:
    #             existing_data = json.load(f)
    #     else:
    #         existing_data = {}
    #
    #     metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    #
    #     if run_id not in existing_data:
    #         existing_data[run_id] = {"metrics": {}}
    #     existing_data[run_id]["metrics"].update(metrics)
    #
    #     output_dir = os.path.dirname(output_path)
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #
    #     with open(output_path, 'w') as f:
    #         json.dump(existing_data, f, indent=2)
    #
    #     print("Final metrics saved:")
    #     for key, value in metrics.items():
    #         print(f"\t{key} = {value}")
