import os
import json
import wandb

import torch
from torchmetrics import MetricCollection

import src.modules.metric as metric
import src.model as model_package


class ModelMetrics(MetricCollection):
    def __init__(self, true_model=None, monitor=None):
        self.metrics_list = [monitor]
        self.monitor = monitor
        self.show_plots = False
        self.log_plots = False
        self.log_wandb = True
        self.save_plots = False
        self.true_model = true_model

        self._setup_metrics()

    def _setup_metrics(self):
        all_metrics = {
            'subspace_distance': metric.SubspaceDistance(),
            'r_square': metric.ResidualNonlinearity(
                show_plot=self.show_plots, log_plot=self.log_plots, save_plot=self.save_plots
            ),
            'latent_mse': metric.data_mse.DataMse(),
            'latent_sam': metric.SpectralAngle(),
            # 'mixture_mse_db': metric.MatrixMse(db=True),
            # 'mixture_sam': metric.SpectralAngle(),
            # 'mixture_matrix_change': metric.MatrixChange(),
            # 'mixture_log_volume': metric.MatrixVolume()
        }
        # fixme: add unmixing metric

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

        self.linear_mixture_true = self.true_model.linear_mixture if self.true_model else None

        super().__init__(metrics)
        return metrics

    def _update(self, observed_sample, model_output, labels, idxes, model):
        if labels:
            latent_sample_true = labels["latent_sample"]
            latent_sample_qr = labels["latent_sample_qr"]
            # latent_sample_averaged = model_output["latent_sample_mean"].mean(0)
            latent_sample_mean = model_output["latent_sample"].mean(0)
            model.unmixing = None
            latent_sample_unmixed, linear_mixture = self.unmix(latent_sample_mean, model)

            linearly_mixed_sample = model.decoder.linear_mixture(latent_sample_mean)

            metric_updates = {
                'subspace_distance': (idxes, latent_sample_unmixed, latent_sample_qr),
                'r_square': (model_output, labels, linearly_mixed_sample, observed_sample, latent_sample_unmixed),
                'latent_mse': (latent_sample_unmixed, latent_sample_true),
                'latent_sam': (latent_sample_unmixed, latent_sample_true),
                # 'mixture_mse_db': (self.linear_mixture_true, linear_mixture),
                # 'mixture_sam': (self.linear_mixture_true, linear_mixture),
                # 'mixture_log_volume': (linear_mixture,),
                # 'mixture_matrix_change': (linear_mixture,)
            }
            # fixme: why i use idxes only in subspace distance

            for metric_name, args in metric_updates.items():
                if self.metrics_list is None or metric_name in self.metrics_list:
                    self[metric_name].update(*args)

    def save_metrics(self, metrics, save_dir=None):
        if wandb.run is not None and save_dir is None:  # Check if wandb.run is active
            base_dir = os.path.join(wandb.run.dir.split('wandb')[0], 'results')
            sweep_id = wandb.run.dir.split('/')[-4].split('-')[-1]
            output_path = os.path.join(base_dir, f'sweep-{sweep_id}', "sweep_data.json")
        else:
            if save_dir is None:
                save_dir = './results'

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_path = os.path.join(
                project_root, 'experiments', os.environ["EXPERIMENT"], save_dir, os.environ["RUN_ID"],"sweep_data.json"
            )
        run_id = os.environ.get("RUN_ID", "default")

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

        if run_id not in existing_data:
            existing_data[run_id] = {"metrics": {}}
        existing_data[run_id]["metrics"].update(metrics)

        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print("Final metrics:")
        for key, value in metrics.items():
            print(f"\t{key} = {value}")

    def unmix(self, latent_sample, model):
        if model.unmixing:
            latent_dim = latent_sample.size(-1)
            unmixing_model = getattr(model_package, model.unmixing.upper()).Model
            unmixing = unmixing_model(
                latent_dim=latent_dim,
                dataset_size=model.trainer.datamodule.dataset_size,
            )
            latent_sample, mixing_matrix = unmixing.estimate_abundances(latent_sample.squeeze().cpu().detach())

            # unmixing.plot_multiple_abundances(latent_sample, [0,1,2,3,4,5,6,7,8,9])
            # unmixing.plot_mse_image(rows=100, cols=10)

            return latent_sample, mixing_matrix
        return latent_sample, model.decoder.linear_mixture.matrix
