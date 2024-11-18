import json
import os
import wandb

import torch
from torchmetrics import MetricCollection

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

    def setup_metrics(self, metrics_list=None):
        self.metrics_list = metrics_list
        dims = self.true_model.transform.unflatten(self.true_model.dataset.data).shape
        all_metrics = {
            'reconstruction': metric.Hyperspectral(
                image_dims=dims,
                show_plot=self.show_plot,
                log_plot=self.log_plot,
                save_plot=self.save_plot
            ),
            'abundance': metric.Hyperspectral(
                image_dims=dims,
                show_plot=self.show_plot,
                log_plot=self.log_plot,
                save_plot=self.save_plot
            ),
            'psnr': metric.PSNR(
                image_dims=dims,
                show_plot=self.show_plot,
                log_plot=self.log_plot,
                save_plot=self.save_plot
            ),
        }

        if not self.metrics_list:
            self.metrics_list = all_metrics.keys()

        metrics = {name: m for name, m in all_metrics.items() if name in self.metrics_list}

        super().__init__(metrics)
        return metrics

    def update(self, observed_sample, model_output, labels, idxes, model):
        metric_updates = {
            'reconstruction': {
                "noiseless": labels['noiseless_data'],
                "reconstructed": model_output['reconstructed_sample'].mean(dim=0),
                "noisy": observed_sample,
                # "noise": model_output['reconstructed_sample'].std(dim=0) if model_output['reconstructed_sample'].shape[0] > 1 else model.sigma,
                "mse": ((model_output['reconstructed_sample'] - labels['noiseless_data']) ** 2).mean(dim=0)
            },
            'abundance': {
                "abundance": model_output['latent_sample'].mean(dim=0),
                "noise": model.transform(model_output['posterior_parameterization'][1])
                # if model_output['latent_sample'].shape[0] == 1
                # else model_output['latent_sample'].std(dim=0),
            },
            'psnr': {
                "reconstructed": model_output['reconstructed_sample'].mean(dim=0),
                "target": labels['noiseless_data']
            },
        }

        for metric_name, kwargs in metric_updates.items():
            if self.metrics_list is None or metric_name in self.metrics_list:
                self[metric_name].update(**kwargs)

    def save_metrics(self, metrics, save_dir=None):
        if wandb.run is not None and save_dir is None:
            base_dir = os.path.join(wandb.run.dir.split('wandb')[0], 'results')
            sweep_id = wandb.run.dir.split('/')[-4].split('-')[-1]
            output_path = os.path.join(base_dir, f'sweep-{sweep_id}', "sweep_data.json")
        else:
            if save_dir is None:
                save_dir = './results'

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_path = os.path.join(
                project_root, 'experiments', os.environ["EXPERIMENT"], save_dir, os.environ["RUN_ID"], "sweep_data.json"
            )

        run_id = os.environ.get("RUN_ID", "default")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

        if run_id not in existing_data:
            existing_data[run_id] = {"metrics": {}}

        existing_data[run_id]["metrics"].update(metrics)

        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

        print("Final metrics saved:")
        for key, value in metrics.items():
            print(f"\t{key} = {value}")


# todo: make a parent experiment class module with save_metrics and other universal structures
# todo: rec and kl save too
# todo: check if training history for metrics (login additional during validation) works
# todo: implement partially labeled metric:
#  (some data points are clear 1 material), find appropriate image
# todo: where is split dataset used?
