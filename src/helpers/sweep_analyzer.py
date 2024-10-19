import json
import os
from src.helpers.wandb_tools import login_wandb, fetch_wandb_sweep
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class SweepAnalyzer:
    def __init__(self, experiment, sweep_id):
        self.experiment = experiment
        self.sweep_id = sweep_id
        self.output_dir = f"../experiments/{experiment}/sweeps"
        self.output_file = f"{sweep_id}.json"

        os.makedirs(self.output_dir, exist_ok=True)

        self.sweep_data = None
        self._fetch_data()
        self._save_data()

    def extract_metrics(self, metric="latent_mse_db", covariate="snr", comparison="model_name"):
        data = defaultdict(lambda: defaultdict(list))

        for run_id, content in self.sweep_data.items():
            seed = content['config']['torch_seed']

            metric_value = content['metrics'][metric]
            covariate_name = content['config'][covariate]
            comparison_name = content['config'][comparison]

            data[comparison_name][covariate].append(covariate_name)
            data[comparison_name]['metric'].append(metric_value)
            data[comparison_name]['seed'].append(seed)

        return data

    def average_seeds(self, data):
        averaged_data = {}
        first_experiment = data[list(data.keys())[0]]
        covariate = next(key for key in first_experiment if key not in ['metric', 'seed'])

        for model, values in data.items():
            snr_values = np.array(values[covariate])
            metric_values = np.array(values['metric'])

            unique_snrs = np.unique(snr_values)
            snr_metric_averages = []
            snr_metric_stddev = []

            for snr in unique_snrs:
                indices = np.where(snr_values == snr)
                snr_metric = metric_values[indices]
                snr_metric_averages.append(np.mean(snr_metric))
                snr_metric_stddev.append(np.std(snr_metric))

            averaged_data[model] = {
                covariate: unique_snrs,
                'metric_avg': np.array(snr_metric_averages),
                'metric_std': np.array(snr_metric_stddev)
            }

        return averaged_data

    def plot_metric(self, data, save=False):
        first_experiment = data[list(data.keys())[0]]
        covariate = next(key for key in first_experiment if key not in ['metric', 'seed'])

        plt.figure(figsize=(10, 6))

        for model, values in data.items():
            snr_values = values[covariate]
            metric_avg = values['metric_avg']
            metric_std = values['metric_std']

            plt.fill_between(
                snr_values,
                metric_avg - metric_std,
                metric_avg + metric_std,
                alpha=0.2
            )

            plt.plot(snr_values, metric_avg, label=f'Model {model}')

        plt.xlabel(covariate)
        plt.ylabel('Metric')
        plt.title('Metric vs SNR (Averaged over Seeds)')
        plt.legend()
        plt.show()

        if save:
            plt.savefig(os.path.join(self.output_dir, f"{self.sweep_id}_metric_plot.png"))

    def _fetch_data(self):
        login_wandb()
        self.sweep_data = fetch_wandb_sweep(self.experiment, self.sweep_id)

    def _save_data(self):
        with open(os.path.join(self.output_dir, self.output_file), "w") as f:
            json.dump(self.sweep_data, f, indent=4)

        print(f"Saved sweep data to {os.path.join(self.output_dir, self.output_file)}")