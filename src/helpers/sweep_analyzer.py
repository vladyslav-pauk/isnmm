import json
import os
from src.helpers.wandb_tools import login_wandb, fetch_wandb_sweep
from src.helpers.utils import font_style, format_string
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class SweepAnalyzer:
    def __init__(self, experiment, sweep_id):
        self.experiment = experiment
        self.sweep_id = sweep_id
        self.output_dir = f"../experiments/{experiment}/sweeps"
        self.output_file = f"{sweep_id}.json"
        # todo: fix directories, run from project root

        os.makedirs(self.output_dir, exist_ok=True)

        self.sweep_data = None
        self._fetch_data()
        self._save_data()

    def extract_metrics(self, metric="latent_mse_db", covariate="snr", comparison="model_name"):
        # Refactor the data structure to have three main keys
        refactored_data = {
            'covariate': {'name': covariate, 'values': defaultdict(list)},
            'metric': {'name': metric, 'values': defaultdict(list)},
            'comparison': {'name': comparison, 'values': defaultdict(list)}
        }

        for run_id, content in self.sweep_data.items():
            seed = content['config']['torch_seed']

            # Check if the metric is a dictionary (e.g., {'max': value})
            metric_value = content['metrics'][metric]
            if isinstance(metric_value, dict):
                # Extract the 'max' value if available, or another key if needed
                if 'max' in metric_value:
                    metric_value = metric_value['max']
                else:
                    raise ValueError(f"Expected 'max' key in metric dictionary, found: {metric_value.keys()}")

            # For direct numerical values (like reconstruction loss), use them as is
            covariate_value = content['config'][covariate]
            comparison_value = content['config'][comparison]

            # Assign values to corresponding refactored sections
            refactored_data['covariate']['values'][comparison_value].append(covariate_value)
            refactored_data['metric']['values'][comparison_value].append(metric_value)
            refactored_data['comparison']['values'][comparison_value].append(seed)

        return refactored_data
        # todo: metrix is automatically one with 'max'? 'r_square': {'max': 0.4832684993743897},...

    def average_seeds(self, refactored_data):
        formatted_data = []
        covariate_name = refactored_data['covariate']['name']
        metric_name = refactored_data['metric']['name']

        for comparison_value, covariate_values in refactored_data['covariate']['values'].items():
            metric_values = np.array(refactored_data['metric']['values'][comparison_value])
            covariate_values = np.array(covariate_values)

            unique_covariates = np.unique(covariate_values)
            metric_averages = []
            metric_stddevs = []

            for val in unique_covariates:
                indices = np.where(covariate_values == val)
                covariate_metric = metric_values[indices]
                metric_averages.append(np.mean(covariate_metric))
                metric_stddevs.append(np.std(covariate_metric))

            formatted_data.append({
                'model_name': comparison_value,
                covariate_name: unique_covariates,
                f'{metric_name}_avg': np.array(metric_averages),
                f'{metric_name}_std': np.array(metric_stddevs)
            })
        return formatted_data

    def plot_metric(self, averaged_data, save=False):
        font = font_style()

        plt.rc('font', **font)

        first_experiment = averaged_data[0]
        comparison_name = list(first_experiment.keys())[0]
        covariate_name = list(first_experiment.keys())[1]
        metric_name = list(first_experiment.keys())[2].replace('_avg', '')

        plt.figure(figsize=(10, 6))

        for experiment_data in averaged_data:
            model_name = experiment_data['model_name']
            covariate_values = experiment_data[covariate_name]
            metric_avg = experiment_data[f'{metric_name}_avg']
            metric_std = experiment_data[f'{metric_name}_std']

            plt.fill_between(
                covariate_values,
                metric_avg - metric_std,
                metric_avg + metric_std,
                alpha=0.2
            )

            plt.plot(
                covariate_values,
                metric_avg,
                label=f'{format_string(comparison_name)}: {format_string(model_name)}'
            )

        plt.subplots_adjust(bottom=0.15)
        plt.subplots_adjust(left=0.15)
        plt.xlabel(format_string(covariate_name))
        plt.ylabel(format_string(metric_name))
        plt.title(f'{format_string(metric_name)} vs {format_string(covariate_name)} (averaged over seeds)')
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
