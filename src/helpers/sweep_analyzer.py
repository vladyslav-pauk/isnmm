import json
import os
from src.utils.wandb_tools import fetch_wandb_sweep#, login_wandb
from src.utils.utils import font_style, format_string
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class SweepAnalyzer:
    def __init__(self, experiment, sweep_id):
        self.experiment = experiment
        self.sweep_id = sweep_id
        project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
        self.output_dir = f"{project_root}experiments/{experiment}/results/sweep-{self.sweep_id}"
        self.output_file = f"sweep_summary.json"
        # todo: fix directories, run from project root

        os.makedirs(self.output_dir, exist_ok=True)

        self.sweep_data = None
        try:
            self._fetch_data()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Metrics file not found for sweep {sweep_id}")

    def extract_metrics(self, metric=None, covariate=None, comparison=None):
        # fixme: fix default metric, covariate, comparison everywhere in extract_metric and analyze_sweep
        refactored_data = {
            'covariate': {'name': covariate, 'values': defaultdict(list)},
            'metric': {'name': metric, 'values': defaultdict(list)},
            'comparison': {'name': comparison, 'values': defaultdict(list)},
            'run_ids': defaultdict(list)  # New section for run IDs
        }
        if not metric:
            metric = self.sweep_data[next(iter(self.sweep_data))]['config']['metric']['name']

        for run_id, content in self.sweep_data.items():
            seed = content['config']['torch_seed']

            metric_value = content['metrics'][metric]
            if isinstance(metric_value, dict):
                if 'max' in metric_value:
                    metric_value = metric_value['max']
                else:
                    raise ValueError(f"Expected 'max' key in metric dictionary, found: {metric_value.keys()}")

            covariate_value = content['config'][covariate]
            comparison_value = content['config'][comparison]

            refactored_data['covariate']['values'][comparison_value].append(covariate_value)
            refactored_data['metric']['values'][comparison_value].append(metric_value)
            refactored_data['comparison']['values'][comparison_value].append(seed)
            refactored_data['run_ids'][comparison_value].append(run_id)  # Add run_id to the refactored data

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

    def plot_metric(self, averaged_data, save=True, save_dir=None):
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

        if save:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            if save_dir is None:
                save_dir = f'experiments/{self.experiment}/results/sweep-{self.sweep_id}'

            plot_file_name = f"{metric_name}-{covariate_name}.png"
            plt.savefig(os.path.join(project_root, save_dir, plot_file_name))

        plt.close()

    def _fetch_data(self, wandb=False):
        if wandb:
            self.sweep_data = fetch_wandb_sweep(self.experiment, self.sweep_id)
        else:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            path_to_data = f"{project_root}experiments/{self.experiment}/results/sweep-{self.sweep_id}/sweep_data.json"
            if os.path.exists(path_to_data):
                with open(path_to_data, 'r') as f:
                    self.sweep_data = json.load(f)
            else:
                raise FileNotFoundError(f"Metrics file not found at {path_to_data}")

    def save_comparison_data(self, save_dir=None, metric=None, covariate=None, comparison=None):
        # Extract metrics and covariates
        extracted_data = self.extract_metrics(metric=metric, covariate=covariate, comparison=comparison)
        averaged_data = self.average_seeds(extracted_data)
        averaged_data = json.loads(
            json.dumps(averaged_data, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        )

        metric_name = extracted_data['metric']['name']
        covariate_name = extracted_data['covariate']['name']
        summary_file_name = f"{metric_name}-{covariate_name}.json"

        if save_dir is not None:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            self.output_dir = os.path.join(project_root, save_dir)

        with open(os.path.join(self.output_dir, summary_file_name), "w") as f:
            json.dump(averaged_data, f, indent=4)

        print(f"Saved sweep summary to {os.path.join(self.output_dir, summary_file_name)}")
