import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.utils.wandb_tools import fetch_wandb_sweep
from src.utils.utils import font_style, format_string, init_plot


class SweepAnalyzer:
    def __init__(self, experiment, sweep_id):
        self.experiment = experiment
        self.sweep_id = sweep_id
        project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
        self.output_dir = f"{project_root}experiments/{experiment}/results/sweep-{self.sweep_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.sweep_data = self._fetch_data()
        self.extracted_data = None

    def extract_metrics(self, metric=None, covariate=None, comparison=None):
        # if not metric:
        #     metric = next(iter(self.sweep_data.values()))['config']['metric']['name']
        #
        # if not covariate:
        #     covariate = 'snr'
        #
        # if not comparison:
        #     comparison = 'model_name'

        refactored_data = {
            'covariate': {'name': covariate, 'values': defaultdict(list)},
            'metric': {'name': metric, 'values': defaultdict(list)},
            'comparison': {'name': comparison, 'values': defaultdict(list)},
            'run_ids': defaultdict(list)
        }

        for run_id, content in self.sweep_data.items():
            seed = content['config']['torch_seed']
            metric_value = content['metrics'].get(metric)
            if isinstance(metric_value, dict) and 'max' in metric_value:
                metric_value = metric_value['max']

            refactored_data['covariate']['values'][content['config'][comparison]].append(content['config'][covariate])
            refactored_data['metric']['values'][content['config'][comparison]].append(metric_value)
            refactored_data['comparison']['values'][content['config'][comparison]].append(seed)
            refactored_data['run_ids'][content['config'][comparison]].append(run_id)

        self.extracted_data = refactored_data
        return refactored_data

    def average_seeds(self, refactored_data):
        formatted_data = []
        covariate_name = refactored_data['covariate']['name']
        metric_name = refactored_data['metric']['name']

        for comparison_value, covariate_values in refactored_data['covariate']['values'].items():
            metric_values = np.array(refactored_data['metric']['values'][comparison_value])
            unique_covariates = np.unique(covariate_values)
            averages, stddevs = [], []

            for val in unique_covariates:
                indices = np.where(covariate_values == val)
                averages.append(np.mean(metric_values[indices]))
                stddevs.append(np.std(metric_values[indices]))

            formatted_data.append({
                'model_name': comparison_value,
                covariate_name: unique_covariates,
                f'{metric_name}_avg': np.array(averages),
                f'{metric_name}_std': np.array(stddevs)
            })

        return formatted_data

    def plot_metric(self, averaged_data, save=True, show=False, save_dir=None):
        init_plot()

        comparison_name = list(averaged_data[0].keys())[0]
        covariate_name = list(averaged_data[0].keys())[1]
        metric_name = list(averaged_data[0].keys())[2].replace('_avg', '')

        plt.figure(figsize=(10, 6))
        for data in averaged_data:
            plt.fill_between(
                data[covariate_name],
                data[f'{metric_name}_avg'] - data[f'{metric_name}_std'],
                data[f'{metric_name}_avg'] + data[f'{metric_name}_std'],
                alpha=0.2
            )
            plt.plot(data[covariate_name], data[f'{metric_name}_avg'], label=f'{format_string(comparison_name)}: {format_string(data["model_name"])}')

        plt.xlabel(format_string(covariate_name))
        plt.ylabel(format_string(metric_name))
        plt.title(f'{format_string(metric_name)} vs {format_string(covariate_name)} (averaged over seeds)')
        plt.legend()

        if save:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            save_dir = save_dir or f'experiments/{self.experiment}/results/sweep-{self.sweep_id}'
            plt.savefig(os.path.join(project_root, save_dir, f"{metric_name}-{covariate_name}.png"))
            print(f"Saved {metric_name} vs {covariate_name} plot to {os.path.join(project_root, save_dir, f'{metric_name}-{covariate_name}.png')}")

        if show:
            plt.show()

        plt.close()

    def _fetch_data(self, wandb=False):
        if wandb:
            return fetch_wandb_sweep(self.experiment, self.sweep_id)

        project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
        data_path = f"{project_root}experiments/{self.experiment}/results/sweep-{self.sweep_id}/sweep_data.json"
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                return json.load(f)

        raise FileNotFoundError(f"Metrics file not found at {data_path}")

    def save_comparison_data(self, save_dir=None):
        extracted_data = self.extracted_data
        averaged_data = self.average_seeds(extracted_data)
        averaged_data = json.loads(json.dumps(averaged_data, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))

        metric_name = extracted_data['metric']['name']
        covariate_name = extracted_data['covariate']['name']
        summary_file_name = f"{metric_name}-{covariate_name}.json"

        if save_dir:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            self.output_dir = os.path.join(project_root, save_dir)

        with open(os.path.join(self.output_dir, summary_file_name), "w") as f:
            json.dump(averaged_data, f, indent=4)

        print(f"Saved sweep summary to {os.path.join(self.output_dir, summary_file_name)}")