import os
import json
import numpy as np
from tabulate import tabulate

from src.utils.wandb_tools import login_wandb

from src.helpers.sweep_analyzer import SweepAnalyzer


def analyze_sweep(experiment, sweep_id, metric="latent_mse", covariate="snr", comparison="model_name", save=True, save_dir=None):
    experiment_analyzer = SweepAnalyzer(experiment, sweep_id)

    # Step 1: Extract the specified metric
    data = experiment_analyzer.extract_metrics(
        metric=metric, covariate=covariate, comparison=comparison
    )
    averaged_data = experiment_analyzer.average_seeds(data)
    experiment_analyzer.plot_metric(averaged_data, save=save, save_dir=save_dir)
    if save:
        experiment_analyzer.save_data(save_dir=save_dir, metric=metric, covariate=covariate, comparison=comparison)

    # Load and process metrics data
    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
    metrics_file_path = f"{project_root}experiments/{experiment}/results/{sweep_id}/sweep_data.json"
    if os.path.exists(metrics_file_path):
        with open(metrics_file_path, 'r') as f:
            metrics_data = json.load(f)
    else:
        raise FileNotFoundError(f"Metrics file not found at {metrics_file_path}")

    for run_id, content in data.items():
        if run_id in metrics_data:
            for metric_name, metric_value in metrics_data[run_id]["metrics"].items():
                if "values" not in data[run_id]:
                    data[run_id]["values"] = {}
                if metric_name not in data[run_id]["values"]:
                    data[run_id]["values"][metric_name] = []
                data[run_id]["values"][metric_name].append(metric_value)

    averaged_data = experiment_analyzer.average_seeds(data)

    table = tabulate_dict(averaged_data)

    print(f"The {metric.replace('_', ' ')} for different {covariate.replace('_', ' ')}, averaged over random seeds:")
    print(tabulate(table, headers="keys", tablefmt="grid"))

    return table, data


def tabulate_dict(data):
    table = []
    for data_dict in data:
        row = {}
        for key, value in data_dict.items():
            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                value = value[0]
            row[key] = value
        table.append(row)
    return table


if __name__ == "__main__":
    experiment = "synthetic_data"
    sweep_id = "n4iaxyyh"

    login_wandb(experiment)
    analyze_sweep(
        experiment, sweep_id, metric="subspace_distance", covariate="snr", comparison="model_name"
    )

    analyze_sweep(
        experiment, sweep_id, metric="latent_mse", covariate="latent_dim", comparison="model_name"
    )

    analyze_sweep(
        experiment, sweep_id, metric="_runtime", covariate="dataset_size", comparison="model_name"
    )


# todo: save tables to latex

# fixme: train history plots

# fixme: finish all sweep configs
# fixme: test single runs for SNRs and models I want to do, make sure I get results
# fixme: fine tune scheduling and early stopping
# fixme: make pipeline for automatic run with hyperparameter search (special config)
# fixme: make a container and run on server
# fixme: notebooks for exploring models and analyzing sweeps
#  (scripts do standard extraction, pandas analysis is in notebooks)
# fixme: main readme describing scripts usage and what they do, as well as notebooks
# fixme: experiment docs describing setups, metrics, datasets, models and results
# fixme: clean up the code (when finished, copy to template before cleaning todos)
