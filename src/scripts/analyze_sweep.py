from src.helpers.sweep_analyzer import SweepAnalyzer
import numpy as np
from tabulate import tabulate

from src.utils.wandb_tools import login_wandb


def analyze_sweep(experiment, sweep_id, save=True, save_dir=None):
    experiment_analyzer = SweepAnalyzer(experiment, sweep_id)

    data = experiment_analyzer.extract_metrics(
        metric="latent_mse", covariate="snr", comparison="model_name"
    )
    averaged_data = experiment_analyzer.average_seeds(data)
    experiment_analyzer.plot_metric(averaged_data, save=save, save_dir=save_dir)
    if save:
        experiment_analyzer.save_data(save_dir=save_dir)

    table = []
    for data_dict in averaged_data:
        row = {}
        for key, value in data_dict.items():
            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                value = value[0]
            row[key] = value
        table.append(row)

    print(tabulate(table, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    experiment = "synthetic_data"
    sweep_id = "ptxlpyn7"

    login_wandb(experiment)
    analyze_sweep(experiment, sweep_id)

# todo: covariate, metric, comparison markers to sweep dataset
