from src.helpers.sweep_analyzer import SweepAnalyzer
import numpy as np
from tabulate import tabulate


def analyze_sweep(experiment, sweep_id):
    experiment_analyzer = SweepAnalyzer(experiment, sweep_id)

    data = experiment_analyzer.extract_metrics(
        metric="latent_mse", covariate="snr", comparison="model_name"
    )
    averaged_data = experiment_analyzer.average_seeds(data)
    experiment_analyzer.plot_metric(averaged_data)

    print("Sweep results:")
    table = []
    for data_dict in averaged_data:
        row = {}
        for key, value in data_dict.items():
            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                value = value[0]
            row[key] = value
        table.append(row)

    # Print as table
    print(tabulate(table, headers="keys", tablefmt="grid"))


if __name__ == "__main__":
    experiment = "synthetic_data"
    sweep_id = "c9mus55a"

    analyze_sweep(experiment, sweep_id)

# todo: covariate, metric, comparison markers to sweep dataset
