import os
import json
from tabulate import tabulate
from src.utils.wandb_tools import login_wandb
from src.utils.utils import logging_setup, tabulate_dict
from src.helpers.sweep_analyzer import SweepAnalyzer


def analyze_sweep(experiment, sweep_id, metric=None, covariate=None, comparison=None, save=True, save_dir=None):
    try:
        analyzer = SweepAnalyzer(experiment, sweep_id)
    except FileNotFoundError as e:
        print(e)
        return

    data = analyzer.extract_metrics(metric=metric, covariate=covariate, comparison=comparison)
    averaged_data = analyzer.average_seeds(data)
    analyzer.plot_metric(averaged_data, save=save, show=save, save_dir=save_dir)

    if save:
        analyzer.save_comparison_data(save_dir=save_dir)

    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
    metrics_path = f"{project_root}experiments/{experiment}/results/sweep-{sweep_id}/sweep_data.json"

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)

    for run_id, content in data.items():
        if run_id in metrics_data:
            for metric_name, metric_value in metrics_data[run_id]["metrics"].items():
                content.setdefault("values", {}).setdefault(metric_name, []).append(metric_value)

    table = tabulate_dict(analyzer.average_seeds(data))

    print(f"The {metric.replace('_', ' ')} for different {covariate.replace('_', ' ')}, averaged over random seeds:")
    print(tabulate(table, headers="keys", tablefmt="grid"))
    return table, data


if __name__ == "__main__":
    experiment = "hyperspectral"
    sweep_id = "fphc1rbk"
    logging_setup()
    login_wandb(experiment)

    metrics_to_analyze = [
        ("subspace_distance", "snr"),
        ("validation_loss", "dataset_size"),
        ("latent_mse", "latent_dim"),
        ("_runtime", "dataset_size")
    ]
    # metrics_to_analyze = [
    #     ("psnr", "snr")
    # ]

    for metric, covariate in metrics_to_analyze:
        analyze_sweep(experiment, sweep_id, metric=metric, covariate=covariate, comparison="model_name")


# todo: adjust styling and sizing for plots
# todo: save tables to latex
# todo: unique definition of run path for all calls via wandb tools
# todo: fix directories, run from project root, define all in utils file (enviroment variables?)

# todo: instead of h1, h2, ... use depth and width, make both, if not depth, width read h1, h2...
# todo: run analyze sweep in the sweep_run, run explore_model best for hyperparameter search
# todo: min 1500 epochs, set in config

# fixme: unify module structures and make base classes for all modules
# fixme: test single runs for SNRs and models I want to do, make sure I get results
# fixme: make pipeline for automatic run with hyperparameter search (special config)
# fixme: make a container and run on server
# fixme: notebooks for exploring models and analyzing sweeps
#  (scripts do standard extraction, pandas analysis is in notebooks)
# fixme: main readme describing scripts usage and what they do, as well as notebooks
# fixme: experiment docs describing setups, metrics, datasets, models and results
# fixme: clean up the code (when finished, copy to template before cleaning todos)
