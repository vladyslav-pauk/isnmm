import wandb
import json
import os
from src.helpers.utils import load_experiment_config, parser
from src.helpers.wandb import login_wandb, init_wandb, fetch_wandb_sweep
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Parameters
experiment = "nonlinearity_removal"
sweep_id = "m8x24hhr"
output_dir = f"../experiments/{experiment}/sweeps"
output_file = f"{sweep_id}.json"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Login and fetch sweep data
login_wandb()
sweep_data = fetch_wandb_sweep(experiment, sweep_id)

# Write all data to a single JSON file
with open(os.path.join(output_dir, output_file), "w") as f:
    json.dump(sweep_data, f, indent=4)

print(f"Saved sweep data to {os.path.join(output_dir, output_file)}")


def extract_metrics(json_data):
    """
    Extracts SNR, h_r_square values, and groups by model and seed.
    Returns a dict of model with the corresponding SNR and metrics.
    """
    data = defaultdict(lambda: defaultdict(list))  # Nested defaultdict to store lists

    for run_id, content in json_data.items():
        snr = content['config']['snr']
        metric = content['metrics']['h_r_square']
        model_name = content['config']['model_name']
        seed = content['config']['torch_seed']

        data[model_name]['snr'].append(snr)
        data[model_name]['metric'].append(metric)
        data[model_name]['seed'].append(seed)

    return data


def average_metrics_by_snr(data):
    """
    Averages metrics by SNR across different seeds for each model.
    """
    averaged_data = {}

    for model, values in data.items():
        snr_values = np.array(values['snr'])
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
            'snr': unique_snrs,
            'metric_avg': np.array(snr_metric_averages),
            'metric_std': np.array(snr_metric_stddev)
        }

    return averaged_data


def plot_metric_vs_snr(data):
    """
    Plot the chosen metric against SNR, averaging over seeds, with shaded variance region.
    """
    plt.figure(figsize=(10, 6))

    for model, values in data.items():
        snr_values = values['snr']
        metric_avg = values['metric_avg']
        metric_std = values['metric_std']

        # Plot shaded region for variance
        plt.fill_between(
            snr_values,
            metric_avg - metric_std,
            metric_avg + metric_std,
            alpha=0.2
        )

        # Plot the mean line
        plt.plot(snr_values, metric_avg, label=f'Model {model}')

    plt.xlabel('SNR')
    plt.ylabel('Metric (h_r_square)')
    plt.title('Metric vs SNR (Averaged over Seeds)')
    plt.legend()
    plt.show()


# Extract data
data = extract_metrics(sweep_data)

# Average metrics by SNR and seed
averaged_data = average_metrics_by_snr(data)

# Plot the results
plot_metric_vs_snr(averaged_data)