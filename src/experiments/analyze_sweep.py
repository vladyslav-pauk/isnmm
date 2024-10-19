import json
import os
from src.helpers.wandb_tools import login_wandb, fetch_wandb_sweep
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

experiment = "simplex_recovery"
sweep_id = "jh0xq8w1"
output_dir = f"../experiments/{experiment}/sweeps"
output_file = f"{sweep_id}.json"

os.makedirs(output_dir, exist_ok=True)

login_wandb()
sweep_data = fetch_wandb_sweep(experiment, sweep_id)

with open(os.path.join(output_dir, output_file), "w") as f:
    json.dump(sweep_data, f, indent=4)

print(f"Saved sweep data to {os.path.join(output_dir, output_file)}")


def extract_metrics(json_data, metric="mixture_mse_db", covariate="snr"):
    data = defaultdict(lambda: defaultdict(list))

    for run_id, content in json_data.items():
        snr = content['config'][covariate]
        metric_value = content['metrics'][metric]
        model_name = content['config']['model_name']
        seed = content['config']['torch_seed']

        data[model_name][covariate].append(snr)
        data[model_name]['metric'].append(metric_value)
        data[model_name]['seed'].append(seed)

    return data


def average_metrics(data, covariate="snr"):
    averaged_data = {}

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


def plot_metric_vs_snr(data, covariate="snr"):
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


data = extract_metrics(sweep_data)
averaged_data = average_metrics(data)
plot_metric_vs_snr(averaged_data)

# fixme: finish plotting given metric and covariate (snr, network_width)
