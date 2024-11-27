import json
import os
import numpy as np
import glob


from src.utils.utils import init_plot


class RunAnalyzer:
    def __init__(self, experiment, run_id):
        self.experiment = experiment
        self.run_id = run_id
        project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
        self.output_dir = f"{project_root}experiments/{experiment}/predictions/{self.run_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.run_data = None
        self._fetch_data()

    def extract_metrics(self, metric="latent_mse", covariate="snr"):
        extracted_data = {
            'metric': {'name': metric, 'value': None},
            'covariate': {'name': covariate, 'value': None}
        }
        if self.run_data:
            metric_value = self.run_data['metrics'].get(metric)
            if isinstance(metric_value, dict) and 'max' in metric_value:
                metric_value = metric_value['max']
            extracted_data['metric']['value'] = metric_value
            extracted_data['covariate']['value'] = self.run_data['config'].get(covariate)
        return extracted_data

    def plot_metric(self, metric="latent_mse", save=True, save_dir=None):
        plt = init_plot()
        A4_WIDTH = 8.27

        if not self.run_data:
            print("No data available for plotting.")
            return
        metric_values = np.array(self.run_data["data"].get(metric, []))
        steps = np.array(self.run_data["data"].get("_step", range(len(metric_values))))
        valid_indices = ~np.isnan(metric_values)
        metric_values, steps = metric_values[valid_indices], steps[valid_indices]

        plt.figure(figsize=(A4_WIDTH/2, 3))
        plt.plot(steps, metric_values, marker='o', linestyle='-', label=f'Run {self.run_id}')
        plt.xlabel('Steps' if "_step" in self.run_data["data"] else 'Index')
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.title(f'Metric History for {metric} (Run {self.run_id})')
        plt.legend()
        plt.grid(True)

        if save:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            save_dir = save_dir or f'experiments/{self.experiment}/results/run-{self.run_id}'
            plt.savefig(os.path.join(project_root, save_dir, f"{metric}-history.png"))
            print(f"Saved metric history plot to {os.path.join(project_root, save_dir, f'{metric}-history.png')}")
        plt.close()

    def _fetch_data(self):
        project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
        results_dir = f"{project_root}experiments/{self.experiment}/results"
        sweep_dirs = glob.glob(f"{results_dir}/sweep-*/sweep_data.json", recursive=True)

        if not sweep_dirs:
            raise FileNotFoundError(f"File 'sweep_data.json' not found in {results_dir}")

        for sweep_file in sweep_dirs:
            with open(sweep_file, 'r') as f:
                sweep_data = json.load(f)
                if self.run_id in sweep_data:
                    self.run_data = sweep_data[self.run_id]
                    # print(f"Found history data for run ID {self.run_id} in '{sweep_file}'")
                    return
        raise FileNotFoundError(f"Run ID {self.run_id} not found in any 'sweep_data.json' files in {results_dir}")

    def save_data(self, save_dir=None, metric="latent_mse", covariate="snr"):
        extracted_data = self.extract_metrics(metric=metric, covariate=covariate)
        summary_file_name = f"{metric}-{covariate}.json"
        save_dir = save_dir or self.output_dir
        with open(os.path.join(save_dir, summary_file_name), "w") as f:
            json.dump(extracted_data, f, indent=4)
        print(f"Saved run summary to {os.path.join(save_dir, summary_file_name)}")

    def plot_training_history(self, metric_key="validation_loss"):
        plt = init_plot()
        A4_WIDTH = 8.27

        if self.run_data and "data" in self.run_data and metric_key in self.run_data["data"]:
            metric_values = np.array(self.run_data["data"][metric_key])
            steps = np.array(self.run_data["data"].get("_step", range(len(metric_values))))
            valid_indices = ~np.isnan(metric_values)
            metric_values, steps = metric_values[valid_indices], steps[valid_indices]

            plt.figure(figsize=(A4_WIDTH/2, 3))
            plt.plot(
                steps,
                metric_values,
                marker='o',
                markersize=4,
                linestyle='-',
                label=f'Run {self.run_id}'
            )
            plt.xlabel('Steps' if "_step" in self.run_data["data"] else 'Index')
            plt.ylabel(metric_key.replace("_", " ").capitalize())
            plt.title(f'Training History for {metric_key}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            save_dir = f'experiments/{self.experiment}/predictions/{self.run_id}'
            plt.savefig(os.path.join(project_root, save_dir, f"training-history-{metric_key}.png"))
            print(f"Saved {metric_key} training history plot to {os.path.join(project_root, save_dir, f'training-history-{metric_key}.png')}")
            plt.show()
            plt.close()