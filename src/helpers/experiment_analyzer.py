import json
import os
import matplotlib.pyplot as plt
import numpy as np
import glob


class ExperimentAnalyzer:
    def __init__(self, experiment, run_id):
        self.experiment = experiment
        self.run_id = run_id
        project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
        self.output_dir = f"{project_root}experiments/{experiment}/predictions/{self.run_id}"
        self.output_file = f"run_summary.json"

        os.makedirs(self.output_dir, exist_ok=True)

        self.run_data = None
        self._fetch_data()

    def extract_metrics(self, metric="latent_mse", covariate="snr"):
        # Simplified data extraction for a single run
        extracted_data = {
            'metric': {'name': metric, 'value': None},
            'covariate': {'name': covariate, 'value': None}
        }

        if self.run_data:
            metric_value = self.run_data['metrics'].get(metric, None)
            if isinstance(metric_value, dict) and 'max' in metric_value:
                metric_value = metric_value['max']
            extracted_data['metric']['value'] = metric_value

            covariate_value = self.run_data['config'].get(covariate, None)
            extracted_data['covariate']['value'] = covariate_value

        return extracted_data

    def plot_metric(self, metric="latent_mse", save=True, save_dir=None):
        if not self.run_data:
            print("No data available for plotting.")
            return

        metric_values = self.run_data["data"].get(metric, [])
        steps = self.run_data["data"].get("_step", range(len(metric_values)))

        metric_values = np.array(metric_values)
        steps = np.array(steps)
        valid_indices = ~np.isnan(metric_values)
        metric_values = metric_values[valid_indices]
        steps = steps[valid_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, metric_values, marker='o', linestyle='-', label=f'Run {self.run_id}')
        plt.xlabel('Steps' if "_step" in self.run_data["data"] else 'Index')
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.title(f'Metric History for {metric} (Run {self.run_id})')
        plt.legend()
        plt.grid(True)

        if save:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            if save_dir is None:
                save_dir = f'experiments/{self.experiment}/results/run-{self.run_id}'
            plot_file_name = f"{metric}-history.png"
            plt.savefig(os.path.join(project_root, save_dir, plot_file_name))

        plt.close()

    def _fetch_data(self):
        project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
        results_dir = f"{project_root}experiments/{self.experiment}/results"

        # Search for all sweep directories in the results directory
        sweep_dirs = glob.glob(f"{results_dir}/sweep-*/sweep_data.json", recursive=True)

        if not sweep_dirs:
            raise FileNotFoundError(
                f"No 'sweep_data.json' files found in any 'sweep-*' directories under {results_dir}")

        found = False
        # Iterate over each sweep_data.json file to find the run_id
        for sweep_file in sweep_dirs:
            with open(sweep_file, 'r') as f:
                sweep_data = json.load(f)
                if self.run_id in sweep_data:
                    self.run_data = sweep_data[self.run_id]
                    found = True
                    print(f"Found data for run ID {self.run_id} in '{sweep_file}'")
                    break

        if not found:
            raise FileNotFoundError(f"Run ID {self.run_id} not found in any 'sweep_data.json' files in {results_dir}")

    def save_data(self, save_dir=None, metric="latent_mse", covariate="snr"):
        extracted_data = self.extract_metrics(metric=metric, covariate=covariate)
        summary_file_name = f"{metric}-{covariate}.json"

        if save_dir is not None:
            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            self.output_dir = os.path.join(project_root, save_dir)

        with open(os.path.join(self.output_dir, summary_file_name), "w") as f:
            json.dump(extracted_data, f, indent=4)

        print(f"Saved run summary to {os.path.join(self.output_dir, summary_file_name)}")

    def plot_training_history(self, metric_key="latent_mse"):
        if self.run_data and "data" in self.run_data and metric_key in self.run_data["data"]:
            metric_values = self.run_data["data"][metric_key]
            steps = self.run_data["data"].get("_step", range(len(metric_values)))

            metric_values = np.array(metric_values)
            steps = np.array(steps)
            valid_indices = ~np.isnan(metric_values)
            metric_values = metric_values[valid_indices]
            steps = steps[valid_indices]

            plt.figure(figsize=(10, 6))
            plt.plot(steps, metric_values, marker='o', linestyle='-', label=f'Run {self.run_id}')
            plt.xlabel('Steps' if "_step" in self.run_data["data"] else 'Index')
            plt.ylabel(metric_key.replace("_", " ").capitalize())
            plt.title(f'Training History for {metric_key} (Run {self.run_id})')
            plt.legend()
            plt.grid(True)

            project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
            save_dir = f'experiments/{self.experiment}/predictions/{self.run_id}'
            plot_file_name = f"training-history-{metric_key}.png"
            plt.savefig(os.path.join(project_root, save_dir, plot_file_name))
            plt.show()
            plt.close()

        else:
            pass
            # print(f"Metric '{metric_key}' not found in run data.")