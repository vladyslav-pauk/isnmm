import os
import json
import wandb
import shutil


from src.helpers.wandb_tools import init_run, fetch_wandb_sweep, login_wandb
from src.scripts.generate_data import initialize_data_model


class Sweep:
    def __init__(self, sweep_config, trainer):
        self.experiment = sweep_config["parameters"]["experiment_name"]["value"]
        self.train_model = trainer

        self.sweep_data = None
        login_wandb(self.experiment)
        self.id = wandb.sweep(sweep=sweep_config, project=self.experiment)

    def run(self):
        wandb.agent(self.id, function=self.step)

        path_to_remove = f"{os.path.dirname(os.path.abspath(__file__)).split('src')[0]}experiments/{self.experiment}/results/nisca"
        if os.path.exists(path_to_remove):
            print(f"Removing path: {path_to_remove}")
            shutil.rmtree(path_to_remove, ignore_errors=False)
        else:
            print(f"Path does not exist: {path_to_remove}")

    def step(self):

        config = init_run(self.experiment)

        data_model = initialize_data_model(**config)
        data_model.sample()
        data_model.save_data()

        self.train_model(**config)

    def fetch_data(self, sweep_id=None, save=True):
        if sweep_id is None:
            sweep_id = self.id
        sweep_data = fetch_wandb_sweep(self.experiment, sweep_id)
        if save:
            self.save_data(sweep_data)
        return sweep_data
        # shutil.rmtree(f"{os.path.dirname(os.path.abspath(__file__)).split("src")[0]}models/nisca/", ignore_errors=True)

    def save_data(self, sweep_data):
        save_dir = f"../experiments/{self.experiment}/results/sweeps"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(f"{save_dir}/{self.id}.json", "w") as f:
            json.dump(sweep_data, f)