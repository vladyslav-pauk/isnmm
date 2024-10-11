import os
import json
import wandb
import shutil


from src.helpers.wandb_tools import init_run, fetch_wandb_sweep, login_wandb
from src.scripts.generate_data import initialize_data_model


class Sweep:
    def __init__(self, sweep_config, trainer):
        self.sweep_config = sweep_config
        self.experiment = sweep_config["parameters"]["experiment_name"]["value"]
        self.train_model = trainer

        self.sweep_data = None
        self.experiment_dir = f"../experiments/{self.experiment}/results"

        login_wandb(self.experiment_dir)
        self.id = wandb.sweep(sweep=self.sweep_config, project=self.experiment)

    def run(self):
        # models = "_".join(sweep_config["parameters"]["model_name"]["values"])
        # data = "_".join(sweep_config["parameters"]["data_model_name"]["values"])
        # wandb.sweep.name = f'{self.sweep_config["name"]}-{self.id}'
        wandb.agent(self.id, function=self.step)

    def step(self):
        config = init_run(self.experiment, self.experiment_dir)

        print(f"Dataset:")
        data_model = initialize_data_model(**config)
        data_model.sample()
        data_model.save_data()

        print(f"Model '{config['model_name']}':")
        self.train_model(**config)

    def fetch_data(self, sweep_id=None, save=True):
        if sweep_id is None:
            sweep_id = self.id
        sweep_data = fetch_wandb_sweep(self.experiment, sweep_id)
        if save:
            self.save_data(sweep_data)
        # shutil.rmtree(f"{os.path.dirname(os.path.abspath(__file__)).split("src")[0]}models/nisca/", ignore_errors=True)

    def save_data(self, sweep_data):
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        with open(f"{self.experiment_dir}/sweep-{self.id}.json", "w") as f:
            json.dump(sweep_data, f)