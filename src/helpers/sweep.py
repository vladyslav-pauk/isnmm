import json
import wandb

from src.helpers.wandb import init_wandb, fetch_wandb_sweep
from src.scripts.generate_data import initialize_data_model


class Sweep:
    def __init__(self, sweep_config, trainer):
        self.sweep_config = sweep_config
        self.experiment = sweep_config["parameters"]["experiment_name"]["value"]
        self.train_model = trainer
        self.id = wandb.sweep(sweep=sweep_config, project=self.experiment)
        self.sweep_data = None

    def run(self):
        wandb.sweep.name = self.sweep_config["name"]
        wandb.agent(self.id, function=self.step)

    def step(self):
        model_name, dataset_name, config = init_wandb(self.experiment, self.id)

        print(f"Dataset '{dataset_name}':")
        data_model = initialize_data_model(**config)
        data_model.sample()
        data_model.save_data()

        print(f"Model '{model_name}':")

        self.train_model(**config)

    def fetch_data(self, sweep_id=None, save=True):
        if sweep_id is None:
            sweep_id = self.id
        sweep_data = fetch_wandb_sweep(self.experiment, sweep_id)
        if save:
            self.save_data(self.experiment, sweep_data)

    def save_data(self, experiment, sweep_data):
        import os
        wandb_sweep_dir = f"../experiments/{experiment}/results/sweeps/{self.id}"
        if not os.path.exists(wandb_sweep_dir):
            os.makedirs(wandb_sweep_dir)
        with open(f"{wandb_sweep_dir}/summary.json", "w") as f:
            json.dump(sweep_data, f)