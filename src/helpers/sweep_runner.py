import os
import json
import wandb


from src.utils.wandb_tools import init_run, fetch_wandb_sweep
from src.helpers.generate_data import initialize_data_model


class Sweep:
    def __init__(self, sweep_config, trainer):
        self.experiment = sweep_config["parameters"]["experiment_name"]["value"]
        self.train_model = trainer
        self.sweep_data = None

        self.id = wandb.sweep(sweep=sweep_config, project=self.experiment)
        os.environ["SWEEP_ID"] = self.id

    def run(self):
        wandb.agent(self.id, function=self.step)

    def step(self):
        config = init_run(self.experiment, self.id)

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

    def save_data(self, sweep_data):
        save_dir = f"../experiments/{self.experiment}/results/{self.id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save all data to one file
        save_path = f"{save_dir}/sweep_data.json"
        with open(save_path, "w") as f:
            json.dump(sweep_data, f, indent=2)