import itertools
import wandb
from train import run_training
from src.utils import load_config


def get_parameter_combinations(config, prefix=""):
    params = {}

    for key, value in config.items():
        new_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            nested_params = get_parameter_combinations(value, new_key)
            params.update(nested_params)
        else:
            params[new_key] = value

    return params


model = 'vansca'
schedule = 'schedule'
# todo: run with configuration like train.py, argparse

schedule_config = load_config(f'{model}-{schedule}')

param_dict = get_parameter_combinations(schedule_config)

param_keys = list(param_dict.keys())
param_values = list(param_dict.values())

parameter_combinations = list(itertools.product(*param_values))

for param_combination in parameter_combinations:
    kwargs = dict(zip(param_keys, param_combination))

    print(f"Training '{model}' with parameters: {kwargs}")

    run_training(model, schedule, **kwargs)
    wandb.finish()

# todo: fix order of training, seed last
# todo: make it display link to wandb even with wandb.finish() command
# todo: Log or handle results
# print(f"Training complete with true_A: {true_A}, est_A: {est_A}")
# todo: https://pytorch-lightning.readthedocs.io/en/0.9.0/hyperparameters.html
