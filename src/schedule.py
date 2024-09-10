import itertools
import wandb

from train import run_training
from src.modules.utils import load_config, get_parameter_combinations


experiment = 'nnmm-vasca'
schedule = 'schedule'

schedule_config = load_config(f'{experiment}-{schedule}')

param_dict = get_parameter_combinations(schedule_config)

param_keys = list(param_dict.keys())
param_values = list(param_dict.values())

parameter_combinations = list(itertools.product(*param_values))

for param_combination in parameter_combinations:
    kwargs = dict(zip(param_keys, param_combination))

    print(f"Executing '{experiment}' with parameters: {kwargs}")

    for iter in range(schedule_config['repeats']):
        run_training(experiment, **kwargs)

    wandb.finish()


# todo: run with configuration like train.py, argparse
# todo: group runs by same hyperparameter setting
# todo: fix order of training, seed last
# todo: make it display link to wandb even with wandb.finish() command
# todo: Log or handle results
# print(f"Training complete with true_A: {true_A}, est_A: {est_A}")
# todo: https://pytorch-lightning.readthedocs.io/en/0.9.0/hyperparameters.html
