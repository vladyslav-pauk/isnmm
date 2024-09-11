import itertools
import wandb
import argparse

from train import train_model
from src.modules.utils import load_experiment_config, get_parameter_combinations
from src.modules.data_module import DataModule

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='nmm', help='Experiment name  (e.g., nmm)')
    parser.add_argument('--models', nargs='+', default='vasca', help='List of models separated by space (e.g., vasca)')
    args = parser.parse_args()

    experiment = args.experiment
    models = args.models

    data_config = load_experiment_config(experiment, 'data')
    datamodule = DataModule(data_config)

    print(f"Starting experiment '{experiment}'")

    for model in models:
        config = load_experiment_config(experiment, model)
        schedule_config = config["schedule"]

        param_dict = get_parameter_combinations(schedule_config)

        param_keys = list(param_dict.keys())
        param_values = list(param_dict.values())

        parameter_combinations = list(itertools.product(*param_values))

        for param_combination in parameter_combinations:
            kwargs = dict(zip(param_keys, param_combination))

            print(f"Training '{model}' with parameters: \n\t{kwargs}")

            for iteration in range(schedule_config['repeats']):
                train_model(experiment, config, datamodule, **kwargs)

            wandb.finish()


# todo: run with configuration like train.py, argparse
# todo: group runs by same hyperparameter setting
# todo: fix order of training, seed last
# todo: make it display link to wandb even with wandb.finish() command
# todo: Log or handle results
# print(f"Training complete with true_A: {true_A}, est_A: {est_A}")
# todo: https://pytorch-lightning.readthedocs.io/en/0.9.0/hyperparameters.html
