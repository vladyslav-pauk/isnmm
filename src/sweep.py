import itertools
import wandb
import argparse

from train import train_model
from src.helpers.utils import load_experiment_config, login_wandb, init_logger
from src.generate_data import initialize_data_model


def run():

    model_name, dataset_name, config = init_wandb(experiment)
    logger = init_logger(experiment_name=experiment, model=model_name)
    # model_config = load_experiment_config(experiment, model_name)
    # data_config = load_experiment_config(experiment, dataset_name)

    # Get parameter combinations from model's schedule configuration
    # param_dict = get_parameter_combinations(model_config["schedule"]["parameters"])
    # param_keys = list(param_dict.keys())
    # param_values = list(param_dict.values())
    # parameter_combinations = list(itertools.product(*param_values))

    # Iterate through parameter combinations and train model
    # for param_combination in parameter_combinations:
    # kwargs = dict(zip(param_keys, param_combination))

    print("--- New run ---")
    print(f"Dataset '{dataset_name}':")
    data_model = initialize_data_model(**config)
    data_model.sample()
    data_model.save_data()

    print(f"Model '{model_name}':")
    train_model(logger=logger, **config)


def init_wandb(experiment):
    import os
    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0].split("/")[-2]
    wandb.init(
        entity=project_root,
        project=experiment,
        # group=model,
        # tags=[model],
        # save_dir=f"../models/{model}",
        # config={}
    )
    config = wandb.config
    model_name = config.model_name
    dataset_name = config.data_model_name
    wandb.finish()
    return model_name, dataset_name, config


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='train_schedule',
        help='Experiment name (e.g., simplex_recovery)'
    )
    parser.add_argument(
        '--sweep',
        type=str,
        default='sweep',
        help='Sweep name (e.g., sweep)'
    )
    # parser.add_argument(
    #     '--data',
    #     nargs='+', default='lmm',
    #     help='List of datasets separated by space (e.g., lmm)'
    # )
    # parser.add_argument(
    #     '--models',
    #     nargs='+', default='vasca',
    #     help='List of models separated by space (e.g., vasca)'
    # )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()

    experiment = args.experiment
    sweep = args.sweep
    sweep_config = load_experiment_config(experiment, sweep)

    print(f"Experiment '{experiment}'")
    login_wandb()

    models = "_".join(sweep_config["parameters"]["model_name"]["values"])
    data = "_".join(sweep_config["parameters"]["data_model_name"]["values"])
    sweep_name = f"{sweep}_{models}_{data}"
    sweep_config["name"] = sweep_name

    sweep_id = wandb.sweep(sweep=sweep_config, project=experiment)
    wandb.sweep.name = sweep_name
    wandb.agent(sweep_id, function=run)
