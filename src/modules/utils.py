import os
import json

import wandb
from pytorch_lightning.loggers import WandbLogger


# def load_config():
#     experiment = sys.argv[1]
#
#     with open(f'experiments/{experiment}.json', 'r') as f:
#         return json.load(f)
#
def load_experiment_config(experiment, config_name):
    with open(f'experiments/{experiment}/{config_name}.json', 'r') as f:
        return json.load(f)


def init_logger(experiment=None, model=None, run_id=None):
    os.environ["WANDB_API_KEY"] = "fcf64607eeb9e076d3cbfdfe0ea3532621753d78"
    os.environ['WANDB_SILENT'] = 'true'
    wandb.require("core")
    wandb.login()

    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0].split("/")[-2]

    logger = WandbLogger(
        entity=project_root,
        project=experiment,
        group=model,
        id=run_id,
        save_dir="models",
        log_model=True,
        resume="allow"
    )

    # if run_id:
    #     logger.experiment.id = run_id
    # else:
    #     logger.experiment.name = logger.experiment.id

    return logger


def get_parameter_combinations(config, prefix=""):
    params = {}

    for key, value in config.items():
        new_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            nested_params = get_parameter_combinations(value, new_key)
            params.update(nested_params)
        elif isinstance(value, list):
            params[new_key] = value

    return params