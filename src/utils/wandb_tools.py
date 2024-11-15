import os
import shutil
import wandb
from pytorch_lightning.loggers import WandbLogger


def set_wandb_dir(directory=None):
    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
    wandb_dir = f'{project_root}{directory}'
    os.environ["WANDB_DIR"] = os.path.abspath(wandb_dir)


def login_wandb(experiment=None):
    set_wandb_dir(f'experiments/{experiment}')

    os.environ["WANDB_API_KEY"] = "fcf64607eeb9e076d3cbfdfe0ea3532621753d78"
    os.environ['WANDB_SILENT'] = 'true'
    # os.environ['WANDB_DISABLE_SERVICE'] = 'True'

    wandb.require("core")
    wandb.login()

    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*GPU available but not used.*",
        category=UserWarning
    )

# fixme: rename experiment to synthetic_data

def init_run(experiment, sweep_id=''):

    sweep_dir = f"../experiments/{experiment}/wandb/sweep-{sweep_id}"
    # if not os.path.exists(sweep_dir):
    #     os.makedirs(sweep_dir)

    # set_wandb_dir(sweep_dir)
    project_name = os.path.dirname(os.path.abspath(__file__)).split("src")[0].split("/")[-2]

    import wandb
    wandb.init(
        entity=project_name,
        project=experiment,
        dir=sweep_dir
    )

    print(f"--- Run ID: {wandb.run.id} ---")
    config = wandb.config
    config["run_id"] = wandb.run.id
    os.environ["RUN_ID"] = wandb.run.id
    wandb.finish()
    return config

# fixme: values are printed as updated, value is printed before all runs


def init_logger(experiment_name=None, config=None, sweep_id=None):
    # set_wandb_dir('models')

    project_name = os.path.dirname(os.path.abspath(__file__)).split("src")[0].split("/")[-2]

    # import sys
    # sys.exit()
    base_path = os.path.dirname(os.path.abspath(__file__)).split('src')[0]

    path = os.path.join(base_path, f"experiments/{experiment_name}/wandb/sweep-{sweep_id}")

    logger = WandbLogger(
        entity=project_name,
        project=experiment_name,
        name=config["run_id"] if "run_id" in config else None,
        notes="notes for the model",
        resume="allow",
        save_dir=path
    )

    # f'{hyperparameters['snr']}'
    # logger.experiment.tags = list(hyperparameters.keys())
    return logger


def fetch_wandb_sweep(project_name, sweep_id):
    import os
    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]

    api = wandb.Api()
    sweep_address = f"{project_root.split("/")[-2]}/{project_name}/{sweep_id}"
    sweep = api.sweep(sweep_address)
    runs = sweep.runs

    all_runs_data = {}
    for run in runs:
        config = run.config
        metrics = run.summary._json_dict
        data = run.history().to_dict(orient='list')

        all_runs_data[run.id] = {
            "metrics": metrics,
            "config": config,
            "data": data
        }
    return all_runs_data

# task: log time epoch
