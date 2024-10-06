import wandb
from pytorch_lightning.loggers import WandbLogger
from src.helpers.utils import flatten_dict
import os


def login_wandb():
    os.environ["WANDB_API_KEY"] = "fcf64607eeb9e076d3cbfdfe0ea3532621753d78"
    os.environ['WANDB_SILENT'] = 'true'
    wandb.require("core")
    wandb.login()

    # log_format = "%(asctime)s - %(levelname)s - %(message)s"
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format=log_format,
    #     datefmt="%Y-%m-%d %H:%M:%S",
    # )
    # logging.basicConfig(level=logging.INFO)
    # logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*GPU available but not used.*",
        category=UserWarning
    )

def init_logger(experiment_name=None, model=None, run_name=None):

    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0].split("/")[-2]

    wandb_logger = WandbLogger(
        entity=project_root,
        project=experiment_name,
        group=model,
        name=run_name,
        # tags=[model],
        save_dir=f"../models/{model}",
        log_model=True,
        resume="allow",
        # config={}
    )

    return wandb_logger


def init_wandb(experiment):
    import os
    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
    project_name = project_root.split("/")[-2]

    wandb.init(
        entity=project_name,
        project=experiment,
        dir=project_root
    )
    config = wandb.config
    model_name = config.model_name
    dataset_name = config.data_model_name
    wandb.finish()
    return model_name, dataset_name, config


def fetch_wandb_sweep(project_name, sweep_id):
    import os
    project_root = os.path.dirname(os.path.abspath(__file__)).split("src")[0]
    api = wandb.Api()
    sweep = api.sweep(f"{project_root.split("/")[-2]}/{project_name}/{sweep_id}")
    runs = sweep.runs

    all_runs_data = {}
    for run in runs:
        config = run.config
        metrics = run.summary._json_dict
        all_runs_data[run.id] = {
            "metrics": metrics,
            "config": config
        }
    return all_runs_data
