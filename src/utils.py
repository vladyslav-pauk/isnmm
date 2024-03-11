import os
import wandb
from pytorch_lightning.loggers import WandbLogger


def init_logger(project=None, experiment=None, id=None):
    os.environ["WANDB_API_KEY"] = "fcf64607eeb9e076d3cbfdfe0ea3532621753d78"
    os.environ['WANDB_SILENT'] = 'true'
    wandb.login()

    logger = WandbLogger(
        project=experiment,
        entity=project,
        save_dir="models",
        log_model=True)

    if id:
        logger.experiment.id = id
    else:
        logger.experiment.name = logger.experiment.id

    return logger
