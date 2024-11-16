import argparse
import ast
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import src.modules.data as data_package
import src.model as model_package
from src.modules.callback import EarlyStoppingCallback
from src.utils.config_tools import load_model_config, load_data_config, update_hyperparameters
from src.utils.wandb_tools import init_logger#, login_wandb
import src.experiments as exp_module


def train_model(experiment_name, model_name, **kwargs):

    config = load_model_config(experiment_name, model_name)
    data_config = load_data_config(experiment_name)

    print(f"Dataset '{experiment_name}':")
    data_config = update_hyperparameters(data_config, kwargs)

    print(f"Model '{config['model_name']}':")
    config = update_hyperparameters(config, kwargs)

    if config.get("torch_seed") is not None:
        seed_everything(config.get("torch_seed"), workers=True)

    logger = _setup_logger(experiment_name, config, data_config, kwargs)

    datamodule = _setup_data_module(data_config, config["data_loader"])
    model = _setup_model(config, logger)
    trainer = _setup_trainer(config, logger)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')
    # logger.experiment.finish()

    return logger.experiment.id


def _setup_data_module(data_config, config):
    datamodule_class = getattr(data_package, config["module_name"]).DataModule
    return datamodule_class(data_config, **config)


def _setup_model(config, logger):
    model_module = getattr(model_package, config['model_name'].upper())
    encoder = model_module.Encoder(config=config['encoder'])
    decoder = model_module.Decoder(config=config['decoder'])

    metrics_module = getattr(exp_module, logger._project)
    metrics = metrics_module.ModelMetrics(monitor=config['metric']['name']).eval()

    model = model_module.Model(
        encoder=encoder,
        decoder=decoder,
        optimizer_config=config['optimizer'],
        model_config=config['model'],
        metrics=metrics
    )
    model.save_hyperparameters(config)
    logger.watch(model, log='parameters')
    return model


def _setup_trainer(config, logger):

    early_stopping_callback = EarlyStoppingCallback(
        monitor=config['metric']['name'],
        mode=config['metric']['goal'][:3],
        **config['early_stopping']
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'../experiments/{logger._project}/checkpoints/{logger.experiment.id}',
        filename=f'{{epoch:02d}}-{{{config["metric"]["name"]}:.2f}}',
        monitor=config['metric']['name'],
        mode=config['metric']['goal'][:3],
        **config['checkpoint']
    )

    trainer = Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=logger,
        **config['trainer']
    )

    return trainer


def _setup_logger(experiment_name, config, data_config, kwargs):
    logger = init_logger(
        experiment_name=experiment_name,
        config=kwargs,
        sweep_id=os.getenv('SWEEP_ID')
    )
    logger.log_hyperparams({
        'config': config,
        'data_config': data_config,
    })
    return logger


# if __name__ == "__main__":
#
#     parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters')
#     parser.add_argument('experiment_name', type=str, help='Name of the experiment (e.g., simplex_recovery)')
#     parser.add_argument('data_name', type=str, help='Name of the dataset (e.g., lmm)')
#     parser.add_argument('model_name', type=str, help='Name of the model (e.g., vasca)')
#     parser.add_argument('--hyperparameters', type=str, default=None, help='Hyperparameter dictionary')
#     args = parser.parse_args()
#
#     if args.hyperparameters:
#         hyperparameters = ast.literal_eval(args.hyperparameters)
#     else:
#         hyperparameters = {}
#
#     login_wandb(args.experiment_name)
#
#     train_model(
#         experiment_name=args.experiment_name,
#         model_name=args.model_name,
#         **hyperparameters
#     )
