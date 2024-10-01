import argparse
import ast
# import logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import src.modules.data as data_package
import src.model as model_package
from src.helpers.callbacks import EarlyStoppingCallback
from src.helpers.utils import init_logger, load_experiment_config, unflatten_dict, hash_name


def train_model(experiment_name, data_model_name, model_name, **kwargs):

    config = load_experiment_config(experiment_name, model_name)
    data_config = load_experiment_config(experiment_name, data_model_name)

    _update_hyperparameters(config, kwargs)

    if config.get("torch_seed") is not None:
        seed_everything(config.get("torch_seed"), workers=True)

    logger = _setup_logger(experiment_name, config, kwargs)

    datamodule_instance = setup_data_module(data_config, config["data_loader"])
    model = _setup_model(config, datamodule_instance, logger)
    trainer = _setup_trainer(config, logger)

    # logging.info(f"Training model {model_name} with data model {data_model_name}")
    trainer.fit(model, datamodule_instance)
    trainer.test(model, datamodule_instance)
    # model.summary()
    # logger.experiment.finish()

    return logger.experiment.id


def setup_data_module(data_config, config):
    # logging.info(f"Setting up data module {data_config['module_name']} with data model {data_config['data_model']}")

    datamodule_class = getattr(data_package, config["module_name"]).DataModule
    datamodule_instance = datamodule_class(data_config, **config)

    return datamodule_instance


def _setup_model(config, datamodule, logger):
    model_module = getattr(model_package, config['model_name'])
    encoder = model_module.Encoder(config=config['encoder'])
    decoder = model_module.Decoder(config=config['decoder'])
    model = model_module.Model(
        ground_truth_model=datamodule,
        encoder=encoder,
        decoder=decoder,
        optimizer_config=config['optimizer'],
        model_config=config['model']
    )
    logger.watch(model, log='parameters')
    return model


def _setup_trainer(config, logger):

    early_stopping_callback = EarlyStoppingCallback(
        **config['early_stopping']
    )

    checkpoint_callback = ModelCheckpoint(
        filename=f'best-model-{{epoch:02d}}-{{{config["checkpoint"]["monitor"]}:.2f}}',
        **config['checkpoint']
    )

    trainer = Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=logger,
        **config['trainer']
    )

    return trainer


def _setup_logger(experiment_name, config, kwargs):

    logger = init_logger(
        experiment_name=experiment_name,
        model=config['model_name'],
        run_name=hash_name(kwargs)
    )
    logger.log_hyperparams({
        'config': config
    })
    return logger


def _update_hyperparameters(config, kwargs):
    unflattened_kwargs = unflatten_dict(kwargs)
    data_config = config.get('data', {})
    for key, value in unflattened_kwargs.items():
        if key in config:
            config[key].update(value)
        if key in data_config:
            data_config[key].update(value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment (e.g., simplex_recovery)')
    parser.add_argument('data_name', type=str, help='Name of the dataset (e.g., lmm)')
    parser.add_argument('model_name', type=str, help='Name of the model (e.g., vasca)')
    parser.add_argument('--hyperparameters', type=str, default=None, help='Hyperparameter dictionary')
    args = parser.parse_args()

    if args.hyperparameters:
        hyperparameters = ast.literal_eval(args.hyperparameters)
    else:
        hyperparameters = {}

    train_model(
        experiment_name=args.experiment_name,
        data_model_name=args.data_name,
        model_name=args.model_name,
        **hyperparameters
    )
