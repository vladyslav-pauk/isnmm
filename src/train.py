import argparse
import logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import isolate_rng

import src.data as data_package
import src.model as model_package
from src.helpers.utils import init_logger, load_experiment_config, unflatten_dict


def train_model(experiment_name, model_name, data_model_name, **kwargs):

    training_config = load_experiment_config(experiment_name, model_name)
    data_config = load_experiment_config(experiment_name, data_model_name)
    _update_hyperparameters(training_config, data_config, kwargs)

    if training_config.get("torch_seed"):
        seed_everything(training_config.get("torch_seed"), workers=True)
    logger = _setup_logger(experiment_name, training_config, data_config, kwargs)

    datamodule_instance = setup_data_module(data_config)
    model = _setup_model(training_config, datamodule_instance, logger)
    trainer = _setup_trainer(training_config, logger)

    logging.info(f"Training model {model_name} with data model {data_model_name}")
    trainer.fit(model, datamodule_instance)
    # logger.experiment.finish()

    return logger.experiment.id


def setup_data_module(data_config):
    logging.info(f"Setting up data module {data_config['module_name']} with data model {data_config['data_model']}")
    if data_config["seed"]:
        with isolate_rng():
            seed_everything(data_config["seed"], workers=True)
            datamodule_class = getattr(data_package, data_config["module_name"]).DataModule
            datamodule_instance = datamodule_class(data_config["data_model"], **data_config["dataset"])
    return datamodule_instance


def _setup_model(training_config, datamodule, logger):

    model_module = getattr(model_package, training_config['model_name'])

    encoder = model_module.Encoder(
        input_dim=datamodule.observed_dim,
        latent_dim=datamodule.latent_dim,
        **training_config['encoder']
    )
    decoder = model_module.Decoder(
        latent_dim=datamodule.latent_dim,
        output_dim=datamodule.observed_dim,
        **training_config['decoder']
    )
    model = model_module.Model(
        ground_truth_model=datamodule,
        encoder=encoder,
        decoder=decoder,
        train_config=training_config['train']
    )

    logger.watch(model, log=training_config['logger_watch'])
    return model


def _setup_trainer(config, logger):

    early_stopping_callback = EarlyStopping(
        **config["train"]["monitor"],
        **config['early_stopping']
    )

    checkpoint_callback = ModelCheckpoint(
        filename=f'best-model-{{epoch:02d}}-{{{config["train"]["monitor"]["monitor"]}:.2f}}',
        **config["train"]["monitor"],
        **config['checkpoint']
    )
    trainer = Trainer(
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=logger,
        **config['trainer']
    )

    return trainer


def _setup_logger(experiment_name, training_config, data_config, kwargs):

    import hashlib
    if kwargs:
        kwargs_str = str(sorted(kwargs.items()))
        run_name = hashlib.md5(kwargs_str.encode()).hexdigest()
    else:
        run_name = None
    # if any(value is not None for value in kwargs.values()):
    #     run_name = "-".join([f"{key}_{value}" for key, value in kwargs.items() if value is not None])
    # else:
    #     run_name = None

    logger = init_logger(
        experiment_name=experiment_name,
        model=training_config['model_name'],
        run_name=run_name
    )
    logger.log_hyperparams({
        'training_config': training_config,
        'data_config': data_config
    })
    return logger


def _update_hyperparameters(training_config, data_config, kwargs):
    unflattened_kwargs = unflatten_dict(kwargs)
    for key, value in unflattened_kwargs.items():
        if key in training_config:
            training_config[key].update(value)
        if key in data_config:
            data_config[key].update(value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment (e.g., simplex_recovery)')
    parser.add_argument('data_name', type=str, help='Name of the dataset (e.g., lmm)')
    parser.add_argument('model_name', type=str, help='Name of the model (e.g., vasca)')
    parser.add_argument('--hyperparameters', type=str, default=None, help='Hyperparameter dictionary')
    args = parser.parse_args()

    import ast
    if args.hyperparameters:
        hyperparameters = ast.literal_eval(args.hyperparameters)
    else:
        hyperparameters = {}
    # import json
    # if args.hyperparameters:
    #     print(args.hyperparameters)
    #     hyperparameters = json.loads(args.hyperparameters)
    # else:
    #     hyperparameters = {}

    train_model(
        experiment_name=args.experiment_name,
        data_model_name=args.data_name,
        model_name=args.model_name,
        **hyperparameters
    )

