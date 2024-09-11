import argparse

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.modules.utils import init_logger, load_experiment_config
from src.modules.data_module import DataModule
import src.network as network_package
import src.model as model_package


def train_model(experiment_name, config, datamodule, **kwargs):
    model_name = config['model']

    for key, value in kwargs.items():
        if key in config['train']:
            config['train'][key] = value
        elif key in datamodule.config['data']:
            config['data'][key] = value
        else:
            config[key] = value

    if config["train"]["seed"] is not None:
        torch.manual_seed(config["train"]["seed"])

    module = getattr(model_package, model_name)
    network = getattr(network_package, model_name)

    encoder = network.Encoder(
        input_dim=datamodule.config['data']['observed_dim'],
        latent_dim=datamodule.config['data']['latent_dim'],
        hidden_layers=config['encoder']['hidden_dim'],
    )
    decoder = network.Decoder(
        latent_dim=datamodule.config['data']['latent_dim'],
        output_dim=datamodule.config['data']['observed_dim'],
        hidden_layers=config['decoder']['hidden_dim'],
        activation=config['decoder']['activation'],
        sigma=datamodule.dataset.sigma
    )
    model = module.Model(
        encoder=encoder,
        decoder=decoder,
        data_model=datamodule,
        mc_samples=config['train']['mc_samples'],
        lr=config['train']['lr'],
        metrics=config['train']['metrics'],
        monitor=config['train']['monitor'],
        config=config,
        data_config=datamodule.config
    )

    if any(value is not None for value in kwargs.values()):
        run_id = "-".join([f"{key}_{value}" for key, value in kwargs.items() if value is not None])
    else:
        run_id = None
    logger = init_logger(
        experiment=experiment_name,
        model=model_name,
        run_id=run_id)
    logger.log_hyperparams(config)
    logger.log_hyperparams(datamodule.config)
    logger.watch(model, log=config['train']['log'])

    early_stopping_callback = EarlyStopping(
        monitor=config['train']['monitor'],
        min_delta=config['train']['tolerance'],
        patience=config['train']['patience']
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=config["train"]["monitor"],
        mode='min',
        save_top_k=1,
        filename=f'best-model-{{epoch:02d}}-{{{config["train"]["monitor"]}:.2f}}',
        verbose=True
    )

    trainer = Trainer(
        max_epochs=config['train']["epochs"],
        logger=logger,
        accelerator=config['train']['accelerator'],
        log_every_n_steps=config['train']['train_log_rate'],
        limit_val_batches=config['train']["epochs"],
        limit_test_batches=1,
        enable_progress_bar=False,
        callbacks=[early_stopping_callback, checkpoint_callback]
    )

    trainer.fit(model, datamodule)

    return logger.experiment.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with specified hyperparameters')
    parser.add_argument('data_name', type=str, help='Name of the dataset (e.g., lmm)')
    parser.add_argument('model_name', type=str, help='Name of the model (e.g., vasca)')
    parser.add_argument('--hyperparameters', type=str, default=None, help='Hyperparameter dictionary')
    args = parser.parse_args()

    data_config = load_experiment_config(args.data_name, 'data')
    training_config = load_experiment_config(args.data_name, args.model_name)

    train_model(
        experiment_name=args.data_name,
        config=training_config,
        datamodule=DataModule(data_config)
    )
