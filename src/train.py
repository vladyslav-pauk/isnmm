from pytorch_lightning import Trainer
import torch
from src.utils import init_logger, load_config
from src.data_module import DataModule
import src.network as network_package
import src.model as model_package
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def run_training(model_name, lr_th=None, lr_ph=None, snr=None, seed=None):
    config = load_config(model_name)

    # Update config with passed hyperparameters
    if lr_th is not None:
        config['train']['lr']['th'] = lr_th
    if lr_ph is not None:
        config['train']['lr']['ph'] = lr_ph
    if snr is not None:
        config['data']['SNR'] = snr

    if seed is not None:
        config['train']['seed'] = seed
    if config["train"]["seed"] is not None:
        torch.manual_seed(config["train"]["seed"])

    # todo: make it implicit, above and name_id

    # Dynamically load model and network
    module = getattr(model_package, model_name)
    network = getattr(network_package, model_name)
    datamodule = DataModule(config)

    encoder = network.Encoder(
        input_dim=config['data']['observed_dim'],
        latent_dim=config['data']['latent_dim'],
        hidden_layers=config['encoder']['hidden_dim'],
    )
    decoder = network.Decoder(
        latent_dim=config['data']['latent_dim'],
        output_dim=config['data']['observed_dim'],
        hidden_layers=config['decoder']['hidden_dim'],
        activation=config['decoder']['activation'],
        sigma=datamodule.dataset.sigma
    )

    model = module.Model(
        encoder=encoder,
        decoder=decoder,
        data_model=datamodule,
        mc_samples=config['train']['mc_samples'],
        lr=config['train']['lr']
    )

    # Initialize logger
    run_id = f"seed_{seed}-snr_{snr}-lr_th_{lr_th}-lr_ph_{lr_ph}"
    logger = init_logger(project=config['project'], experiment=config['experiment'], run_id=run_id)
    logger.log_hyperparams(config)
    logger.watch(model, log=config['train']['log'])

    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor='validation_loss',
        min_delta=config['train']['tolerance'],
        patience=config['train']['patience']
    )

    # Model checkpoint callback to save the best model based on R-squared
    checkpoint_callback = ModelCheckpoint(
        monitor=config["train"]["metric"],  # Monitor R-squared
        mode='max',  # We want to maximize R-squared
        save_top_k=1,  # Only save the best model
        filename='best-model-{epoch:02d}-{val_r_squared:.2f}',  # File name format
        verbose=True  # Print info on save
    )

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=config['train']["epochs"],
        logger=logger,
        accelerator=config['train']['accelerator'],
        log_every_n_steps=10,
        enable_progress_bar=False,
        callbacks=[early_stopping_callback, checkpoint_callback]
    )

    # Start training
    trainer.fit(model, datamodule)
    import wandb
    wandb.finish()

    # Log the true and estimated transformation matrices
    # true_nonlinearity = datamodule.dataset.lin_transform.detach().cpu().numpy()
    # est_nonlinearity = model.decoder.lin_transform.matrix.detach().cpu().numpy()
    #
    # return true_nonlinearity, est_nonlinearity


# Parse arguments and allow running as a standalone script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model with specified parameters')
    parser.add_argument('model_name', type=str, help='Name of the model to train (e.g., vansca)')
    parser.add_argument('--lr_th', type=float, default=None, help='Learning rate threshold (th)')
    parser.add_argument('--lr_ph', type=float, default=None, help='Learning rate phase (ph)')
    parser.add_argument('--snr', type=float, default=None, help='Signal-to-noise ratio (SNR)')

    args = parser.parse_args()

    # Call the function with parsed arguments
    run_training(
        model_name=args.model_name,
        lr_th=args.lr_th,
        lr_ph=args.lr_ph,
        snr=args.snr
    )

    # from pytorch_lightning import Trainer
    # import torch
    #
    # from src.utils import init_logger, load_config
    # from src.data_module import DataModule
    # from src.training_module import EarlyStoppingA
    # import src.network as network_package
    # import src.model as model_package
    #
    # if __name__ == '__main__':
    #
    #     config = load_config()
    #     if config["train"]["seed"] is not None:
    #         torch.manual_seed(config["train"]["seed"])
    #
    #     module = getattr(model_package, config['model'])
    #     network = getattr(network_package, config['model'])
    #     datamodule = DataModule(config)
    #
    #     encoder = network.Encoder(
    #         input_dim=config['data']['observed_dim'],
    #         latent_dim=config['data']['latent_dim'],
    #         hidden_layers=config['encoder']['hidden_dim'],
    #     )
    #     decoder = network.Decoder(
    #         latent_dim=config['data']['latent_dim'],
    #         output_dim=config['data']['observed_dim'],
    #         hidden_layers=config['decoder']['hidden_dim'],
    #         activation=config['decoder']['activation'],
    #         sigma=datamodule.dataset.sigma
    #     )
    #
    #     model = module.Model(
    #         encoder=encoder,
    #         decoder=decoder,
    #         data_model=datamodule,
    #         mc_samples=config['train']['mc_samples'],
    #         lr=config['train']['lr']
    #     )
    #
    #     logger = init_logger(project=config['project'], experiment=config['experiment'])
    #     logger.log_hyperparams(config)
    #     logger.watch(model, log=config['train']['log'])
    #
    #     early_stopping_callback = EarlyStoppingA(tolerance=config['train']['tolerance'])
    #     trainer = Trainer(
    #         max_epochs=config['train']["epochs"],
    #         logger=logger,
    #         accelerator=config['train']['accelerator'],
    #         log_every_n_steps=10,
    #         enable_progress_bar=False,
    #         # check_val_every_n_epoch=1,
    #         # val_check_interval=1.0,
    #         callbacks=[early_stopping_callback]
    #     )
    #     trainer.fit(model, datamodule)
    #
    #     true_A = datamodule.dataset.lin_transform.detach().cpu().numpy()
    #     est_A = model.decoder.lin_transform.matrix.detach().cpu().numpy()

        # print(true_A, "\n", est_A)

        # import wandb
        #
        # def rescale_matrix(matrix):
        #     matrix_min = torch.min(matrix).numpy()
        #     matrix_max = torch.max(matrix).numpy()
        #     return (matrix - matrix_min) / (matrix_max - matrix_min)
        #
        # true_A = rescale_matrix(true_A)
        # est_A = rescale_matrix(est_A)
        #
        # true_A_panel = wandb.Image(true_A, caption="True Matrix")
        # est_A_panel = wandb.Image(est_A, caption="Estimated Matrix")
        #
        # logger.experiment.log({
        #     "true_A": true_A_panel,
        #     "est_A": est_A_panel
        # })
        # logger.experiment.log(model.metric())

        # checkpoint_callback = ModelCheckpoint(
        #     monitor='val_loss',
        #     dirpath='my/path/',
        #     filename='sample-{epoch:02d}-{val_loss:.2f}',
        #     save_top_k=3,
        #     mode='min',
        # )
        # early_stopping_callback = EarlyStopping(
        #     monitor='val_loss',
        #     patience=3,
        #     verbose=True,
        #     mode='min'
        # )
        #
        # # Setup trainer
        # trainer = pl.Trainer(
        #     max_epochs=20,
        #     logger=logger,
        #     callbacks=[checkpoint_callback, early_stopping_callback]
        # )

    # feat: probabilistic model
    # feat: metrics
    # feat: plots
