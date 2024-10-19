import os
import torch

import src.model as model_package
from src.helpers.utils import load_model_config, load_data_config
# from src.scripts.train import train_model
from src.modules.data.synthetic import DataModule
# from src.modules.metric.matrix_mse import mse_matrix_db


def load_model(run_id, model_name, experiment_name, datamodule):

    module = getattr(model_package, model_name)

    checkpoints_dir = f"../../models/{experiment_name}/{run_id}/checkpoints/"
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
    best_model_path = os.path.join(checkpoints_dir, checkpoint_files[0])

    checkpoint = torch.load(best_model_path)

    config = checkpoint["hyper_parameters"]['config']
    config['data'] = checkpoint["hyper_parameters"]['data_config']['data']

    encoder = module.Encoder(
        input_dim=config['data']['observed_dim'],
        latent_dim=config['data']['latent_dim'],
        hidden_layers=config['encoder']['hidden_dim'],
    )
    decoder = module.Decoder(
        latent_dim=config['data']['latent_dim'],
        output_dim=config['data']['observed_dim'],
        hidden_layers=config['decoder']['hidden_dim'],
        activation=config['decoder']['activation'],
    )

    model = module.Model.load_from_checkpoint(
        checkpoint_path=best_model_path,
        encoder=encoder,
        data_model=datamodule,
        decoder=decoder,
        lr=config['train']['lr'],
    )

    model.eval()

    return model


if __name__ == "__main__":
    experiment_name = "simplex_recovery"
    model_name = "VASCA"

    data_config = load_data_config(experiment_name)
    config = load_model_config(experiment_name, model_name)

    datamodule = DataModule(data_config, **config['data_loader'])

    run_id = 'olk8thyp'
    # training_config = load_experiment_config(experiment_name, model_name)
    # run_id = train_model(
    #     experiment_name=experiment_name,
    #     config=training_config,
    #     datamodule=datamodule,
    #     run='eval_script'
    # )

    model = load_model(
        run_id=run_id,
        model_name=model_name,
        experiment_name=experiment_name,
        datamodule=datamodule
    )

    print(model.decoder.linear_mixing.matrix)
    print(model.data_model.dataset.data_model.linear_mixing)
    # print(mse_matrix_db(model.decoder.linear_mixing.matrix, model.data_model.dataset.data_model.linear_mixing))

    # x_data, z_data = next(iter(datamodule.test_dataloader()))
    x_data, z_data = datamodule.dataset.data
    with torch.no_grad():
        predictions = model.decoder.nonlinear_transform(x_data)
    print("Predictions:", predictions)

# todo: finish explore_model script
# task: load config from the loaded model snapshot wandb
# task: hyperparameters (configs) not saved to checkpoints
# todo: docs describing experiments, datasets, models
