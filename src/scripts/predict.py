import os
import torch
import src.model as model_package
from src.helpers.utils import load_experiment_config
from src.train import train_model
from src.modules.data_module import DataModule
from src.helpers.metrics import mse_matrix_db


def load_best_model(run_id, model_name, experiment_name, datamodule):

    module = getattr(model_package, model_name)

    checkpoints_dir = f"models/{model_name}/{experiment_name}/{run_id}/checkpoints/"
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
    model_name = "vasca"
    data_model_name = "lmm"

    data_config = load_experiment_config(experiment_name, data_model_name)
    datamodule = DataModule(data_config)
    datamodule.setup()

    run_id = 'ryauf8s3'
    # training_config = load_experiment_config(experiment_name, model_name)
    # run_id = train_model(
    #     experiment_name=experiment_name,
    #     config=training_config,
    #     datamodule=datamodule,
    #     run='eval_script'
    # )

    model = load_best_model(
        run_id=run_id,
        model_name=model_name,
        experiment_name=experiment_name,
        datamodule=datamodule
    )

    print(model.decoder.linear_mixing.matrix)
    print(model.data_model.dataset.data_model.linear_mixing)
    print(mse_matrix_db(model.decoder.linear_mixing.matrix, model.data_model.dataset.data_model.linear_mixing))

    # x_data, z_data = next(iter(datamodule.test_dataloader()))
    x_data, z_data = datamodule.dataset.data
    with torch.no_grad():
        predictions = model.decoder.nonlinear_transform(x_data)
    print("Predictions:", predictions)


# todo: fix folder structure
# todo: implement prism
# todo: clean and squash github commits
# todo: run schedule
# todo: reconsider the importing approach (getattr), ask google how to properly do it with modules
# todo: use yaml for config
# todo: check inverse stability when the residual is discontinuous todo: subspace metric
# todo: predict.py with wandb loader;
#  Download the best model file from a sweep. This snippet downloads the model file with the highest
#  validation accuracy from a sweep with runs that saved model files to model.h5. see history in gpt
#  todo: load config from the loaded model snapshot wandb
# todo: make x-axis epoch instead of global step
