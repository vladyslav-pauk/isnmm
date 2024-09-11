import os
import torch
import src.network as network_package
import src.model as model_package
from src.modules.utils import load_experiment_config
from src.train import train_model
from src.modules.data_module import DataModule
from src.modules.metrics import mse_matrix_db


def load_best_model(run_id, model_name, experiment_name, datamodule):

    module = getattr(model_package, model_name)
    network = getattr(network_package, model_name)

    checkpoints_dir = f"models/{experiment_name}/{run_id}/checkpoints/"
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
    best_model_path = os.path.join(checkpoints_dir, checkpoint_files[0])

    checkpoint = torch.load(best_model_path)
    config = checkpoint["hyper_parameters"]['config']
    config['data'] = checkpoint["hyper_parameters"]['data_config']['data']

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
    experiment = "lmm"
    model_name = "vasca"

    data_config = load_experiment_config(experiment, 'data')
    training_config = load_experiment_config(experiment, model_name)

    datamodule = DataModule(data_config)
    datamodule.setup()

    # run_id = train_model(
    #     experiment_name=experiment,
    #     config=training_config,
    #     datamodule=datamodule
    # )
    run_id = 'w85y5w1i'

    model = load_best_model(run_id=run_id, model_name=model_name, experiment_name='lmm', datamodule=datamodule)
    print(model.decoder.lin_transform.matrix)
    print(model.data_model.dataset.lin_transform)
    print(mse_matrix_db(model.decoder.lin_transform.matrix, model.data_model.dataset.lin_transform))

    # x_data, z_data = next(iter(datamodule.test_dataloader()))
    # x_data, z_data = datamodule.dataset.data
    # with torch.no_grad():
    #     predictions = model.decoder.nonlinear_transform(x_data)
    # print("Predictions:", predictions)


# fixme: adapt prism to new code
# fixme: run with only reconstruction term, check convergence, compare with prism, play with those
# fixme: clean and squash github commits
# fixme: run schedule
# fixme: training seed should be different from the data seed. when doing repeting runs in schedule should be random
# fixme: repeated runs for every hyperparameter in scheduler (number of runs in config)

# todo: extend neural network architecture and expressiveness (batch, other methods), ask gpt
# todo: check inverse stability when the residual is discontinuous todo: subspace metric
# todo: evaluate.py with wandb loader;
#  Download the best model file from a sweep. This snippet downloads the model file with the highest
#  validation accuracy from a sweep with runs that saved model files to model.h5. see history in gpt
#  todo: load config from the loaded model snapshot wandb
