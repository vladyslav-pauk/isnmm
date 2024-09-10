import torch
import src.network as network_package
import src.model as model_package
from src.modules.utils import load_config

from src.modules.data_module import DataModule


def load_best_model(best_model_path, experiment_name):
    config = load_config(experiment_name)
    module = getattr(model_package, config["model"])
    network = getattr(network_package, config["model"])
    # todo: load config with the model

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
    seed = None
    best_model_path = "models/vansca/seed_None-snr_None-lr_th_None-lr_ph_None/checkpoints/best-model-epoch=00-val_r_squared=0.00.ckpt"

    config = load_config('nnmm-vasca')
    datamodule = DataModule(config)
    datamodule.setup()
    x_data, z_data = next(iter(datamodule.test_dataloader()))

    model = load_best_model(best_model_path, experiment_name='nnmm-vasca')

    with torch.no_grad():
        predictions = model.decoder.nonlinear_transform(x_data)

    print("Predictions:", predictions)


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
