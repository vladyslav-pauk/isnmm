import os

import numpy.random
import torch

import src.model as model_package
from src.modules.data.synthetic import DataModule


def load_model(run_id, model_name, experiment_name):
    module = getattr(model_package, model_name)

    checkpoints_dir = f"../models/{experiment_name}/{run_id}/checkpoints/"
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
    best_model_path = os.path.join(checkpoints_dir, checkpoint_files[0])

    checkpoint = torch.load(best_model_path)
    config = checkpoint["hyper_parameters"]

    encoder = module.Encoder(config=config['encoder'])
    decoder = module.Decoder(config=config['decoder'])

    encoder.construct(latent_dim=config['data_config']['latent_dim'], observed_dim=config['data_config']['observed_dim'])
    decoder.construct(latent_dim=config['data_config']['latent_dim'], observed_dim=config['data_config']['observed_dim'])

    model = module.Model.load_from_checkpoint(
        checkpoint_path=best_model_path,
        encoder=encoder,
        decoder=decoder,
        optimizer_config=config['optimizer'],
        model_config=config['model'],
        strict=False
    )

    model.eval()

    return model, config


if __name__ == "__main__":
    numpy_seed = 0
    torch_seed = 0
    torch.manual_seed(torch_seed)
    numpy.random.seed(numpy_seed)

    experiment_name = "nonlinearity_removal"
    model_name = "NISCA"
    base_model = 'MVES'

    run_id = 'gkoq3xw4'

    model, config = load_model(
        run_id=run_id,
        model_name=model_name,
        experiment_name=experiment_name,
    )

    datamodule = DataModule(config['data_config'], **config['data_loader'])
    datamodule.prepare_data()
    observed_data, latent_data = datamodule.observed_sample, datamodule.latent_sample

    with torch.no_grad():
        latent_sample_mixed = model(observed_data)['reconstructed_sample']
    linear_mixture = model.decoder.linear_mixture.matrix.cpu().detach()

    seed = 0
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    M = 3  # Number of spectral bands
    N = 3  # Number of endmembers
    L = 10000  # Number of pixels
    linear_mixture = torch.rand(M, N)
    alpha = torch.ones(N)
    latent_sample_true = torch.distributions.Dirichlet(alpha).sample((L,))
    latent_sample_mixed = latent_sample_true @ linear_mixture.T

    unmixing_model = getattr(model_package, base_model).Model
    unmixing = unmixing_model(observed_dim=observed_data.size(-1), latent_dim=latent_data.size(-1), dataset_size=observed_data.size(0))
    latent_sample = unmixing.estimate_abundances(latent_sample_mixed)

    print("Mean SAM Endmembers: {}\nMean SAM Abundances: {}".format(*unmixing.compute_metrics(linear_mixture, latent_sample, latent_sample_true)))

    # unmixing.plot_multiple_abundances(latent_sample, [0,1,2,3,4,5,6,7,8,9])
    # unmixing.plot_mse_image(rows=100, cols=100)

# todo: finish explore_model script
# task: load config from the loaded model snapshot wandb
# task: hyperparameters (configs) not saved to checkpoints
# todo: docs describing experiments, datasets, models
