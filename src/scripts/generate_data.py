import torch
from pytorch_lightning import seed_everything

from src.helpers.utils import load_data_config, update_hyperparameters
import src.modules.distribution as distribution_package


def initialize_data_model(experiment_name, data_model_name, **kwargs):

    config = load_data_config(experiment_name)
    config = update_hyperparameters(config, kwargs)

    seed = config["data_seed"]
    if seed:
        seed_everything(seed, workers=True)

    linear_mixture_matrix = torch.randn(config["observed_dim"], config["latent_dim"])

    generative_model = getattr(distribution_package, config["data_module_name"])
    model = generative_model(linear_mixture_matrix, data_model_name, **config)

    return model


if __name__ == "__main__":
    data_model_name = 'nnmm'
    experiment_name = 'nonlinearity_removal'

    model = initialize_data_model(experiment_name, data_model_name)

    model.sample()
    model.save_data()

    # model.plot_nonlinearities()
