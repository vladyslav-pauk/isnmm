import torch
from pytorch_lightning import seed_everything

from src.helpers.config_tools import load_data_config, update_hyperparameters
import src.modules.distribution as distribution_package


def initialize_data_model(experiment_name, data_model, **kwargs):

    config = load_data_config(experiment_name)
    config = update_hyperparameters(config, kwargs, show_log=False)
    config['data_model'] = data_model

    seed = config["data_seed"]
    if seed:
        seed_everything(seed, workers=True)

    linear_mixture_matrix = torch.randn(config["observed_dim"], config["latent_dim"])
    generative_model = getattr(distribution_package, config["data_module_name"])
    model = generative_model(linear_mixture_matrix, data_model, **config)

    return model


if __name__ == "__main__":
    data_model_name = 'nonlinear'
    experiment_name = 'nonlinearity_removal'

    model = initialize_data_model(experiment_name, data_model_name)

    model.sample()
    model.save_data()

    # model.plot_nonlinearities()
