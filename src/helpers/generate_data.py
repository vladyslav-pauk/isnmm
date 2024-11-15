import torch
from pytorch_lightning import seed_everything

from src.utils.config_tools import load_data_config, update_hyperparameters
import src.modules.distribution.mixture_model as mixture_model


def initialize_data_model(experiment_name, **kwargs):

    config = load_data_config(experiment_name)
    config = update_hyperparameters(config, kwargs, show_log=False)
    # config['data_model'] = data_model

    seed = config["data_seed"]
    if seed:
        seed_everything(seed, workers=True)

    linear_mixture_matrix = torch.randn(config["observed_dim"], config["latent_dim"])
    generative_model = mixture_model.GenerativeModel
    model = generative_model(linear_mixture_matrix, **config)

    return model


if __name__ == "__main__":
    experiment_name = 'synthetic_data'

    model = initialize_data_model(experiment_name)

    model.sample()
    model.save_data()

    # model.plot_nonlinearities()
