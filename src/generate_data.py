import torch
from pytorch_lightning import seed_everything

from src.helpers.utils import load_experiment_config
import src.modules.distribution as distribution_package


if __name__ == "__main__":
    data_model_name = 'nnmm'
    config = load_experiment_config('nonlinearity_removal', data_model_name)

    linear_mixture_matrix = torch.randn(config["observed_dim"], config["latent_dim"])

    generative_model = getattr(distribution_package, config["model_name"])
    model = generative_model(linear_mixture_matrix, data_model_name, **config)

    sample, (latent, linearly_mixed, noiseless) = model.sample()
    model.save_data()

    model.plot_nonlinearities()
