import torch
from pytorch_lightning import seed_everything

from src.helpers.utils import load_experiment_config
from src.modules.distribution.mixture_model import GenerativeModel


if __name__ == "__main__":
    data_model_name = 'noisy_nmm'
    config = load_experiment_config('nonlinearity_removal', data_model_name)

    linear_mixture_matrix = torch.randn(config["observed_dim"], config["latent_dim"])
    model = GenerativeModel(linear_mixture_matrix, data_model_name, **config)

    sample, (latent, linearly_mixed, noiseless) = model.sample(torch.Size([10000]))
    model.save_data()

    model.plot_sample()
    model.plot_nonlinearities()
