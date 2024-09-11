# README

A codebase for training a VAE with AEVB algorithm  on MNIST dataset, implemented with PyTorch Lightning framework and WandB logger.

- experiments/{experiment_name}/ contains config files for data and models in each experiment.

model config json includes the following identification fields:
{
    "project": "isnmm", // team name on wandb
    "experiment": "lmm", // experiment name on wandb
    "model": "vasca", // group name on wandb
    "run_id": "run_name", // run name on wandb
}