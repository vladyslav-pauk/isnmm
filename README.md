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

### Notebooks

Explore notebooks covering a range of topics from VAE training to model evaluation and visualization, as well as some experiments.


### Training module

Run train.py to train a model with a specific config file.

```bash
python train.py --config experiments/{experiment_name}/model_config.json
```

### Scripts

Run run_sweep.py to train a model with a specific config file.

```bash
python run_sweep.py ....
```

Run schedule.py to schedule a model with a specific config file.

```bash
python schedule.py --config experiments/{experiment_name}/model_config.json
```

### Experiments

Configuration files `{model}.json` and `{data}.json` in `experiments/{experiment_name}/` contain the following fields:

- `data`: data configuration
...