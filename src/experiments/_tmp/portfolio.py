from src.train import train_model
import os, sys, json
import wandb


if __name__ == '__main__':
    model = 'nisca'
    experiment = "portfolio_return"
    data_model = "yahoo"

    with open(f'experiments/{experiment}/{model}.json', 'r') as f:
        config = json.load(f)

    train_model(
        experiment_name=experiment,
        data_model_name=data_model,
        model_name=model,
        hyperparameters={}
    )

    wandb.finish()