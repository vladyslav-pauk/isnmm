import wandb

from train import train_model
from src.helpers.utils import load_experiment_config, sweep_parser
from src.helpers.wandb import login_wandb, init_wandb, fetch_wandb_sweep
from src.scripts.generate_data import initialize_data_model


def run():
    model_name, dataset_name, config = init_wandb(experiment)

    print(f"Dataset '{dataset_name}':")
    data_model = initialize_data_model(**config)
    data_model.sample()
    data_model.save_data()

    print(f"Model '{model_name}':")

    experiment_id = train_model(**config)


if __name__ == '__main__':
    args = sweep_parser()

    experiment = args.experiment
    sweep = args.sweep
    sweep_config = load_experiment_config(experiment, sweep)

    print(f"Experiment '{experiment}'")
    login_wandb()

    models = "_".join(sweep_config["parameters"]["model_name"]["values"])
    data = "_".join(sweep_config["parameters"]["data_model_name"]["values"])
    sweep_name = f"{sweep}_{models}_{data}"
    sweep_config["name"] = sweep_name

    sweep_id = wandb.sweep(sweep=sweep_config, project=experiment)
    wandb.sweep.name = sweep_name
    wandb.agent(sweep_id, function=run)

    sweep_data = fetch_wandb_sweep(experiment, 'pbgaxukm')
    import json
    with open(f"../experiments/{experiment}/sweeps/{sweep_id}.json", "w") as f:
        json.dump(sweep_data, f)

# todo: model using schedule script, for each model a sweep?
