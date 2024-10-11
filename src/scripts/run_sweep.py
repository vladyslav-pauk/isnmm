from train import train_model
from src.helpers.utils import load_sweep_config, sweep_parser
from src.helpers.wandb import login_wandb
from src.helpers.sweep import Sweep


if __name__ == '__main__':
    args = sweep_parser()

    experiment = args.experiment
    sweep = args.sweep
    sweep_config = load_sweep_config(experiment, sweep)

    print(f"Experiment '{experiment}'")
    login_wandb()

    models = "_".join(sweep_config["parameters"]["model_name"]["values"])
    data = "_".join(sweep_config["parameters"]["data_model_name"]["values"])
    sweep_config["name"] = f"{sweep}_{models}_{data}"

    sweep = Sweep(sweep_config, train_model)
    sweep.run()
    sweep.fetch_data()

# todo: model using schedule script, for each model a sweep?
