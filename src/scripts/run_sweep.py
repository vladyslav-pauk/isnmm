from train import train_model
from src.helpers.utils import load_sweep_config, sweep_parser
from src.helpers.sweep import Sweep


if __name__ == '__main__':
    args = sweep_parser()

    experiment = args.experiment
    sweep = args.sweep
    sweep_config = load_sweep_config(experiment, sweep)
    # sweep_config["name"] = f"{sweep}"

    print(f"Experiment '{experiment}'")

    sweep = Sweep(sweep_config, train_model)
    sweep.run()
    sweep.fetch_data()

# todo: model using schedule script, for each model a sweep?
# todo: discard run, if metrics is bad (positive mse_db)
