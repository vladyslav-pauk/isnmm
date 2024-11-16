from src.helpers.trainer import train_model
from src.utils.config_tools import load_sweep_config
from src.utils.utils import sweep_parser
from src.helpers.sweep_runner import Sweep

from analyze_sweep import analyze_sweep
from src.utils.wandb_tools import login_wandb
from src.utils.utils import logging_setup
from src.scripts.explore_model import predict, load_model


if __name__ == '__main__':
    args = sweep_parser()

    experiment = args.experiment
    sweep = args.sweep
    sweep_config = load_sweep_config(experiment, sweep)

    logging_setup()
    login_wandb(experiment)

    print(f"Experiment '{experiment}'")
    sweep = Sweep(sweep_config, train_model)
    sweep.run(save=True)
    # sweep.fetch_data(save=True)

    _, data = analyze_sweep(experiment, sweep.id, save=True)

    if sweep == 'test_run':
        run_id = list(data.keys())[0]
        model = data[run_id]['config']['model_name']
        model = load_model(run_id, model, experiment)
        predict(experiment, model, run_id)

    import os
    import shutil
    path_to_remove = f"{os.path.dirname(os.path.abspath(__file__)).split('src')[0]}experiments/{experiment}/nisca"
    if os.path.exists(path_to_remove):
        shutil.rmtree(path_to_remove, ignore_errors=False)


# todo: discard run, if metrics is below threshold (-10 positive mse_db)
# task: latent_sample contains nans -> skip
