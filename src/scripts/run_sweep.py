from src.helpers.trainer import train_model
from src.utils.config_tools import load_sweep_config
from src.utils.utils import sweep_parser
from src.helpers.sweep_runner import Sweep

from analyze_sweep import analyze_sweep
from src.utils.wandb_tools import login_wandb


if __name__ == '__main__':
    args = sweep_parser()

    experiment = args.experiment
    sweep = args.sweep
    sweep_config = load_sweep_config(experiment, sweep)
    login_wandb(experiment)

    print(f"Experiment '{experiment}'")
    sweep = Sweep(sweep_config, train_model)
    sweep.run()

    sweep.fetch_data(save=True)
    analyze_sweep(experiment, sweep.id, save=True)

    import os
    import shutil
    path_to_remove = f"{os.path.dirname(os.path.abspath(__file__)).split('src')[0]}experiments/{experiment}/nisca"
    if os.path.exists(path_to_remove):
        shutil.rmtree(path_to_remove, ignore_errors=False)


# fixme: discard run, if metrics is below threshold (-10 positive mse_db)
# fixme: test single runs for SNRs and models I want to do, make sure I get results, fine tune configs and schedule parameters
# fixme: run schedule to unmix latents with NISCA, CNAE+MVES, VASCA, MVES depending on noise, and latent dimension
# todo: latent_sample contains nans -> skip
