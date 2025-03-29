from src.utils.config_tools import load_sweep_config
from src.utils.utils import sweep_parser
from src.utils.wandb_tools import login_wandb
from src.utils.utils import logging_setup, clean_up

from src.helpers.sweep_runner import Sweep
from src.helpers.trainer import train_model
from analyze_sweep import analyze_sweep
from src.scripts.explore_model import predict, plot_training_history


if __name__ == '__main__':
    args = sweep_parser()

    experiment = args.experiment
    sweep_name = args.sweep
    sweep_config = load_sweep_config(experiment, sweep_name)

    logging_setup()
    login_wandb(experiment)

    print(f"Experiment '{experiment}'")
    sweep = Sweep(sweep_config, train_model)

    sweep.run(save=True)

    _, data = analyze_sweep(
        experiment, sweep.id, metric='validation_loss', covariate='snr', comparison='model_name', save=True)

    print('Running prediction for the best model ')
    show_plots = True
    if show_plots:
        run_id = data['run_ids'][next(iter(data['run_ids']))][0]
        model, _ = predict(experiment, run_id)
        plot_training_history(model)
        # todo: use latest run here

    clean_up(experiment)

# todo: discard run, if metrics is below threshold (-10 positive mse_db)
# todo: analyze all metrics, by getting metrics list from the model