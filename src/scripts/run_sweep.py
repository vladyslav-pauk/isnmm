from src.helpers.trainer import train_model
from src.helpers.config_tools import load_sweep_config
from src.helpers.utils import sweep_parser
from src.helpers.sweep_runner import Sweep
from src.helpers.sweep_analyzer import SweepAnalyzer
from pprint import pprint


if __name__ == '__main__':
    args = sweep_parser()

    experiment = args.experiment
    sweep = args.sweep
    sweep_config = load_sweep_config(experiment, sweep)
    # sweep_config["name"] = f"{sweep}"

    print(f"Experiment '{experiment}'")

    sweep = Sweep(sweep_config, train_model)
    sweep.run()
    sweep.fetch_data(save=True)

    experiment_analyzer = SweepAnalyzer(experiment, sweep.id)
    data = experiment_analyzer.extract_metrics(
        metric="latent_mse", covariate="snr", comparison="model_name"
    )
    averaged_data = experiment_analyzer.average_seeds(data)
    experiment_analyzer.plot_metric(averaged_data, save=False)
    pprint(averaged_data)

# fixme: discard run, if metrics is below threshold (-10 positive mse_db)
# todo: saving plots and results
# fixme: test single runs for SNRs and models I want to do, make sure I get results, fine tune configs and schedule parameters
# fixme: run schedule to unmix latents with NISCA, CNAE+MVES, VASCA, MVES depending on noise, and latent dimension
# todo: latent_sample contains nans -> skip
