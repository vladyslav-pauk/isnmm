from src.helpers.sweep_analyzer import SweepAnalyzer
from pprint import pprint


def analyze_sweep(experiment, sweep_id):
    experiment_analyzer = SweepAnalyzer(experiment, sweep_id)

    data = experiment_analyzer.extract_metrics(
        metric="latent_mse", covariate="snr", comparison="model_name"
    )
    averaged_data = experiment_analyzer.average_seeds(data)
    experiment_analyzer.plot_metric(averaged_data)
    pprint(averaged_data)


if __name__ == "__main__":
    experiment = "nonlinearity_removal"
    sweep_id = "c9mus55a"

    analyze_sweep(experiment, sweep_id)

# todo: covariate, metric, comparison markers to sweep dataset
