from src.helpers.sweep_analyzer import SweepAnalyzer


if __name__ == "__main__":
    experiment = "simplex_recovery"
    sweep_id = "g42wx08q"

    experiment_analyzer = SweepAnalyzer(experiment, sweep_id)

    data = experiment_analyzer.extract_metrics(
        metric="latent_mse_db", covariate="snr", comparison="model_name"
    )
    averaged_data = experiment_analyzer.average_seeds(data)
    experiment_analyzer.plot_metric(averaged_data)
    print(averaged_data)

# todo: covariate, metric, comparison markers to sweep dataset
