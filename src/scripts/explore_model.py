import os
import torch
from pytorch_lightning import Trainer
import src.model as model_package
import src.modules.data as data_package
import src.experiments as exp_module
from src.helpers.run_analyzer import RunAnalyzer
from src.utils.utils import logging_setup


def load_model(run_id, experiment_name):

    checkpoints_dir = f"../experiments/{experiment_name}/checkpoints/{run_id}/"
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found for run ID {run_id} in {checkpoints_dir}")

    best_model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
    checkpoint = torch.load(best_model_path)
    config = checkpoint["hyper_parameters"]
    print(f"Loaded model {config["model_name"]} from {best_model_path}")

    module = getattr(model_package, config['model_name'].upper())
    encoder = module.Encoder(config=config['encoder'])
    decoder = module.Decoder(config=config['decoder'])

    encoder.construct(latent_dim=config['model']['latent_dim'],
                      observed_dim=config['model']['observed_dim'])
    decoder.construct(latent_dim=config['model']['latent_dim'],
                      observed_dim=config['model']['observed_dim'])

    metrics_module = getattr(exp_module, experiment_name)
    metrics = metrics_module.ModelMetrics(monitor=config['metric']['name']).eval()

    model = module.Model.load_from_checkpoint(
        checkpoint_path=best_model_path,
        encoder=encoder,
        decoder=decoder,
        optimizer_config=config['optimizer'],
        model_config=config['model'],
        strict=False,
        metrics=metrics
    )
    model.eval()
    return model, config


def predict(experiment, run_id):
    os.environ["EXPERIMENT"] = experiment
    os.environ["RUN_ID"] = run_id

    print(f"Predicting for experiment {experiment} and run ID {run_id}")

    logging_setup()
    model, config = load_model(run_id, experiment)

    torch.manual_seed(config['torch_seed'])
    trainer = Trainer(**config['trainer'], logger=False)
    datamodule = getattr(data_package, experiment).DataModule(config['data_config'], **config['data_loader'])
    trainer.predict(model, datamodule)
    return model, datamodule


def plot_training_history(model):
    try:
        analyzer = RunAnalyzer(os.environ["EXPERIMENT"], os.environ["RUN_ID"])
    except FileNotFoundError as e:
        print(e)
        return
    analyzer.plot_training_history()
    for metric in model.metrics.metrics_list:
        analyzer.plot_training_history(metric_key=metric)


if __name__ == "__main__":
    model, _ = predict(
        "hyperspectral", "u1wxkz3r")

    plot_training_history(model)

# todo: use number of parameters and number of layers as parameters, every architecture can be different
# todo: latent_dims are not saved to the model, save them when training model, when we get them from data
# todo: automatically adjust layer dims from config, so it's compatible with CNN
# todo: pass transform to the metric
# todo: automatic metrics list and arg parsing
# todo: unmixing plots
# todo: kl and reconstruction plots
# task: load config from the loaded model snapshot wandb
# task: hyperparameters (configs) not saved to checkpoints
