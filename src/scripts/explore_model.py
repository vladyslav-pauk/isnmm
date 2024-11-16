import os

import numpy.random
import torch
from pytorch_lightning import Trainer

import src.model as model_package
from src.modules.data.synthetic import DataModule
import src.experiments as exp_module
from src.helpers.experiment_analyzer import ExperimentAnalyzer


def load_model(run_id, model_name, experiment_name):
    module = getattr(model_package, model_name)

    checkpoints_dir = f"../experiments/{experiment_name}/checkpoints/{run_id}/"
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
    best_model_path = os.path.join(checkpoints_dir, checkpoint_files[0])

    checkpoint = torch.load(best_model_path)
    config = checkpoint["hyper_parameters"]

    encoder = module.Encoder(config=config['encoder'])
    decoder = module.Decoder(config=config['decoder'])

    encoder.construct(latent_dim=config['data_config']['latent_dim'], observed_dim=config['data_config']['observed_dim'])
    decoder.construct(latent_dim=config['data_config']['latent_dim'], observed_dim=config['data_config']['observed_dim'])

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


if __name__ == "__main__":
    os.environ["EXPERIMENT"] = "synthetic_data"
    model_name = "NISCA"
    os.environ["RUN_ID"] = "dfqq3dti"

    model, config = load_model(
        run_id=os.environ["RUN_ID"],
        model_name=model_name,
        experiment_name=os.environ["EXPERIMENT"],
    )

    torch.manual_seed(config['torch_seed'])

    trainer = Trainer(**config['trainer'], logger=False)

    datamodule = DataModule(config['data_config'], **config['data_loader'])

    trainer.predict(model, datamodule)

    experiment_analyzer = ExperimentAnalyzer(os.environ["EXPERIMENT"], os.environ["RUN_ID"])
    experiment_analyzer.plot_training_history(metric_key=config['metric']['name'])

    # base_model = 'MVES'
    # datamodule.prepare_data()
    # datamodule.setup()
    # observed_data, latent_data = datamodule.observed_sample, datamodule.latent_sample
    # with torch.no_grad():
    #     latent_sample_mixed = model(observed_data)['reconstructed_sample'].mean(0)
    #     linear_mixture = model.decoder.linear_mixture.matrix.cpu().detach()
    # linear_mixture_true = datamodule.linear_mixture
    # latent_sample_true = latent_data
    #
    # unmixing_model = getattr(model_package, base_model).Model
    # unmixing = unmixing_model(observed_dim=observed_data.size(-1), latent_dim=latent_data.size(-1), dataset_size=observed_data.size(0))
    # latent_sample = unmixing.estimate_abundances(latent_sample_mixed)
    #
    # print("Mean SAM Endmembers: {}\nMean SAM Abundances: {}".format(*unmixing.compute_metrics(linear_mixture, latent_sample, latent_sample_true)))

    # unmixing.plot_multiple_abundances(latent_sample, [0,1,2,3,4,5,6,7,8,9])
    # unmixing.plot_mse_image(rows=100, cols=100)

# fixme: implement the explore model script and jupyter notebook (add making and saving plots, and model parameters to tables)
# fixme: all plots and tables for the chosen best model on given settings
# fixme: save the plots and tables

# task: load config from the loaded model snapshot wandb
# task: hyperparameters (configs) not saved to checkpoints
