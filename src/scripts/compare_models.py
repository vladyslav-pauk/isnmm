import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import Trainer

from torch.utils.data import DataLoader
import src.model as model_package
import src.modules.data as data_package
from src.utils.plot_tools import init_plot
from src.modules.utils import run_dir, unmix, permute
import src.experiments as exp_module
from src.experiments.hyperspectral import ModelMetrics


def load_model(run_id, experiment_name):
    checkpoints_dir = f"../../experiments/{experiment_name}/checkpoints/{run_id}/"
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found for run ID {run_id} in {checkpoints_dir}")

    best_model_path = os.path.join(checkpoints_dir, checkpoint_files[0])
    checkpoint = torch.load(best_model_path)
    config = checkpoint["hyper_parameters"]

    module = getattr(model_package, config['model_name'].upper())
    encoder = module.Encoder(config=config['encoder'])
    decoder = module.Decoder(config=config['decoder'])
    encoder.construct(latent_dim=config['model']['latent_dim'], observed_dim=config['model']['observed_dim'])
    decoder.construct(latent_dim=config['model']['latent_dim'], observed_dim=config['model']['observed_dim'])

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


def generate_predictions(model, datamodule):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        datamodule.dataset,
        batch_size=datamodule.dataset.__len__(),
        shuffle=False,
        num_workers=datamodule.num_workers
    )
    all_predictions = {}

    for batch in dataloader:
        data = batch["data"]
        with torch.no_grad():
            predictions = model(data)

        all_predictions = reduce_preds(predictions, all_predictions)

    all_predictions = cat_preds(predictions, all_predictions, config, datamodule)
    posterior0 = model.transform(all_predictions['posterior_parameterization'][0])
    posterior1 = model.transform(all_predictions['posterior_parameterization'][1])
    if config['model']['unmixing']:

        latent_sample, mixing_matrix = unmix(
            all_predictions['latent_sample'], config['model']['latent_dim'], config['model']['unmixing']
        )
        mixing_matrix_pinv = torch.linalg.pinv(mixing_matrix)
        all_predictions['latent_sample'] = torch.matmul(mixing_matrix_pinv, all_predictions['latent_sample'].T).T
        all_predictions['latent_sample_mean'] = torch.matmul(mixing_matrix_pinv, all_predictions['latent_sample_mean'].T).T
        all_predictions['posterior_parameterization'] = (
            torch.matmul(mixing_matrix_pinv, posterior0.T).T,
            torch.matmul(mixing_matrix_pinv, posterior1.T).T
        )

    latent_sample_true = datamodule.dataset.labels["latent_sample"]
    all_predictions['latent_sample'], permutation = permute(all_predictions['latent_sample'], latent_sample_true)
    all_predictions['latent_sample_mean'] = all_predictions['latent_sample_mean'][permutation]
    all_predictions['posterior_parameterization'] = (
        posterior0.transpose(1, 0)[permutation].T,
        posterior1.transpose(1, 0)[permutation].T
    )

    return all_predictions


def cat_preds(predictions, all_predictions, config, datamodule):
    for key in all_predictions:
        if isinstance(all_predictions[key][0], tuple):
            all_predictions[key] = tuple(
                torch.cat([batch[i] for batch in all_predictions[key]], dim=0)
                for i in range(len(all_predictions[key][0]))
            )
        else:
            all_predictions[key] = torch.cat(all_predictions[key], dim=0)
    return all_predictions


def reduce_preds(predictions, all_predictions=None):
    for key, value in predictions.items():
        if key not in all_predictions:
            all_predictions[key] = []

        if isinstance(value, tuple):
            reduced_tuple = tuple(
                elem.mean(dim=0).cpu() if elem.ndim == 3 else elem.cpu()
                for elem in value
            )
            all_predictions[key].append(reduced_tuple)
        else:
            reduced_value = value.mean(dim=0).cpu() if value.ndim == 3 else value.cpu()
            all_predictions[key].append(reduced_value)
    return all_predictions


def plot_predictions(models, predictions, image_dims, save_dir="comparison", visualization_key="latent_sample"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt = init_plot()
    A4_WIDTH = 8.27

    num_models = len(predictions)
    _, height, width = image_dims

    visualization_data = {
        run_id: preds[visualization_key].cpu().numpy()
        for run_id, preds in predictions.items()
    }

    num_components = list(visualization_data.values())[0].shape[-1]

    fig, axs = plt.subplots(
        num_components, num_models,
        figsize=(A4_WIDTH, A4_WIDTH * height / width * num_components / num_models),
        dpi=300
    )
    axs = np.atleast_2d(axs)

    # global_min = min(data.min() for data in visualization_data.values())
    # global_max = max(data.max() for data in visualization_data.values())

    for col_idx, (run_id, model_preds) in enumerate(visualization_data.items()):
        if run_id != 'true' and run_id != 'MVES':
            model_name = list(models[run_id].keys())[0]
        elif run_id == 'true':
            model_name = "True"
        else:
            model_name = "MVES"
        for row_idx in range(num_components):
            ax = axs[row_idx, col_idx]
            component = model_preds[..., row_idx].reshape(height, width)
            ax.imshow(component, cmap='viridis', vmin=0, vmax=1)

            if row_idx == 0:
                ax.set_title(f"{model_name.upper()}") #\nRun ID: {run_id}")

            if col_idx == 0:
                ax.set_ylabel(f"Component {row_idx + 1}")

            # ax.set_xticks([])
            # ax.set_yticks([])
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    plt.savefig(f"{save_dir}/{visualization_key}.png", transparent=True, dpi=300)
    print(f"Saved comparison plot to {save_dir}/{visualization_key}.png")
    plt.show()

# def plot_predictions(models, predictions, image_dims, save_dir="comparison", visualization_key="latent_sample"):
#     from matplotlib.gridspec import GridSpec
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     plt = init_plot()
#
#     _, height, width = image_dims
#     num_components = list(predictions.values())[0][visualization_key].shape[-1]
#     num_models = len(predictions)
#
#     A4_WIDTH = 8.27
#     fig_width = A4_WIDTH
#     fig_height = fig_width * num_components / num_models * (height / width)
#     fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
#     grid = GridSpec(num_components, num_models, figure=fig)
#
#     for col_idx, (run_id, preds) in enumerate(predictions.items()):
#         model_name = (
#             "True" if run_id == "true" else "MVES" if run_id == "MVES" else list(models[run_id].keys())[0].upper()
#         )
#
#         # Prepare data for plotting
#         tensors = {model_name: preds[visualization_key].cpu().numpy()}
#         data = tensors[model_name]
#
#         for row_idx in range(num_components):
#             component = data[..., row_idx].reshape(height, width)  # Reshape each component
#             new_ax = fig.add_subplot(grid[row_idx, col_idx])  # Create subplot for the grid
#             new_ax.imshow(component, cmap='viridis', vmin=0, vmax=1)
#             new_ax.set_title(f"{model_name}" if row_idx == 0 else "", fontsize=10)
#             if col_idx == 0:
#                 new_ax.set_ylabel(f"Component {row_idx + 1}", fontsize=10)
#             new_ax.axis('off')  # Remove axes for cleaner visualization
#
#     plt.tight_layout()
#
#     # Save and display the final combined plot
#     save_path = os.path.join(save_dir, f"{visualization_key}.png")
#     plt.savefig(save_path, transparent=True, dpi=300)
#     print(f"Saved combined comparison plot to {save_path}")
#     plt.show()

    # fixme: this has to use plots from plot tools and only arrange them in a grid, return axes, plt


def compare_component_metrics(metrics, metric_name, save_dir="comparison/components"):
    """
    Generate and combine component plots for one metric from ModelMetrics into a grid.

    :param metrics: ModelMetrics object with metrics already computed.
    :param metric_name: Name of the metric to plot.
    :param save_dir: Directory to save the combined comparison plot.
    """
    import os
    from matplotlib.gridspec import GridSpec

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Extract metric data
    if metric_name not in metrics.keys():
        print(f"Metric '{metric_name}' not found in metrics.")
        return

    metric_data = metrics[metric_name].compute()
    if metric_data is None:
        print(f"Metric '{metric_name}' has no data to plot.")
        return

    # Extract image dimensions
    image_dims = metrics.image_dims
    latent_dim = image_dims[0]
    _, height, width = image_dims

    # Determine grid size
    num_components = latent_dim  # Rows are components
    num_models = len(metric_data.keys())  # Columns are models

    A4_WIDTH = 8.27
    fig_width = A4_WIDTH
    fig_height = fig_width * num_components / num_models
    combined_fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
    grid = GridSpec(num_components, num_models, figure=combined_fig)

    # Plot each component for each model
    for col_idx, (run_id, run_data) in enumerate(metric_data.items()):
        components = run_data.cpu().numpy()  # Shape: [latent_dim, height, width]
        for row_idx in range(latent_dim):
            # Create subplot for each component and model
            new_ax = combined_fig.add_subplot(grid[row_idx, col_idx])
            component_data = components[row_idx].reshape(height, width)
            new_ax.imshow(component_data, cmap="viridis", vmin=0, vmax=1)
            if row_idx == 0:
                new_ax.set_title(f"Model: {run_id}", fontsize=10)
            if col_idx == 0:
                new_ax.set_ylabel(f"Component {row_idx + 1}", fontsize=10)
            new_ax.axis("off")  # Optional: Hide axes for cleaner visualization

    # Save and show the combined plot
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{metric_name}_combined_comparison.png")
    combined_fig.savefig(save_path, transparent=True, dpi=300)
    print(f"Saved combined component comparison plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    experiment = "hyperspectral"
    run_ids = ["xtni77vc", "fap50pvb"]

    models = {}
    predictions = {}
    for run_id in run_ids:
        model, config = load_model(run_id, experiment)

        models[run_id] = {config["model_name"]: model}
        datamodule = getattr(data_package, experiment).DataModule(config['data_config'], **config['data_loader'])
        datamodule.prepare_data()
        datamodule.setup()

        model_preds = generate_predictions(model, datamodule)
        predictions[run_id] = model_preds

    predictions['MVES'] = {}
    predictions['MVES']["latent_sample"], _ = unmix(
        datamodule.dataset.labels["latent_sample"],
        config['model']['latent_dim'],
        'MVES'
    )
    predictions['MVES']["latent_sample"], _ = permute(predictions['MVES']["latent_sample"], datamodule.dataset.labels["latent_sample"])
    predictions['MVES']["reconstructed_sample"] = model(
        datamodule.dataset.data
    )["reconstructed_sample"].detach()

    predictions['true'] = {}
    predictions['true']["latent_sample"] = datamodule.dataset.labels["latent_sample"]
    predictions['true']["reconstructed_sample"] = datamodule.dataset.data

    # fixme: print table and comparison latent plots

    metrics = ModelMetrics(show_plot=False, log_plot=False, save_plot=False, monitor='latent_mse')

    metrics.model = model
    metrics.true_model = datamodule
    metrics.setup_metrics(metrics_list=[])
    metrics.latent_dim = config['model']['latent_dim']
    metrics.unmixing = 'MVES'

    prediction = predictions['MVES']

    prediction['latent_sample'] = prediction['latent_sample'].unsqueeze(0)
    posterior_parameterization = torch.zeros_like(prediction['latent_sample'][..., :-1].squeeze(0))

    prediction['posterior_parameterization'] = (posterior_parameterization, posterior_parameterization)

    metrics.update(
        observed_sample=datamodule.dataset.data.unsqueeze(0),
        model_output=prediction,
        labels=datamodule.dataset.labels,
        idxes=None,
        model=model
    )
    computed_metrics = metrics.compute()

    print("Metrics for MVES:")
    for key, value in computed_metrics.items():
        print(f"{key}: {value}")

    for run_id in run_ids:
        model = list(models[run_id].values())[0]
        model_metrics = ModelMetrics(show_plot=False, log_plot=False, save_plot=False, monitor='latent_mse')
        model_metrics.model = model
        model_metrics.true_model = datamodule
        model_metrics.setup_metrics(metrics_list=[])
        model_metrics.latent_dim = config['model']['latent_dim']
        model_metrics.unmixing = config['model']['unmixing']

        prediction = predictions[run_id]

        prediction['latent_sample'] = prediction['latent_sample'].unsqueeze(0)
        prediction['posterior_parameterization'] = (prediction['posterior_parameterization'][0].squeeze(0), prediction['posterior_parameterization'][1].squeeze(0))

        model_metrics.update(
            observed_sample=datamodule.dataset.data.unsqueeze(0),
            model_output=prediction,
            labels=datamodule.dataset.labels,
            idxes=None,
            model=model
        )

        model_computed_metrics = model_metrics.compute()
        print(f"Metrics for run ID {run_id}:")
        for key, value in model_computed_metrics.items():
            print(f"{key}: {value}")

    # metrics.save(computed_metrics, save_dir="./metrics")

    if models and predictions:
        image_dims = (config['model']['latent_dim'], datamodule.transform.height, datamodule.transform.width)
        plot_predictions(
            models,
            predictions,
            image_dims,
            save_dir=f"../../experiments/{experiment}/comparison",
            visualization_key="reconstructed_sample"
        )
        plot_predictions(
            models,
            predictions,
            image_dims,
            save_dir=f"../../experiments/{experiment}/comparison",
            visualization_key="latent_sample"
        )

    metrics = ModelMetrics(show_plot=False, log_plot=False, save_plot=False, monitor="latent_mse")
    metrics.model = model
    metrics.true_model = datamodule
    metrics.setup_metrics(metrics_list=[])

    metrics.update(
        observed_sample=datamodule.dataset.data.unsqueeze(0),
        model_output=prediction,
        labels=datamodule.dataset.labels,
        idxes=None,
        model=model
    )

    computed_metrics = metrics.compute()  # Compute metrics as usual

    # Generate and save the comparison grid for 'latent_components'
    compare_component_metrics(metrics, metric_name="latent_components", save_dir="comparison/components")

# todo: run model comparison on a sweep with fixed covariate and best seed
