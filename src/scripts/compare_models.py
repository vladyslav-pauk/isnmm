import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
import src.model as model_package
import src.modules.data as data_package
from src.utils.utils import init_plot
from src.modules.utils import run_dir


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
    encoder.construct(latent_dim=config['model']['latent_dim'], observed_dim=config['data_config']['observed_dim'])
    decoder.construct(latent_dim=config['model']['latent_dim'], observed_dim=config['data_config']['observed_dim'])

    model = module.Model.load_from_checkpoint(
        checkpoint_path=best_model_path,
        encoder=encoder,
        decoder=decoder,
        optimizer_config=config['optimizer'],
        model_config=config['model'],
        strict=False
    )
    model.eval()
    return model, config


def generate_predictions(model, datamodule):
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        datamodule.dataset,
        batch_size=datamodule.batch_size,
        shuffle=False,
        num_workers=datamodule.num_workers
    )
    all_predictions = {}

    for batch in dataloader:
        data = batch["data"]
        with torch.no_grad():
            predictions = model(data)

        for key, value in predictions.items():
            if key not in all_predictions:
                all_predictions[key] = []

            if isinstance(value, tuple):
                # Handle tuples by processing each element and reassembling
                reduced_tuple = tuple(
                    elem.mean(dim=0).cpu() if elem.ndim == 3 else elem.cpu()
                    for elem in value
                )
                all_predictions[key].append(reduced_tuple)
            else:
                # Handle non-tuple tensors
                reduced_value = value.mean(dim=0).cpu() if value.ndim == 3 else value.cpu()
                all_predictions[key].append(reduced_value)

    # Concatenate predictions across batches
    for key in all_predictions:
        if isinstance(all_predictions[key][0], tuple):
            # Handle concatenation for tuples
            all_predictions[key] = tuple(
                torch.cat([batch[i] for batch in all_predictions[key]], dim=0)
                for i in range(len(all_predictions[key][0]))
            )
        else:
            # Handle concatenation for non-tuples
            all_predictions[key] = torch.cat(all_predictions[key], dim=0)

    return all_predictions


def plot_predictions(models, predictions, image_dims, save_dir="comparison", visualization_key="latent_sample"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt = init_plot()
    A4_WIDTH = 8.27

    num_models = len(predictions)
    _, height, width = image_dims

    # Extract predictions for visualization_key
    visualization_data = {
        run_id: preds[visualization_key].cpu().numpy()
        for run_id, preds in predictions.items()
    }

    num_components = list(visualization_data.values())[0].shape[-1]

    fig, axs = plt.subplots(
        num_components, num_models,
        figsize=(A4_WIDTH, 1.2 * A4_WIDTH * height / width * num_components / num_models),
        dpi=300
    )
    axs = np.atleast_2d(axs)  # Ensure axs is always 2D for compatibility

    global_min = min(data.min() for data in visualization_data.values())
    global_max = max(data.max() for data in visualization_data.values())

    for col_idx, (run_id, model_preds) in enumerate(visualization_data.items()):
        model_name = next(name for (name, id_), model in models.items() if id_ == run_id)
        for row_idx in range(num_components):
            ax = axs[row_idx, col_idx]
            component = model_preds[..., row_idx].reshape(height, width)
            ax.imshow(component, cmap='viridis', vmin=global_min, vmax=global_max)

            # Add model name below each column
            if row_idx == 0:
                ax.set_title(f"{model_name.upper()}\nRun ID: {run_id}")

            # Add component label for each row
            if col_idx == 0:
                ax.set_ylabel(f"Component {row_idx + 1}")

            # Remove ticks and spines
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    plt.savefig(f"{save_dir}/{visualization_key}.png", transparent=True, dpi=300)
    print(f"Saved comparison plot to {save_dir}/{visualization_key}.png")
    plt.show()


if __name__ == "__main__":
    experiment = "hyperspectral"
    run_ids = ["9trmnn96", "8ge1s35j", "mzn0rge0"]

    models = {}
    predictions = {}
    for run_id in run_ids:
        model, config = load_model(run_id, experiment)

        models[(config["model_name"], run_id)] = model
        datamodule = getattr(data_package, experiment).DataModule(config['data_config'], **config['data_loader'])
        datamodule.prepare_data()
        datamodule.setup()

        model_preds = generate_predictions(model, datamodule)
        predictions[run_id] = model_preds

    if models and predictions:
        image_dims = (config['model']['latent_dim'], datamodule.transform.height, datamodule.transform.width)
        plot_predictions(models, predictions, image_dims, save_dir=f"../../experiments/{experiment}/comparison", visualization_key="reconstructed_sample")
        plot_predictions(models, predictions, image_dims, save_dir=f"../../experiments/{experiment}/comparison",
                         visualization_key="latent_sample")
