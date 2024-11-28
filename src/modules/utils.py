import os
import json
import numpy as np
import torch

from src.utils.utils import init_plot
from src.utils.wandb_tools import run_dir


def dict_to_str(d):
    return '_'.join([f'{value}' for key, value in d.items() if value is not None])


def unmix_components(latent_sample, latent_dim, model=None):
    import src.model as model_package

    dataset_size = latent_sample.size(0)
    unmixing_model = getattr(model_package, model).Model
    unmixing = unmixing_model(
        latent_dim=latent_dim,
        dataset_size=dataset_size
    )

    latent_sample, mixing_matrix = unmixing.estimate_abundances(latent_sample.squeeze().cpu().detach())
    latent_sample = latent_sample / latent_sample.sum(dim=-1, keepdim=True)

    # unmixing.plot_multiple_abundances(latent_sample, [0,1,2,3,4,5,6,7,8,9])
    # unmixing.plot_mse_image(rows=100, cols=10)

    return latent_sample, mixing_matrix


def unmix(state_data, unmixing, latent_dim):
    if unmixing and "latent_sample" in state_data:
        state_data["latent_sample"], mixing_matrix = unmix_components(
            state_data["latent_sample"],
            latent_dim=latent_dim,
            model=unmixing
        )
        mixing_matrix_pinv = torch.linalg.pinv(mixing_matrix)

        for key, value in state_data.items():
            if key != "latent_sample" and key != "true":
                state_data[key] = torch.matmul(mixing_matrix_pinv, value.T).T

    return state_data


def permute(state_data):
    if "latent_sample" in state_data:
        permutation, _ = best_permutation_mse(state_data["latent_sample"], state_data["true"])
        for key in state_data:
            if key != "true":
                state_data[key] = state_data[key][:, permutation]
    return state_data


def plot_data(data, image_dims, show_plot=False, save_plot=False):

    if not show_plot and not save_plot:
        return

    plt = init_plot()

    A4_WIDTH = 8.27

    _, height, width = image_dims

    # all_data = []
    # for key, tensor in data.items():
    #     all_data.append(tensor.T.view(-1, height, width))
    # all_data = torch.cat(all_data, dim=0)
    # all_data = torch.cat([data.T.view(-1, height, width) for data in data.values()], dim=0)

    global_min = 0 #all_data.min().item()
    global_max = 1 #all_data.max().item()
    # print(f"Global normalization: min={global_min}, max={global_max}")

    for key, data in data.items():
        data = data.T.view(-1, height, width)
        num_components = data.shape[0]

        cols = 4
        rows = (num_components + cols - 1) // cols

        aspect_ratio = height / width
        fig_width = A4_WIDTH
        fig_height = fig_width * rows / cols * aspect_ratio

        fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=300)
        axs = np.atleast_2d(axs)

        for i in range(num_components):
            row, col = divmod(i, cols)
            component = data[i].cpu().numpy()
            axs[row, col].imshow(component, cmap='viridis')#, vmin=global_min, vmax=global_max)
            axs[row, col].set_title(f'{key.replace("_", " ").capitalize()} {i + 1}')
            # axs[row, col].axis('off')

        for i in range(num_components, rows * cols):
            row, col = divmod(i, cols)
            axs[row, col].axis('off')

        plt.tight_layout()

        if save_plot:
            dir = run_dir('predictions')
            os.makedirs(dir, exist_ok=True)
            plt.savefig(f"{dir}/{key}_components.png", transparent=True, dpi=300)
            print(f"Saved {key} components image to '{dir}/{key}_components.png'")

        if show_plot:
            plt.show()

        plt.close()
# def plot_data(self, plot_data):
    #     channels, height, width = self.image_dims
    #     for key, data in plot_data.items():
    #     data = data.view(channels, height, width)
    #     for i in range(data.shape[0]):
    #         fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    #         ax.imshow(data[i].cpu().numpy(), cmap='viridis')
    #         ax.set_title(f'{key.replace('_', ' ').capitalize()}, {i} component')
    #         ax.axis('off')
    #
    #         if self.show_plot:
    #             plt.show()
    #         if self.save_plot:
    #             dir = run_dir('predictions')
    #             plt.savefig(f"{dir}/{key}_component_{i}.png", transparent=True, dpi=300)
    #             print(f"Saved {key} component {i} image to '{dir}{key}_component_{i}.png'")
    #
    #         plt.close()

    # def plot_data(self, plot_data):
    #     _, height, width = self.image_dims
    #
    #     plt = init_plot()
    #
    #     key, data = next(iter(plot_data.items()))
    #
    #     data = data.T.view(-1, height, width)
    #
    #     num_components = data.shape[0]
    #
    #     rows = (num_components + 2) // 3
    #
    #     if len(plot_data) == 1:
    #         fig, axs = plt.subplots(rows, 3, figsize=(9, 4.5 * rows), dpi=300)
    #         axs = np.atleast_2d(axs)
    #
    #         for i in range(num_components):
    #             row = i // 3
    #             col = i % 3
    #             component = data[i].cpu().numpy()
    #             axs[row, col].imshow(component, cmap='viridis')
    #             axs[row, col].set_title(f'{key.replace("_", ' ').capitalize()} {i+1}')
    #             axs[row, col].axis('off')
    #
    #         plt.tight_layout()
    #         if self.show_plot:
    #             plt.show()
    #         if self.save_plot:
    #             dir = run_dir('predictions')
    #             plt.savefig(f"{dir}/{key}-components.png", transparent=True, dpi=300)
    #             print(
    #                 f"Saved {key} components image to '{dir}/{key}_components.png'")
    #         plt.close()
    #
    #     else:
    #         for comp_idx in range(num_components):
    #             fig, axs = plt.subplots(1, len(plot_data), figsize=(3 * len(plot_data), 4.5), dpi=300)
    #             axs = np.atleast_1d(axs)
    #
    #             for idx, (key, data) in enumerate(plot_data.items()):
    #                 data = data.T.view(-1, height, width)
    #                 component = data[comp_idx].cpu().numpy()
    #                 axs[idx].imshow(component, cmap='viridis')
    #                 axs[idx].set_title(f'{key.replace("_", " ").capitalize()} {comp_idx+1}')
    #                 axs[idx].axis('off')
    #
    #             plt.tight_layout()
    #
    #             if self.show_plot:
    #                 plt.show()
    #             if self.save_plot:
    #                 dir = run_dir('predictions')
    #                 plt.savefig(f"{dir}/component_{comp_idx}.png", transparent=True, dpi=300)
    #                 print(
    #                     f"Saved {', '.join(list(plot_data.keys()))} component {comp_idx} image to '{dir}/{key}_component_{comp_idx}.png'")
    #
    #             plt.close()


def save_metrics(metrics, save_dir=None):
    import wandb
    if wandb.run is not None and save_dir is None:
        base_dir = os.path.join(wandb.run.dir.split('wandb')[0], 'results')
        sweep_id = wandb.run.dir.split('/')[-4].split('-')[-1]
        output_path = os.path.join(base_dir, f'sweep-{sweep_id}', "sweep_data.json")
    else:
        if save_dir is None:
            save_dir = './results'

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(
            project_root, 'experiments', os.environ["EXPERIMENT"], save_dir, os.environ["RUN_ID"], "sweep_data.json"
        )

    run_id = os.environ.get("RUN_ID", "default")
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}

    if run_id not in existing_data:
        existing_data[run_id] = {"metrics": {}}

    existing_data[run_id]["metrics"].update(metrics)

    print('Final metrics:')
    for key, value in metrics.items():
        print(f"\t{key} = {value}")

    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    print(f"Saved final metrics to {output_path}:")


def best_permutation_mse(model_A, true_A):
    import itertools
    col_permutations = itertools.permutations(range(model_A.size(1)))
    best_mse = float('inf')

    for perm in col_permutations:

        permuted_model_A = model_A[:, list(perm)]
        mean_mse = torch.mean((true_A - permuted_model_A).pow(2))
        mse = (true_A - permuted_model_A).pow(2)

        if mean_mse < best_mse:
            permutation = list(perm)
            best_mse = mean_mse

    return permutation, best_mse

# todo: separate plot utils
