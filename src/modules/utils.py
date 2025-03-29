import os
import json
import numpy as np
import torch
import itertools

from src.utils.plot_tools import init_plot
from src.utils.wandb_tools import run_dir
import src.model as model_package


def dict_to_str(d):
    return '_'.join([f'{value}' for key, value in d.items() if value is not None])


def unmix(latent_sample, latent_dim, model=None):

    dataset_size = latent_sample.size(0)

    unmixing_model = getattr(model_package, model).Model
    unmixing = unmixing_model(
        latent_dim=latent_dim,
        dataset_size=dataset_size
    )

    latent_sample, mixing_matrix = unmixing.estimate_abundances(latent_sample.squeeze().cpu().detach())
    # latent_sample = latent_sample / latent_sample.sum(dim=-1, keepdim=True)

    # unmixing.plot_multiple_abundances(latent_sample, [0,1,2,3,4,5,6,7,8,9])
    # unmixing.plot_mse_image(rows=100, cols=10)

    latent_sample = latent_sample / latent_sample.sum(dim=-1, keepdim=True)

    return latent_sample, mixing_matrix


def permute(sample, true_sample):
    permutation, _ = best_permutation_mse(sample, true_sample)
    sample = sample[:, permutation]
    return sample, permutation


def best_permutation_mse(model_A, true_A):
    col_permutations = itertools.permutations(range(model_A.size(1)))
    best_mse = float('inf')

    for perm in col_permutations:

        permuted_model_A = model_A[:, list(perm)]
        mse = (true_A - permuted_model_A).pow(2)
        mean_mse = torch.mean(mse)

        if mean_mse < best_mse:
            best_permutation = list(perm)
            best_mse = mean_mse

    return best_permutation, best_mse



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


def save_dataset(new_data, new_latents, image_dims):
    """
    Appends new data and latents to the existing datasets stored in .pth files.

    Args:
        new_data (torch.Tensor): The new data tensor to append.
        new_latents (torch.Tensor): The new latent tensor to append.
        data_path (str): Path to the .pth file storing data.
        latents_path (str): Path to the .pth file storing latents.

    Returns:
        None
    """

    data_path = "/Users/home/Work/Projects/nisca/datasets/mri/DWland_DCE/dwi_tensordata/dwi_observed_tensordata.pth"
    latents_path = "/Users/home/Work/Projects/nisca/datasets/mri/DWland_DCE/dwi_tensordata/dwi_latent_tensordata.pth"

    _, height, width = image_dims
    new_data = new_data.T.reshape(-1, height, width)
    new_latents = new_latents.T.reshape(-1, height, width)
    new_data = new_data.unsqueeze(0)
    new_latents = new_latents.unsqueeze(0)

    # Ensure new_data and new_latents have the same number of samples
    assert new_data.size(0) == new_latents.size(0), "New data and latents must have the same number of samples."

    # Load existing data
    if os.path.exists(data_path):
        existing_data = torch.load(data_path)
    else:
        existing_data = torch.empty((0, *new_data.size()[1:]))  # Empty tensor with correct shape

    if os.path.exists(latents_path):
        existing_latents = torch.load(latents_path)
    else:
        existing_latents = torch.empty((0, *new_latents.size()[1:]))  # Empty tensor with correct shape

    # Append new data and latents
    updated_data = torch.cat([existing_data, new_data], dim=0)
    updated_latents = torch.cat([existing_latents, new_latents], dim=0)

    # Save back to the .pth files
    torch.save(updated_data, data_path)
    torch.save(updated_latents, latents_path)

    print(f"Data and latents successfully saved. Total samples: {updated_data.size(0)}")


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
    # todo: make print('{stage} metrics:')
    for key, value in metrics.items():
        print(f"\t{key} = {"{:.2f}".format(value)}")

    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=2)
    print(f"Saved final metrics to {output_path}:")

# todo: separate plot utils
