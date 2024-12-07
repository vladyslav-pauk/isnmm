import os
import numpy as np
import torch

from src.utils.wandb_tools import run_dir


def init_plot():
    import warnings
    warnings.filterwarnings("ignore", message=".*path .*")

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": ["Computer Modern Roman"],
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    # todo: my color palette

    return plt


def plot_components(labels=None, scale=False, show_plot=False, save_plot=False, name=None, max_points=10e8, diagonal=False, **kwargs):

    import os
    plt = init_plot()
    A4_WIDTH = 8.27

    first_key = list(kwargs.keys())[0]
    first_pair = kwargs[first_key]
    num_components = first_pair[list(first_pair.keys())[0]].shape[-1]

    n_cols = next(i for i in range(3, 6) if num_components % i == 0)
    n_rows = (num_components + n_cols - 1) // n_cols

    aspect_ratio = 1.0
    fig_width = A4_WIDTH
    fig_height = fig_width * n_rows / n_cols * aspect_ratio

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=300)
    axes = np.atleast_1d(axes.flatten())

    markers = ['o', 'x', '^', 's', 'd']
    marker_size = 9

    for i in range(num_components):
        for j, (k, pair) in enumerate(kwargs.items()):
            x, v = pair.values()
            kx, kv = pair.keys()
            x_component = x[..., i].clone().detach().cpu().numpy()
            y_component = v(x)[..., i].clone().detach().cpu().numpy() if callable(v) else v[..., i].clone().detach().cpu().numpy()

            if (torch.max(torch.tensor(y_component)) - torch.min(torch.tensor(y_component))).item() < 1e-6:
                continue

            if len(x_component) > max_points:
                indices = torch.randperm(len(x_component))[:int(max_points)]
                x_component = x_component[indices]
                y_component = y_component[indices]

            marker = markers[j % len(markers)]
            axes[i].scatter(
                visual_normalization(torch.tensor(x_component)) if scale else x_component,
                visual_normalization(torch.tensor(y_component)) if scale else y_component,
                label=k.replace('_', ' ').capitalize(),
                marker=marker,
                s=marker_size
            )

            axes[i].set_aspect('equal')

        if diagonal:
            min_val = 0 #min(x_component.min(), y_component.min())
            max_val = 1 #max(x_component.max(), y_component.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], color='orange', linestyle='--', linewidth=1, label='Diagonal')

        axes[i].set_title(f"{name} {i + 1}")
        # axes[i].legend()
        axes[i].grid(True)

        axes[i].set_xlabel(f"{kx}", fontsize=10)
        axes[i].set_ylabel(f"{kv}", fontsize=10)

    for i in range(num_components, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_plot:
        dir = run_dir('predictions')
        os.makedirs(dir, exist_ok=True)
        path = f"{dir}/{name}.png"
        fig.savefig(path, transparent=True)
        print(f"Saved {name} plot to '{path}'")

    if show_plot:
        plt.show()

    plt.close()
    return plt, axes


def plot_image(tensors, image_dims, show_plot=False, save_plot=False):
    for key, value in tensors.items():
        data = {key: value}

        # if not show_plot and not save_plot:
        #     return None, None

        plt = init_plot()

        A4_WIDTH = 8.27

        _, height, width = image_dims

        global_min = 0 #all_data.min().item()
        global_max = 1 #all_data.max().item()
        # print(f"Global normalization: min={global_min}, max={global_max}")

        for key, data in data.items():
            data = data.T.reshape(-1, height, width)
            num_components = data.shape[0]

            cols = next(i for i in range(3, 6) if num_components % i == 0)
            rows = (num_components + cols - 1) // cols

            aspect_ratio = 1.0
            fig_width = A4_WIDTH
            fig_height = fig_width * rows / cols * aspect_ratio

            fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=300)
            axs = np.atleast_1d(axs.flatten())

            for i in range(num_components):
                # row, col = divmod(i, cols)
                component = data[i].cpu().numpy()
                axs[i].imshow(component, cmap='viridis')#, vmin=global_min, vmax=global_max)
                axs[i].set_title(f'{key.replace("_", " ").capitalize()} {i + 1}')
                # axs[row, col].axis('off')

            for i in range(num_components, rows * cols):
                axs[i].axis('off')

            plt.tight_layout()

            if save_plot:
                dir = run_dir('predictions')
                os.makedirs(dir, exist_ok=True)
                plt.savefig(f"{dir}/{key}_components.png", transparent=True, dpi=300)
                print(f"Saved {key} components image to '{dir}/{key}_components.png'")

            if show_plot:
                plt.show()

            plot = plt, axs
            plt.close()

    return plot


def visual_normalization(x):
    bound = 10
    x = x - torch.min(x)
    x = x / (torch.max(x)) * bound
    return x
