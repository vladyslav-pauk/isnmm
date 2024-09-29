import matplotlib.pyplot as plt
import math


def plot_components(x, labels=None, scale=False, **kwargs):
    # todo: rescale each curve so that it fits in the plot (fitter has to refit exactly as original)
    # todo: make it a line instead of scatter
    # todo: adjust styling for the paper
    import warnings
    warnings.filterwarnings("ignore", message=".*path .*")
    font_size = 12
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": font_size,
        "font.size": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.latex.preamble": r"\usepackage{amsmath}"
    })
    num_components = x.shape[-1]

    n_cols = math.ceil(math.sqrt(num_components))
    n_rows = math.ceil(num_components / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i in range(num_components):
        x_component = x[..., i].detach().cpu().numpy()

        for k, v in kwargs.items():
            if callable(v):
                y_component = v(x)[..., i].detach().cpu().numpy()
            else:
                y_component = v[..., i].detach().cpu().numpy()

            axes[i].scatter(x_component, visual_normalization(y_component) if scale else y_component, label=k.replace('_', ' ').capitalize())

        if labels is not None:
            axes[i].text(0.5, 0.1, f"R-squared: {labels[i]:.4f}", horizontalalignment='center', verticalalignment='center',
                     transform=axes[i].transAxes)
        # axes[i].set_title(f'Component {i + 1} Nonlinearity')

    # axes[0].legend()

    plt.tight_layout()

    return plt

def visual_normalization(x):
    import numpy as np
    bound = 10
    x = x - np.min(x)  # Min normalization
    x = x / np.max(x) * bound  # Scale to the range [0, bound]
    return x