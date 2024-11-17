import matplotlib.pyplot as plt
import math

import wandb
import torch
import torch.nn as nn
import torchmetrics

from src.utils.wandb_tools import run_dir


class ResidualNonlinearity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False, log_plot=True, save_plot=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_r_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.fitter = LineFitter()

        self.show_plot = show_plot
        self.log_plot = log_plot
        self.save_plot = save_plot

    def update(self, model_output, labels, linearly_mixed_sample, observed_sample, latent_sample_unmixed):

        self.reconstructed_sample = model_output["reconstructed_sample"].mean(0)
        self.linearly_mixed_sample_true = labels["linearly_mixed_sample"]
        self.latent_sample_qr = labels["latent_sample_qr"]
        # self.latent_sample = model_output["latent_sample"].mean(0)

        self.latent_sample = latent_sample_unmixed
        self.latent_sample_true = labels["latent_sample"]
        self.linearly_mixed_sample = linearly_mixed_sample
        self.observed_sample = observed_sample
        self.noiseless_sample_true = labels["noiseless_sample"]

        r_squared_values = self.fitter.fit(self.linearly_mixed_sample, self.linearly_mixed_sample_true)
        self.sum_r_squared += r_squared_values.sum()
        self.count += r_squared_values.shape[0]

    def compute(self):
        self.plot(show_plot=self.show_plot)
        r_squared_average = self.sum_r_squared / self.count
        self.sum_r_squared = torch.tensor(0.0)
        self.count = torch.tensor(0.0)
        return r_squared_average

    def match_components(self, matrix_model, matrix_true):
        num_cols = matrix_model.size(1)
        import itertools
        col_permutations = itertools.permutations(range(num_cols))

        best_mse = float('inf')
        best_perm = None

        for perm in col_permutations:
            permuted_matrix = matrix_model[:, list(perm)]
            mse = torch.mean(torch.sum((matrix_true - permuted_matrix) ** 2, dim=1))

            if mse < best_mse:
                best_mse = mse
                best_perm = permuted_matrix

        return best_perm

    def plot(self, show_plot=False):

        plot_components(
            model=(self.reconstructed_sample, self.linearly_mixed_sample),
            true=(self.noiseless_sample_true, self.linearly_mixed_sample_true),
            # residual=(self.linearly_mixed_sample_true, self.reconstructed_sample),
            # nr=(self.reconstructed_sample, self.noiseless_sample_true),
            scale=True,
            show_plot=show_plot,
            save_plot=self.save_plot,
            name=f"model_true_nonlinearity"
        )
        # plot_components(
        #     model=(self.linearly_mixed_sample, self.reconstructed_sample),
        #     true=(self.linearly_mixed_sample_true, self.noiseless_sample_true),
        #     scale=True,
        #     show_plot=show_plot,
        #     name=f"Reconstruction vs True Noiseless"
        # )
        # plot_components(
        #     model=(self.linearly_mixed_sample_true, self.linearly_mixed_sample),
        #     fitter=(self.linearly_mixed_sample_true, self.fitter),
        #     labels=self.fitter.rsquared,
        #     scale=False,
        #     show_plot=show_plot,
        #     name=f"Residual Nonlinearity"
        # )

        plot_components(
            model=(self.latent_sample_true, self.match_components(self.latent_sample, self.latent_sample_true)),
            true=(torch.linspace(0, 1, 100).repeat(3, 1).t(), torch.linspace(0, 1, 100).repeat(3, 1).t()),
            scale=False,
            max_points=300,
            show_plot=show_plot,
            save_plot=self.save_plot,
            name=f"latent_correlation"
        )

        # plt.plot(self.latent_sample[:, 0], self.latent_sample[:, 1], 'o')
        # if show_plot:
        #     plt.show()
        # else:
        #     wandb.log({
        #         "Latent Space": plt
        #     })
        plt.close()


class LineFitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.slopes = None
        self.intercepts = None

    def check_straightness(self, x, y):
        x_flat, y_flat = x.flatten().to('cpu'), y.flatten().to('cpu')
        Y = torch.stack([y_flat, torch.ones_like(y_flat)], dim=1)
        params = torch.linalg.lstsq(Y, x_flat).solution
        slope, intercept = params[0], params[1]
        x_pred = slope * y + intercept
        mse = nn.functional.mse_loss(x_pred, x)

        ss_total = torch.sum((x - x.mean()) ** 2)
        ss_residual = torch.sum((x - x_pred) ** 2)
        r_squared = 1 - ss_residual / ss_total

        return r_squared, slope, intercept

    def fit(self, f, y):
        num_components = y.shape[-1]
        slopes, intercepts, mse_values = [], [], []
        x = f

        for i in range(num_components):
            x_component, y_component = x[:, i:i + 1], y[:, i:i + 1]
            r_squared, slope, intercept = self.check_straightness(x_component, y_component)
            slopes.append(slope.item())
            intercepts.append(intercept.item())
            mse_values.append(r_squared.item())

        self.slopes = torch.tensor(slopes)
        self.intercepts = torch.tensor(intercepts)
        self.rsquared = torch.tensor(mse_values)

        return self.rsquared

    def forward(self, x):
        if self.slopes is None or self.intercepts is None:
            raise ValueError("Slopes and intercepts are not initialized. Call fit first.")
        transformed_components = [self.slopes[i] * x[:, i:i + 1] + self.intercepts[i] for i in range(x.shape[-1])]
        return torch.cat(transformed_components, dim=-1)


def plot_components(labels=None, scale=False, show_plot=False, save_plot=False, name=None, max_points=10e8, **kwargs):
    import warnings
    warnings.filterwarnings("ignore", message=".*path .*")
    font_size = 24
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

    num_components = kwargs[list(kwargs.keys())[0]][0].shape[-1]
    n_cols = 3 + 1e-5
    n_rows = int(num_components // n_cols) + 1
    fig, axes = plt.subplots(n_rows, int(n_cols), figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()
    markers = ['o', 'o']

    for i in range(num_components):
        for j, (k, (x, v)) in enumerate(kwargs.items()):
            x_component = x[..., i].clone().detach().cpu().numpy()
            if callable(v):
                y_component = v(x)[..., i].clone().detach().cpu()
            else:
                y_component = v[..., i].clone().detach().cpu()
                if (torch.max(y_component) - torch.min(y_component)).any() < 1e-6:
                    continue

            if len(x_component) > max_points:
                indices = torch.randperm(len(x_component))[:int(max_points)]
                x_component = x_component[indices]
                y_component = y_component[indices]

            marker = markers[j % len(markers)]
            axes[i].scatter(
                visual_normalization(torch.tensor(x_component)) if scale else x_component,
                visual_normalization(y_component) if scale else y_component,
                label=k.replace('_', ' ').capitalize(),
                marker=marker
            )
        if labels is not None:
            if show_plot:
                print(f"R-squared for component {i}: {labels[i]:.4f}")
    plt.tight_layout()

    plt.xlabel(r"$A z$")
    plt.ylabel(r"$f(A z)$")

    if show_plot:
        plt.show()
    else:
        wandb.log({
            name: plt
        })

    if save_plot:
        dir = run_dir('predictions')
        path = f"{dir}/{name}.png"

        fig.savefig(path, transparent=True)
    plt.close()
    return plt


def visual_normalization(x):
    bound = 10
    x = x - torch.min(x)
    x = x / (torch.max(x)) * bound
    return x


# task: residual nonlinearity and plot components shall go under the Fitter class
# task: crop outliers
# task: i can also check variance of the residual, that's more statistical metric
# task: matched latent correlation plot
