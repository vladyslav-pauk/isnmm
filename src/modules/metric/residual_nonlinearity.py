import numpy as np
import os

import torch
import torch.nn as nn
import torchmetrics

from src.utils.wandb_tools import run_dir
from src.utils.utils import init_plot

from src.modules.utils import unmix


class ResidualNonlinearity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, unmixing=None, show_plot=False, log_plot=False, save_plot=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_r_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.fitter = LineFitter()

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.log_plot = log_plot
        self.unmixing = unmixing

    def update(self, model_output=None, labels=None, linearly_mixed_sample=None, observed_sample=None, latent_sample_unmixed=None):

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
        self.latent_sample = unmix({"latent_sample": self.latent_sample}, self.unmixing, self.latent_sample_true.shape[-1])["latent_sample"]

        self.plot(show_plot=self.show_plot, save_plot=self.save_plot)

        r_squared_average = self.sum_r_squared / self.count
        self.sum_r_squared = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

        return r_squared_average

    def _match_components(self, matrix_model, matrix_true):
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

    def plot(self, show_plot=False, save_plot=False):

        plot_components(
            model=(self.reconstructed_sample, self.linearly_mixed_sample),
            true=(self.noiseless_sample_true, self.linearly_mixed_sample_true),
            # residual=(self.linearly_mixed_sample_true, self.reconstructed_sample),
            # nr=(self.reconstructed_sample, self.noiseless_sample_true),
            scale=True,
            show_plot=show_plot,
            save_plot=save_plot,
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
            model=(self.latent_sample_true, self._match_components(self.latent_sample, self.latent_sample_true)),
            true=(torch.linspace(0, 1, 100).repeat(3, 1).t(), torch.linspace(0, 1, 100).repeat(3, 1).t()),
            scale=False,
            max_points=300,
            show_plot=show_plot,
            save_plot=save_plot,
            name=f"latent_correlation"
        )

        # plt.plot(self.latent_sample[:, 0], self.latent_sample[:, 1], 'o')
        # if show_plot:
        #     plt.show()
        # else:
        #     wandb.log({
        #         "Latent Space": plt
        #     })



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
    import os
    plt = init_plot()
    A4_WIDTH = 8.27

    num_components = kwargs[list(kwargs.keys())[0]][0].shape[-1]
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
        for j, (k, (x, v)) in enumerate(kwargs.items()):
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

        axes[i].set_title(f"Component {i + 1}")
        axes[i].legend()
        axes[i].grid(True)

        axes[i].set_xlabel(r"$A z$", fontsize=10)
        axes[i].set_ylabel(r"$f(A z)$", fontsize=10)

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
