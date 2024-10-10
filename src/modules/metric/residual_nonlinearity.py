import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import matplotlib.pyplot as plt
import math
import torch
import wandb


class ResidualNonlinearity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_r_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.fitter = LineFitter()

        self.show_plot = show_plot

    def update(self, model_output, labels, linearly_mixed_sample, observed_sample):

        self.reconstructed_sample = model_output["reconstructed_sample"].mean(0)
        self.linearly_mixed_sample_true = labels["linearly_mixed_sample"]
        self.latent_sample_qr = labels["latent_sample_qr"]
        self.latent_sample = model_output["latent_sample"].mean(0)
        self.latent_sample_true = labels["latent_sample"]
        self.linearly_mixed_sample = linearly_mixed_sample
        self.observed_sample = observed_sample

        r_squared_values = self.fitter.fit(self.linearly_mixed_sample, self.linearly_mixed_sample_true)
        self.sum_r_squared += r_squared_values.sum()
        self.count += r_squared_values.shape[0]

    def compute(self):
        self.plot(show_plot=self.show_plot)
        return self.sum_r_squared / self.count

    def plot(self, show_plot=False):

        plot_components(
            self.observed_sample,
            model=self.linearly_mixed_sample,
            true=self.linearly_mixed_sample_true,
            scale=True,
            show_plot=show_plot,
            name=f"Model vs True Nonlinearity"
        )
        plot_components(
            self.linearly_mixed_sample_true,
            model=self.linearly_mixed_sample,
            fitter=self.fitter,
            labels=self.fitter.rsquared,
            scale=True,
            show_plot=show_plot,
            name=f"Residual Nonlinearity"
        )
        # plot_components(
        #     self.latent_sample_true,
        #     model=self.latent_sample,
        #     scale=True,
        #     show_plot=show_plot,
        #     name=f"Latent Correlation"
        # )


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
        mse = F.mse_loss(x_pred, x)

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


def plot_components(x, labels=None, scale=False, show_plot=False, name=None, **kwargs):
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
        x_component = x[..., i].clone().detach().cpu().numpy()

        for k, v in kwargs.items():
            if callable(v):
                y_component = v(x)[..., i].clone().detach().cpu()
            else:
                y_component = v[..., i].clone().detach().cpu()
                if (torch.max(y_component) - torch.min(y_component)).any() < 1e-6:
                    print(f"Warning: y-component {i} is constant")

            axes[i].scatter(x_component, visual_normalization(y_component) if scale else y_component, label=k.replace('_', ' ').capitalize())

        if labels is not None:
            axes[i].text(0.5, 0.1, f"R-squared: {labels[i]:.4f}", horizontalalignment='center', verticalalignment='center',
                     transform=axes[i].transAxes)
        # axes[i].set_title(f'Component {i + 1} Nonlinearity')
        # axes[i].set_xlabel(f'Linearly mixed component {i + 1}')
        # axes[i].set_ylabel(f'Nonlinearly transformed component {i + 1}')
        # axes[i].set_yticklabels([])
        # axes[i].set_xticklabels([])
        # axes[i].set_xticks([])
        # axes[i].set_yticks([])

    # axes[0].legend()

    plt.tight_layout()

    if show_plot:
        plt.show()
    else:
        wandb.log({
            name: plt
        })
    plt.close()

    return plt


def visual_normalization(x):
    bound = 10
    x = x - torch.min(x)
    x = x / (torch.max(x)) * bound
    return x


# todo: residual nonlinearity and plot components shall go under the Fitter class
# todo: matrix distance class; vector (sub)space (set) distance class
# todo: crop outliers
# todo: make ResNon not depend on the function, but rather on the values for z_true
#    i.e. pass not f_true and A_true, but the values for f_true(A_true @ z_true), and A_true @ z_true
