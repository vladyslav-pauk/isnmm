import torch
import torch.nn as nn
import torchmetrics

from src.utils.plot_tools import plot_components


class ResidualNonlinearity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_r_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.fitter = LineFitter()

        self.tensors = {}

        self.reconstructed_sample = None
        self.linearly_mixed_sample = None
        self.observed_sample = None
        self.linearly_mixed_sample_true = None
        self.noiseless_sample_true = None

    def update(self, model_output=None, labels=None, linearly_mixed_sample=None, observed_sample=None, latent_sample_unmixed=None):

        self.reconstructed_sample = model_output["reconstructed_sample"].mean(0)
        self.linearly_mixed_sample = linearly_mixed_sample

        self.observed_sample = observed_sample
        self.linearly_mixed_sample_true = labels["linearly_mixed_sample"]
        self.noiseless_sample_true = labels["noiseless_sample"]

        r_squared_values = self.fitter.fit(self.linearly_mixed_sample, self.linearly_mixed_sample_true)
        self.sum_r_squared += r_squared_values.sum()
        self.count += r_squared_values.shape[0]

    def compute(self):
        r_squared_average = self.sum_r_squared / self.count
        self.sum_r_squared = torch.tensor(0.0)
        self.count = torch.tensor(0.0)

        # self.plot(show_plot=self.show_plot, save_plot=self.save_plot)

        return r_squared_average

    def plot(self, image_dims, show_plot=False, save_plot=False):
        plot_components(
            model={r'$x$': self.linearly_mixed_sample, r'$f(x)$': self.reconstructed_sample,},
            true={r'$x$': self.linearly_mixed_sample_true, r'$f(x)$': self.noiseless_sample_true},
            # residual=(self.linearly_mixed_sample_true, self.reconstructed_sample),
            # nr=(self.reconstructed_sample, self.noiseless_sample_true),
            scale=True,
            show_plot=show_plot,
            save_plot=save_plot,
            name=f"Nonlinearity"
        )


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


# task: residual nonlinearity and plot components shall go under the Fitter class
# task: crop outliers
# task: i can also check variance of the residual, that's more statistical metric
# task: matched latent correlation plot
