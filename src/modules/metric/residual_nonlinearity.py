import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from src.helpers.plotter import plot_components


class ResidualNonlinearity(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_r_squared", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.fitter = LineFitter()

        self.show_plot = show_plot

    def update(self, data, reconstructed_sample, linearly_mixed_sample):
        self.data = data
        self.reconstructed_sample = reconstructed_sample
        self.linearly_mixed_sample = linearly_mixed_sample

        r_squared_values = self.fitter.fit(self.reconstructed_sample, data)
        self.sum_r_squared += r_squared_values.sum()
        self.count += r_squared_values.shape[0]

    def compute(self):
        self.plot(show_plot=self.show_plot)
        return self.sum_r_squared / self.count

    def plot(self, show_plot=False):

        nonlinearity_plot = plot_components(
            self.linearly_mixed_sample,
            inferred_nonlinearity=self.reconstructed_sample,
            true_nonlinearity=self.data,
            scale=True
        )
        residual_nonlinearity_plot = plot_components(
            self.data,
            residual_nonlinearity=self.reconstructed_sample,
            fitter=self.fitter,
            labels=self.fitter.rsquared,
        )

        if show_plot:
            nonlinearity_plot.show()
            residual_nonlinearity_plot.show()
        else:
            wandb.log({
                f"Residual Nonlinearity": residual_nonlinearity_plot
            })
            wandb.log({
                "Model and True Nonlinearity": nonlinearity_plot
            })

        nonlinearity_plot.close()
        residual_nonlinearity_plot.close()


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

#
#
# def mse_matrix_db(A0, A_hat):
#     min_mse = torch.tensor(float('inf'))
#     perms = itertools.permutations(range(A0.shape[0]))
#     for perm in perms:
#         A_hat_permuted = A_hat[list(perm), :]
#         mse = torch.mean(torch.sum((A0 - A_hat_permuted) ** 2, dim=1))
#         if mse < min_mse:
#             min_mse = mse
#
#     mse_dB = 10 * torch.log10(min_mse)
#     return mse_dB
#
#
# def spectral_angle_distance(A0, A_hat):
#     A0 = A0 / torch.norm(A0, dim=1, keepdim=True)
#     A_hat = A_hat / torch.norm(A_hat, dim=1, keepdim=True)
#     cosines = torch.sum(A0 * A_hat, dim=1)
#     return torch.acos(cosines).mean()
#
#
# def subspace_distance(S, U):
#     import torch
#
#     S_pseudo_inv = torch.linalg.pinv(S)
#
#     I = torch.eye(S.shape[-1], device=S.device)
#     P_s_orth = I - S_pseudo_inv @ S
#
#     U_u, Q, V_u = torch.linalg.svd(U.T, full_matrices=False)
#     Q_u = V_u.T
#
#     matrix_product = Q_u @ P_s_orth
#
#     singular_values = torch.linalg.svd(matrix_product)[1]
#
#     norm_2 = torch.max(singular_values)
#     return norm_2

# todo: residual nonlinearity and plot components shall go under the Fitter class
# todo: matrix distance class; vector (sub)space (set) distance class
# def plot_components(x, **kwargs):
#     import warnings
#     warnings.filterwarnings("ignore", message=".*path .*")
#     plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "serif",
#         "font.serif": ["Computer Modern Roman"],
#         "axes.labelsize": 20,
#         "font.size": 20,
#         "legend.fontsize": 20,
#         "xtick.labelsize": 20,
#         "ytick.labelsize": 20,
#         "figure.dpi": 300,
#         "savefig.dpi": 300,
#         "text.latex.preamble": r"\usepackage{amsmath}"
#     })
#
#     num_components = x.shape[-1]
#     n_cols = math.ceil(math.sqrt(num_components))
#     n_rows = math.ceil(num_components / n_cols)
#
#     # Create a figure and a set of subplots
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
#     axes = axes.flatten()  # Flatten the axes for easier iteration
#
#     for i in range(num_components):
#         x_component = x[:, i:i + 1]  # Select the i-th component of x
#
#         for k, v in kwargs.items():
#             if callable(v):
#                 # Apply the function v (e.g., fitter or residual_nonlinearity) on x_component
#                 v_result = v(x_component)
#             else:
#                 v_result = v[:, i:i + 1]  # If not callable, assume it's a tensor
#
#             label = k.replace('_', ' ').capitalize()
#             axes[i].scatter(x_component.detach().cpu().numpy(), v_result.detach().cpu().numpy(), label=label)
#
#         axes[i].set_title(f'Component {i + 1} Nonlinearity')
#         axes[i].legend()
#
#     # Remove any unused axes
#     for j in range(num_components, len(axes)):
#         fig.delaxes(axes[j])
#
#     plt.tight_layout()
#
#     return plt
#     # todo: crop outliers
        # todo: make ResNon not depend on the function, but rather on the values for z_true
        #    i.e. pass not f_true and A_true, but the values for f_true(A_true @ z_true), and A_true @ z_true
