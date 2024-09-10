import math
import itertools
import matplotlib.pyplot as plt
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F


class LineFitter(nn.Module):
    def __init__(self):
        super(LineFitter, self).__init__()
        self.slopes = None
        self.intercepts = None

    def check_straightness(self, x, y):
        """
        Fit a line to y = function(x) and calculate the mean squared error (MSE)
        between y and the fitted line. Additionally, use the slope magnitude and R-squared as metrics.
        """
        # Flatten x and y
        x_flat = x.flatten()
        y_flat = y.flatten()

        # Stack y with ones for linear regression (adding bias term)
        Y = torch.stack([y_flat, torch.ones_like(y_flat)], dim=1)

        # Solve for slope and intercept using least squares
        params = torch.linalg.lstsq(Y, x_flat).solution
        slope, intercept = params[0], params[1]

        # Predict x using the fitted line
        x_pred = slope * y + intercept

        # Calculate the MSE between actual x and predicted x
        mse = F.mse_loss(x_pred, x)

        # Calculate the R-squared value
        ss_total = torch.sum((x - x.mean()) ** 2)
        ss_residual = torch.sum((x - x_pred) ** 2)
        r_squared = 1 - ss_residual / ss_total

        return r_squared, slope, intercept

    def fit(self, f, y):
        """
        Fits a separate line to each component of (x, f(x)) and stores the slope and intercept for each component.

        Args:
            f: A function that takes x as input and returns the output y.
            x: A tensor of inputs (each component will be fit separately).

        Returns:
            mse_values: A tensor containing the MSE for each component.
        """
        num_components = y.shape[-1]  # Get the number of components
        mse_values = []

        # Initialize lists to store the slopes and intercepts for each component
        slopes = []
        intercepts = []

        # Apply the function f to get the output y
        x = f(y)

        # Fit each component of x and y independently
        for i in range(num_components):
            x_component = x[:, i:i + 1]  # Select the i-th component of x
            y_component = y[:, i:i + 1]  # Select the i-th component of y

            # Perform the linear fit on the selected component
            mse, slope, intercept = self.check_straightness(x_component, y_component)

            # Store the slope and intercept for this component
            slopes.append(slope)
            intercepts.append(intercept)

            # Save the MSE for this component
            mse_values.append(mse)

        # Store the slopes and intercepts as tensors (but not nn.Parameters, so they refit every time)
        self.slopes = torch.tensor(slopes)  # Shape: [num_components]
        self.intercepts = torch.tensor(intercepts)  # Shape: [num_components]

        mse_values = torch.tensor(mse_values)

        self.rsquared = mse_values

        return mse_values

    def forward(self, x):
        """
        Forward pass applying the linear transformation for each component of x.
        Args:
            x: Input tensor with components to apply the fitted linear transformations.
        Returns:
            Tensor of transformed values based on the linear equations for each component.
        """
        if self.slopes is None or self.intercepts is None:
            raise ValueError("Slopes and intercepts are not initialized. Call fit first.")

        # Apply the linear transformations to each component separately
        transformed_components = [
            self.slopes[i] * x[:, i:i + 1] + self.intercepts[i]
            for i in range(x.shape[-1])
        ]

        return torch.cat(transformed_components, dim=-1)


def residual_nonlinearity(y_true, true_nonlinearity, model_nonlinearity, show_plot=False):

    residual_nonlinearity = lambda x: true_nonlinearity.inverse(model_nonlinearity(x))

    fitter = LineFitter()
    fitter.fit(residual_nonlinearity, y_true).unsqueeze(0)

    nonlinearity_plot = plot_components(
        y_true,
        true_nonlinearity=true_nonlinearity,
        inferred_nonlinearity=model_nonlinearity,
    )
    residual_nonlinearity_plot = plot_components(
        y_true,
        residual_nonlinearity=residual_nonlinearity,
        fitter=fitter,
        labels=fitter.rsquared
    )
    if show_plot:
        nonlinearity_plot.show()
        residual_nonlinearity_plot.show()
    else:
        wandb.log({"nonlinearity": nonlinearity_plot, "residual_nonlinearity": residual_nonlinearity_plot})
    nonlinearity_plot.close()
    residual_nonlinearity_plot.close()

    return fitter


def plot_components(x, labels=None, **kwargs):
    # todo: rescale each curve so that it fits in the plot (fitter has to refit exactly as original)
    # todo: make it a line instead of scatter
    import warnings
    warnings.filterwarnings("ignore", message=".*path .*")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 20,
        "font.size": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
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

            axes[i].scatter(x_component, y_component, label=k.replace('_', ' ').capitalize())

        if labels is not None:
            axes[i].text(0.5, 0.1, f"R-squared: {labels[i]:.4f}", horizontalalignment='center', verticalalignment='center',
                     transform=axes[i].transAxes)
        # axes[i].set_title(f'Component {i + 1} Nonlinearity')

    axes[0].legend()

    plt.tight_layout()

    return plt


def mse_matrix_db(A0, A_hat):
    min_mse = float('inf')
    perms = itertools.permutations(range(A0.shape[0]))
    for perm in perms:
        A_hat_permuted = A_hat[list(perm), :]
        mse = torch.mean(torch.sum((A0 - A_hat_permuted) ** 2, dim=1))
        if mse < min_mse:
            min_mse = mse

    mse_dB = 10 * torch.log10(min_mse)
    return mse_dB


def spectral_angle_distance(A0, A_hat):
    A0 = A0 / torch.norm(A0, dim=1, keepdim=True)
    A_hat = A_hat / torch.norm(A_hat, dim=1, keepdim=True)
    cosines = torch.sum(A0 * A_hat, dim=1)
    return torch.acos(cosines).mean()


def subspace_distance(S, U):
    import torch

    S_pseudo_inv = torch.linalg.pinv(S)

    I = torch.eye(S.shape[-1], device=S.device)
    P_s_orth = I - S_pseudo_inv @ S

    U_u, Q, V_u = torch.linalg.svd(U.T, full_matrices=False)
    Q_u = V_u.T

    matrix_product = Q_u @ P_s_orth

    singular_values = torch.linalg.svd(matrix_product)[1]

    norm_2 = torch.max(singular_values)
    return norm_2

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
