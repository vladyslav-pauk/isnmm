import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import lsq_linear

from src.helpers.matrix_tools import kmeans_torch, match_components, spectral_angle_mapper


class Model:
    def __init__(self, observed_dim, latent_dim, dataset_size):
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.dataset_size = dataset_size
        self.device = torch.device('cpu')

        self.linear_mixture_est = None

    def fit(self, x):
        x = x.T
        x_mean, x_eigenvectors = self._center_and_decompose(x)
        x_projected = self._project_data(x, x_mean, x_eigenvectors)
        vertices = self._construct_convex_hull(x_projected)
        alpha_est = self._estimate_alpha(vertices)
        self.linear_mixture_est, H, g = self._estimate_endmembers(x_eigenvectors, alpha_est, x_mean)
        return self.linear_mixture_est, H, g, x_mean, x_eigenvectors

    def _center_and_decompose(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        covariance_matrix = x_centered @ x_centered.T / (self.dataset_size - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        x_eigenvectors = eigenvectors[:, sorted_indices[:self.latent_dim - 1]]
        assert torch.linalg.matrix_rank(x_eigenvectors) == self.latent_dim - 1
        return x_mean, x_eigenvectors

    def _project_data(self, x, x_mean, x_eigenvectors):
        x_tilde = torch.pinverse(x_eigenvectors) @ (x - x_mean)
        return x_tilde.T

    def _construct_convex_hull(self, x_tilde):
        hull = ConvexHull(x_tilde.cpu().numpy())
        return x_tilde[hull.vertices]

    def _estimate_alpha(self, vertices):
        if vertices.shape[0] > self.latent_dim:
            return kmeans_torch(vertices, self.latent_dim, num_iters=10).T
        else:
            return vertices[:self.latent_dim].T

    def _estimate_endmembers(self, x_eigenvectors, alpha_est, x_mean):
        A_est = x_eigenvectors @ alpha_est + x_mean
        B = alpha_est[:, :-1] - alpha_est[:, [-1]]
        B_inv = torch.inverse(B)
        H = B_inv
        g = B_inv @ alpha_est[:, -1]
        return A_est, H, g

    def estimate_abundances(self, x):
        self.linear_mixture_est, H, g, x_mean, x_eigenvectors = self.fit(x)
        x_projected = self._project_data(x.T, x_mean, x_eigenvectors)
        s_tilde = H @ x_projected.T - g[:, None]
        s_last = 1 - s_tilde.sum(dim=0, keepdim=True)
        s_est = torch.cat((s_tilde, s_last), dim=0).T
        return torch.clamp(s_est, min=0)

    # def estimate_abundances(self, latent_sample_mixed):
    #     linear_mixture_est, _, _, _, _ = self.fit(latent_sample_mixed)
    #     latent_sample_n = []
    #     for x_n in latent_sample_mixed:
    #         x_n_np = x_n.cpu().numpy().astype(np.float64)
    #         res = lsq_linear(linear_mixture_est, x_n_np, bounds=(0, np.inf), method='trf')
    #         if res.success:
    #             latent_sample_n.append(res.x)
    #         else:
    #             print("Warning: Solver did not converge for a sample.")
    #             latent_sample_n.append(np.zeros(linear_mixture_est.shape[1]))
    #     return torch.tensor(np.array(latent_sample_n))

    # def estimate_abundances(self):
    #     """
    #     Estimates the abundances using Non-Negative Least Squares (NNLS). (yields error when used with CNAE training)
    #     """
    #     self.s_est = torch.zeros(self.L, self.N, device=self.device)  # Shape: (L, N)
    #     A_est_np = self.linear_mixture_est.cpu().numpy()  # Shape: (M, N)
    #
    #     for n in range(self.L):
    #         x_n = self.X[n].cpu().numpy()  # Shape: (M,)
    #         s_n, _ = nnls(A_est_np, x_n)
    #         self.s_est[n] = torch.from_numpy(s_n)
    #
    #     # Normalize abundances to sum to one
    #     sums = self.s_est.sum(dim=1, keepdim=True)  # Shape: (L, 1)
    #     self.s_est = self.s_est / sums
    #     return self.s_est

    def compute_metrics(self, linear_mixture_true, latent_sample_true, latent_sample):
        self.latent_sample_true = latent_sample_true
        self.linear_mixture_est_matched, self.s_est_matched = match_components(linear_mixture_true, self.linear_mixture_est, latent_sample)

        self.mean_sam_linear_mixture = spectral_angle_mapper(linear_mixture_true, self.linear_mixture_est_matched)
        # from src.modules.metric.spectral_angle import SpectralAngle
        # sam = SpectralAngle()
        # sam.update(self.linear_mixture_est_matched, linear_mixture_true)
        # self.mean_sam_linear_mixture = sam.compute()
        print(f"Mean SAM (Endmembers): "
              f"{self.mean_sam_linear_mixture.item() * 180 / np.pi:.2f} degrees")

        self.mean_sam_latent_sample = spectral_angle_mapper(latent_sample_true.T, self.s_est_matched.T)
        print(f"Mean SAM (Abundances): "
              f"{self.mean_sam_latent_sample.item() * 180 / np.pi:.2f} degrees")

        return self.mean_sam_linear_mixture, self.mean_sam_latent_sample

    def plot_mse_image(self, rows, cols):
        """
        Plots the per-pixel MSE image.

        Parameters:
        rows (int): Number of rows in the image grid.
        cols (int): Number of columns in the image grid.
        """
        if rows * cols != self.dataset_size:
            raise ValueError("rows * cols must be equal to the number of pixels L.")

        # Reshape the mse_per_pixel into a 2D array
        mse_image = torch.mean((self.latent_sample_true - self.s_est_matched) ** 2, dim=1).cpu().numpy().reshape(rows, cols)

        # Plot the MSE image
        plt.figure(figsize=(10, 6))
        plt.imshow(mse_image, cmap='hot', interpolation='nearest')
        plt.colorbar(label='MSE')
        plt.title('Per-pixel MSE between True and Estimated Abundances')
        plt.xlabel('Pixel Column Index')
        plt.ylabel('Pixel Row Index')
        plt.show()

    def plot_endmembers(self, linear_mixture, pixel_id=0):
        """
        Plots the true and estimated endmembers after matching.

        Parameters:
        pixel_id (int): Index of the endmember to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(linear_mixture[:, pixel_id].cpu().numpy(), label=f'True Endmember {pixel_id + 1}')
        plt.plot(self.linear_mixture_est_matched[:, pixel_id].detach().cpu().numpy(), '--',
                 label=f'Estimated Endmember Matched to {pixel_id + 1}')
        plt.legend()
        plt.title(f'Comparison of True and Estimated Endmember {pixel_id + 1} after Matching')
        plt.xlabel('Spectral Band')
        plt.ylabel('Reflectance')
        plt.show()

    def plot_abundances(self, abundances, pixel_id=0):
        """
        Plots the true and estimated abundances for a given pixel.

        Parameters:
        pixel_id (int): Index of the pixel to plot.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.latent_dim), abundances[pixel_id].cpu().numpy(), alpha=0.7, label='True Abundances')
        plt.bar(range(self.latent_dim), self.s_est_matched[pixel_id].cpu().numpy(), alpha=0.7, label='Estimated Abundances')
        plt.xticks(range(self.latent_dim))
        plt.legend()
        plt.title(f'Comparison of True and Estimated Abundances for Pixel {pixel_id + 1} after Matching')
        plt.xlabel('Endmember Index')
        plt.ylabel('Abundance')
        plt.show()

    def plot_multiple_abundances(self, abundances, pixel_indices):
        """
        Plots the true and estimated abundances for multiple pixel indices in a grid.

        Parameters:
        pixel_indices (list of int): A list of pixel indices to plot.
        """
        num_pixels = len(pixel_indices)
        cols = 2  # Number of columns in the grid
        rows = (num_pixels + 1) // cols  # Calculate the number of rows needed

        fig, axs = plt.subplots(rows, cols, figsize=(12, rows * 4))
        axs = axs.flatten()

        for i, pixel_index in enumerate(pixel_indices):
            if pixel_index < 0 or pixel_index >= self.dataset_size:
                print(f"Pixel index {pixel_index} is out of bounds.")
                continue

            axs[i].bar(range(self.latent_dim), abundances[pixel_index].cpu().numpy(), alpha=0.7, label='True Abundances')
            axs[i].bar(range(self.latent_dim), self.s_est_matched[pixel_index].cpu().numpy(), alpha=0.7,
                       label='Estimated Abundances')
            axs[i].set_xticks(range(self.latent_dim))
            axs[i].set_title(f'Pixel {pixel_index + 1}')
            axs[i].set_xlabel('Endmember Index')
            axs[i].set_ylabel('Abundance')
            axs[i].legend()

        # Hide any unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    M = 100
    N = 4
    L = 10000

    A_true = torch.rand(M, N)

    alpha = torch.ones(N)
    abundances_true = torch.distributions.Dirichlet(alpha).sample((L,))
    assert torch.all(abundances_true >= 0)
    assert torch.allclose(abundances_true.sum(dim=1), torch.ones(L))

    X = abundances_true @ A_true.T

    model = Model(M, N, L)

    latent_sample = model.estimate_abundances(X)
    model.compute_metrics(A_true, abundances_true, latent_sample)

    # model.plot_mse_image(rows=100, cols=100)
    model.plot_endmembers(A_true, pixel_id=0)
    # model.plot_abundances(abundances_true, pixel_id=0)
    pixel_indices = [0, 9, 99, 199]
    model.plot_multiple_abundances(abundances_true, pixel_indices)

    # # List of SNR values in dB
    # snr_values = [10, 15, 20, 25, 30, 50, 100]
    #
    # # Arrays to store results
    # latent_mse_db_avg = []
    # latent_mse_db_std = []
    #
    # # Initialize the model parameters
    # M = 20  # Number of spectral bands
    # N = 4  # Number of endmembers
    # L = 1000  # Number of pixels
    # seed = 0
    #
    # # Number of runs for averaging
    # num_runs = 5  # Increase this number for better statistical significance
    #
    # for snr_db in snr_values:
    #     mse_values = []
    #     for run in range(num_runs):
    #         model = MVES(M=M, N=N, L=L, seed=0)
    #
    #         # Generate synthetic data
    #         model.generate_data()
    #
    #         # Add noise at the specified SNR
    #         snr_db = 20  # Example SNR value
    #         # model.add_noise(snr_db)
    #
    #         # Perform unmixing
    #         model.unmix()
    #
    #         # **Important**: Compute metrics before computing per-pixel MSE
    #         model.compute_metrics()
    #
    #         # Now compute per-pixel MSE
    #         model.compute_per_pixel_mse()
    #
    #         # Compute average MSE over all pixels
    #         mse_avg = torch.mean(model.mse_per_pixel).item()
    #
    #         # Convert MSE to dB
    #         mse_db = 10 * np.log10(mse_avg)
    #
    #         mse_values.append(mse_db)
    #
    #     # Compute average and standard deviation over runs
    #     latent_mse_db_avg.append(np.mean(mse_values))
    #     latent_mse_db_std.append(np.std(mse_values))
    #
    # # Convert results to numpy arrays
    # snr_array = np.array(snr_values)
    # latent_mse_db_avg = np.array(latent_mse_db_avg)
    # latent_mse_db_std = np.array(latent_mse_db_std)
    #
    # # Prepare the result in the desired format
    # results = [{
    #     'model_name': 'HyperspectralUnmixing',
    #     'snr': snr_array,
    #     'latent_mse_db_avg': latent_mse_db_avg,
    #     'latent_mse_db_std': latent_mse_db_std
    # }]
    #
    # # Print the results
    # print(results)

# fixme: ask Xiao if MVES is correct depending on M and L
# todo: unmix when None it just matches components