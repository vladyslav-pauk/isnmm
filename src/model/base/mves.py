import torch
from pytorch_lightning import LightningModule
from scipy.optimize import nnls, linear_sum_assignment
from scipy.spatial import ConvexHull
import numpy as np


class Model(LightningModule):
    def __init__(self, encoder, decoder, model_config, optimizer_config):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.latent_dim = model_config["latent_dim"]

        self.A_est = None
        self.s_est = None

    def setup(self, stage):
        datamodule = self.trainer.datamodule
        data_sample = next(iter(datamodule.train_dataloader()))
        self.dataset_size = data_sample["data"].shape[0]

        self.observed_dim = data_sample["data"].shape[1]
        if self.latent_dim is None and data_sample["labels"]:
            self.latent_dim = data_sample["labels"]["latent_sample"].shape[-1]
            print("Labelled data found.")

        self.N = self.latent_dim
        self.M = self.observed_dim
        self.L = self.dataset_size

    def forward(self, X):
        # MVES does not have a typical forward pass
        pass

    def training_step(self, batch, batch_idx):
        # We need to process the entire dataset at once
        # So we'll collect all batches in a buffer and process them after the last batch
        # However, since PyTorch Lightning handles batches, we can process data in the 'on_fit_start' hook

        return None  # Since we process data elsewhere

    def on_fit_start(self):
        # This method is called before training starts
        # We can access the entire training dataset here
        dataloader = self.trainer.datamodule.train_dataloader()
        all_data = []
        for batch in dataloader:
            X = batch['data']  # Shape: (batch_size, M)
            all_data.append(X)
        X = torch.cat(all_data, dim=0)  # Shape: (L, M)
        X = X.T  # Shape: (M, L)

        # Implement MVES algorithm here using X
        # Center the data
        d = X.mean(dim=1, keepdim=True)
        U = X - d  # Shape: (M, L)

        # Compute covariance matrix
        covariance_matrix = U @ U.T / (self.L - 1)  # Shape: (M, M)

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        C = eigenvectors[:, sorted_indices[:self.N - 1]]  # Shape: (M, N-1)

        # Compute pseudoinverse of C
        C_pinv = torch.pinverse(C)  # Shape: (N-1, M)

        # Project the data
        X_tilde = C_pinv @ (X - d)  # Shape: (N-1, L)

        # Convert to NumPy for ConvexHull
        X_tilde_np = X_tilde.cpu().detach().numpy().T  # Shape: (L, N-1)

        # Compute convex hull
        hull = ConvexHull(X_tilde_np)
        vertices = X_tilde_np[hull.vertices]

        # Select N vertices
        if vertices.shape[0] > self.N:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.N, random_state=0).fit(vertices)
            alpha_est = torch.from_numpy(kmeans.cluster_centers_.T)
        else:
            alpha_est = torch.from_numpy(vertices[:self.N].T)

        # Move to the same device
        alpha_est = alpha_est.to(self.device)

        # Construct B matrix
        B = alpha_est[:, :-1] - alpha_est[:, [-1]]  # Shape: (N-1, N-1)
        B_inv = torch.inverse(B)
        H = B_inv
        g = B_inv @ alpha_est[:, -1]

        # Estimate endmembers in original space
        A_est = C @ alpha_est + d  # Shape: (M, N)

        # Estimate abundances
        s_est = torch.zeros(self.N, X.shape[1], device=self.device)
        A_est_np = A_est.cpu().detach().numpy()
        X_np = X.cpu().detach().numpy()
        for n in range(X.shape[1]):
            x_n = X_np[:, n]
            s_n, _ = nnls(A_est_np, x_n)
            s_est[:, n] = torch.from_numpy(s_n)

        # Normalize abundances
        sums = s_est.sum(dim=0)
        s_est = s_est / sums

        # Store estimated endmembers and abundances
        self.A_est = A_est  # Estimated endmembers
        self.s_est = s_est  # Estimated abundances

        # Compute loss (e.g., reconstruction error)
        X_reconstructed = A_est @ s_est  # Shape: (M, L)
        loss = torch.nn.functional.mse_loss(X_reconstructed, X)

        # self.log('train_loss', loss)
        # Since we're not updating parameters, we don't need to call optimizer

    def configure_optimizers(self):
        # No optimizer needed for MVES
        return None

    def validation_step(self, batch, batch_idx):
        # Since we processed all data already, validation can be skipped or used for evaluation
        pass

    def test_step(self, batch, batch_idx):
        # Evaluate the model on test data if needed
        pass

    def on_fit_end(self):
        # Optionally, you can perform evaluation here using true abundances and endmembers if available
        pass


class Encoder():
    def __init__(self, config):
        self.config = config

    def construct(self, latent_dim, observed_dim):
        pass

    def forward(self, x):
        pass


class Decoder():
    def __init__(self, config):
        self.config = config

    def construct(self, latent_dim, observed_dim):
        pass

    def forward(self, x):
        pass