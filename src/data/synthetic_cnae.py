import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib

# todo: separate generation into model
# todo: data modules only load data from files, the same class for cnae, tabular, and lmm

class SyntheticDataset(Dataset):
    def __init__(self, linear_mixture, nonlinear_mixture, latent, q):
        self.linear_mixture = linear_mixture
        self.nonlinear_mixture = nonlinear_mixture
        self.latent = latent
        self.q = q

    def __len__(self):
        return self.linear_mixture.shape[0]

    def __getitem__(self, idx):
        return {
            'linear_mixture': self.linear_mixture[idx],
            'nonlinear_mixture': self.nonlinear_mixture[idx],
            'latent': self.latent[idx],
            'q': self.q[idx]
        }
        # return self.nonlinear_mixture[idx], (self.latent[idx], self.linear_mixture[idx], self.q[idx])


class DataModule(pl.LightningDataModule):
    def __init__(self, config_data_model=None, observed_dim=None, latent_dim=None, size=None, batch_size=64, split=[0.8, 0.1, 0.1], mixing_scale_factors=None, seed=8):
        super(DataModule, self).__init__()

        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.num_samples = size
        self.batch_size = batch_size
        self.val_split = split[1]
        self.test_split = split[2]
        self.seed = seed

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # Initialize random mixing matrix (A) for linear mixtures
        self.mixing_matrix = torch.randn(self.observed_dim, self.latent_dim)

        # Simplex data generation (latent variables on the simplex)
        latent_data = torch.rand(self.num_samples, self.latent_dim)
        self.s = latent_data / torch.sum(latent_data, dim=1, keepdim=True)

        # QR decomposition for later evaluation
        self.q, _ = torch.linalg.qr(self.s)

        # Optional scaling factors for each mixture component
        if mixing_scale_factors is None:
            mixing_scale_factors = [5.0, 4.0, 1.0]
        self.mixing_scale_factors = mixing_scale_factors

        # Create mixtures (linear and nonlinear) based on latent variables
        self.linear_mixture = self.create_linear_mixture()
        self.nonlinear_mixture = self.create_nonlinear_mixture()

        # Initialize dataset
        self.dataset = SyntheticDataset(
            self.linear_mixture, self.nonlinear_mixture, self.s, self.q
        )

    def create_linear_mixture(self):
        # Linear mixture scaled by given factors
        return (self.s @ self.mixing_matrix.T) * self.mixing_scale_factors

    def create_nonlinear_mixture(self):
        # Create the nonlinear mixture from the linear mixture
        nonlinear_mixture = torch.zeros_like(self.linear_mixture)

        # Nonlinear function for the 1st dimension
        nonlinear_mixture[:, 0] = 5 * torch.sigmoid(self.linear_mixture[:, 0]) + 0.3 * self.linear_mixture[:, 0]

        # Nonlinear function for the 2nd dimension
        nonlinear_mixture[:, 1] = -3 * torch.tanh(self.linear_mixture[:, 1]) - 0.2 * self.linear_mixture[:, 1]

        # Nonlinear function for the 3rd dimension
        nonlinear_mixture[:, 2] = 0.4 * torch.exp(self.linear_mixture[:, 2])

        # nonlinear_mixture[:, 3] = 0.5 * torch.sin(self.linear_mixture[:, 3]) + 0.2 * self.linear_mixture[:, 3]
        #
        # nonlinear_mixture[:, 4] = 0.5 * torch.cos(self.linear_mixture[:, 4]) + 0.2 * self.linear_mixture[:, 4]
        #
        # nonlinear_mixture[:, 5] = 0.5 * torch.tanh(self.linear_mixture[:, 5]) + 0.2 * self.linear_mixture[:, 5]

        return nonlinear_mixture

    def forward(self, x):
        # Define forward pass if required (not necessary for data generation)
        pass

    def setup(self, stage=None):
        # Split dataset into training, validation, and test sets
        test_size = self.test_split
        val_size = self.val_split
        train_size = 1 - val_size - test_size

        self.data_train, self.data_val, self.data_test = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

    def plot_mixtures(self):
        # Visualization of mixtures to compare linear and nonlinear transformations
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.scatter(self.linear_mixture[:, 0].cpu().numpy(), self.nonlinear_mixture[:, 0].cpu().numpy())
        plt.title('1st Dimension')

        plt.subplot(1, 3, 2)
        plt.scatter(self.linear_mixture[:, 1].cpu().numpy(), self.nonlinear_mixture[:, 1].cpu().numpy())
        plt.title('2nd Dimension')

        plt.subplot(1, 3, 3)
        plt.scatter(self.linear_mixture[:, 2].cpu().numpy(), self.nonlinear_mixture[:, 2].cpu().numpy())
        plt.title('3rd Dimension')

        plt.show()

    def save_data(self, filename='post-nonlinear_simplex_synthetic_data.mat'):
        # Save the generated data into a .mat file
        sio.savemat(filename, {
            'observed_sample': self.nonlinear_mixture.cpu().numpy(),
            'noiseless_sample': self.nonlinear_mixture.cpu().numpy(),
            'latent_sample': self.s.cpu().numpy(),
            'latent_sample_qr': self.q.cpu().numpy(),
            'linearly_mixed_sample': self.linear_mixture.cpu().numpy(),
            'linear_mixture': self.mixing_matrix.cpu().numpy(),
            'sigma': 1.0
        })


# Example of instantiating and using the class
if __name__ == "__main__":
    # Define model parameters
    observed_dim = 3
    latent_dim = 3
    num_samples = 5000
    mixing_scale_factors = torch.tensor([5.0, 4.0, 1.0])#, 2.0, 1.5, 2.5])

    # Instantiate the model and move it to the appropriate device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DataModule(observed_dim=observed_dim, latent_dim=latent_dim, size=num_samples, mixing_scale_factors=mixing_scale_factors)

    #  set seeds
    torch.manual_seed(1)

    # Plot the mixtures
    model.plot_mixtures()

    # Save the generated data
    model.save_data()