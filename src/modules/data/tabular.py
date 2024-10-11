import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from torchvision import transforms
import os
import sys


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, split, num_workers, observed_dim=None, latent_dim=None):
        super().__init__()

        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))

        if PROJECT_ROOT not in sys.path:
            sys.path.append(PROJECT_ROOT)

        self.csv_file = f'{PROJECT_ROOT}/datasets/{dataset}/ETF_prices.csv'
        self.batch_size = batch_size
        self.split_sizes = split
        self.num_workers = num_workers
        self.observed_dim = observed_dim
        self.latent_dim = latent_dim
        self.data_model = DataModel()

        self.dataset = None
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        # Load the data from CSV
        data = pd.read_csv(self.csv_file)

        # Preprocess the data
        data = data[['price_date', 'adj_close', 'fund_symbol']]
        data = data.sort_values(by=['fund_symbol', 'price_date'])
        data['return'] = data.groupby('fund_symbol')['adj_close'].pct_change()

        data_pivot = data.pivot(index='price_date', columns='fund_symbol', values='return')

        # Clean the data by removing columns and rows with NaN values
        data_pivot_cleaned = data_pivot.dropna(axis=1, how='all')
        data_pivot_cleaned = data_pivot_cleaned.dropna(axis=0, how='all')

        threshold = int(0.01 * data_pivot_cleaned.shape[1])
        data_pivot_cleaned = data_pivot_cleaned.dropna(thresh=threshold, axis=0)

        threshold_columns = int(1 * data_pivot_cleaned.shape[0])
        returns_matrix = data_pivot_cleaned.dropna(thresh=threshold_columns, axis=1)

        # Calculate the variance of each column (fund) and sort them
        feature_variances = returns_matrix.var().sort_values(ascending=False)

        # Select the top `self.observed_dim` most variated features
        if self.observed_dim is not None and self.observed_dim <= len(feature_variances):
            top_features = feature_variances.index[:self.observed_dim]
            returns_matrix = returns_matrix[top_features]

        # Calculate mean and std for normalization
        feature_mean = returns_matrix.mean().values
        feature_std = returns_matrix.std().values

        # Create the dataset with the normalization transform
        self.dataset = CSVTimeSeriesDataset(returns_matrix, mean=feature_mean, std=feature_std)

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            total_size = len(self.dataset)
            train_size = int(total_size * self.split_sizes[0])
            val_size = int(total_size * self.split_sizes[1])
            test_size = total_size - train_size - val_size

            self.data_train, self.data_val, self.data_test = random_split(
                self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)


class CSVTimeSeriesDataset(Dataset):
    def __init__(self, data, mean, std):
        """
        Custom Dataset for tabular time series data loaded from CSV.
        :param data: Processed pandas DataFrame with returns.
        :param mean: Mean of the features (for normalization).
        :param std: Std of the features (for normalization).
        """
        self.data = torch.tensor(data.values, dtype=torch.float32)  # Convert the DataFrame to a tensor
        self.transform = NormalizeTabular(mean, std)  # Custom normalization transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Apply the normalization transform
        x_normalized = self.transform(x)
        return x_normalized, torch.nan_like(x_normalized)


class NormalizeTabular:
    def __init__(self, mean, std):
        """
        Custom normalization transform for tabular data.
        :param mean: Feature-wise mean.
        :param std: Feature-wise standard deviation.
        """
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, x):
        # Normalize x by subtracting the mean and dividing by the std
        return (x - self.mean) / self.std


class DataModel():
    def __init__(self):
        self.sigma = 25


# Usage example:
# csv_file = 'Yahoo'  # dataset name
# data_module = DataModule(dataset=csv_file, batch_size=32, split=[0.8, 0.1, 0.1], num_workers=4)
# data_module.prepare_data()
# data_module.setup('fit')
# train_loader = data_module.train_dataloader()