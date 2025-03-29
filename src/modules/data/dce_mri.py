import os
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from collections import defaultdict
import matplotlib.pyplot as plt
from src.modules.transform import HyperspectralTransform  # Assuming HyperspectralTransform is available


class PthDataModule(LightningDataModule):
    def __init__(self, data_config, transform=None, **config):
        """
        Pth Data Module for loading and preprocessing tensor data from .pth files.

        Args:
            data_config (dict): Configuration for the data, including file paths and parameters.
            transform (callable, optional): Transformation to apply to the data.
            config (dict): Additional configurations like batch size and workers.
        """
        super().__init__()
        self.data_config = data_config
        self.batch_size = config.get("batch_size", 16)
        self.val_batch_size = config.get("val_batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.shuffle = config.get("shuffle", True)
        self.dataset = None
        self.tensors = {}
        self.metadata = defaultdict(list)  # Store metadata for all data

        # Initialize transform
        self.transform = transform or HyperspectralTransform(
            output_channels=data_config.get("observed_dim", None),
            normalize=data_config.get("normalize", True),
            dataset_size=data_config.get("dataset_size", None),
        )

    def prepare_data(self):
        """
        Load the tensor data from the .pth file and initialize metadata.
        """
        data_path = "/Users/home/Work/Projects/nisca/datasets/mri/DWland_DCE/dwi_tensordata/dwi_tensordata.pth"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        self.raw_data = torch.load(data_path)

        if not isinstance(self.raw_data, torch.Tensor):
            raise ValueError("Expected the .pth file to contain a PyTorch tensor.")

        print(f"Loaded data from {data_path} with shape {self.raw_data.shape}.")

    def extract_metadata(self):
        """
        Extract metadata for all tensor data, categorizing by components or channels.
        """
        num_samples, *shape = self.raw_data.shape
        self.metadata["num_samples"] = num_samples
        self.metadata["shape"] = shape
        self.metadata["num_channels"] = shape[0] if len(shape) > 1 else 1
        print(f"Extracted metadata: {self.metadata}")

    def import_data(self, component_idx):
        """
        Import a specific component of the tensor data and apply normalization.

        Args:
            component_idx (int): Index of the component (e.g., channel) to process.
        Returns:
            observed_data (torch.Tensor): Processed tensor data for the given component.
        """
        if component_idx >= self.metadata["num_channels"]:
            raise ValueError(f"Component index {component_idx} is out of range.")

        component_data = self.raw_data[:, component_idx]

        # Normalize component data
        component_data = (component_data - component_data.min()) / (component_data.max() - component_data.min())
        return component_data

    def setup(self, stage=None):
        """
        Setup the dataset and process each component or channel independently.
        """
        self.extract_metadata()

        self.datasets = {}
        for component_idx in range(self.metadata["num_channels"]):
            observed_data = self.import_data(component_idx)
            transformed_data = self.transform(observed_data) if self.transform else observed_data

            self.datasets[f"component_{component_idx}"] = {
                "transformed_data": transformed_data,
                "original_data": observed_data
            }

        # Default to the first component
        self.current_component = "component_0"
        self.dataset = self.datasets[self.current_component]["transformed_data"]

        print(f"Processed datasets for {len(self.datasets)} components.")

    def train_dataloader(self):
        return DataLoader(
            PthDataset(data=self.datasets[self.current_component]["transformed_data"]),
            batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            PthDataset(data=self.datasets[self.current_component]["transformed_data"]),
            batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            PthDataset(data=self.datasets[self.current_component]["transformed_data"]),
            batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            PthDataset(data=self.datasets[self.current_component]["transformed_data"]),
            batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def plot(self, component, num_images=5, save_plot=False, plot_dir="./plots"):
        """
        Plot slices from the specified component of the tensor data.

        Args:
            component (str): Component key to plot data for.
            num_images (int): Number of images to plot.
            save_plot (bool): Whether to save the plot.
            plot_dir (str): Directory to save the plot.
        """
        if component not in self.datasets:
            raise ValueError(f"Component {component} not found in processed datasets.")

        original_data = self.datasets[component]["original_data"]

        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        for i, ax in enumerate(axes):
            if i >= original_data.size(0):
                break
            ax.imshow(original_data[i].numpy(), cmap="gray")
            ax.axis("off")
            ax.set_title(f"Slice {i}")

        plt.tight_layout()

        if save_plot:
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"tensor_slices_{component}.png")
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")

        plt.show()


class PthDataset(Dataset):
    def __init__(self, data):
        """
        Dataset for handling transformed tensor data.

        Args:
            data (torch.Tensor): Processed tensor data.
        """
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "idxes": idx
        }


if __name__ == "__main__":
    # Configuration for the .pth dataset
    data_config = {
        'data_dir': "/Users/home/Work/Projects/nisca/datasets/mri/DWland_DCE/dwi_tensordata/dwi_tensordata.pth",
        'normalize': True,
        'dataset_size': None,
        'observed_dim': None,
    }

    config = {
        'batch_size': 32,
        'val_batch_size': 32,
        'num_workers': 4,
        'shuffle': True,
    }

    # Instantiate the .pth Data Module
    data_module = PthDataModule(data_config, transform=None, **config)

    # Prepare and set up the data
    data_module.prepare_data()
    data_module.setup()

    # Plot slices from the first component
    for component in data_module.datasets.keys():
        print(f"Plotting slices for component: {component}")
        data_module.plot(component=component, num_images=5, save_plot=True)