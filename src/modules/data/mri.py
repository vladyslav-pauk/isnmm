import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from collections import defaultdict
import matplotlib.pyplot as plt
import pydicom
from src.modules.transform import HyperspectralTransform  # Assuming HyperspectralTransform is available
import numpy as np


class DataModule(LightningDataModule):
    def __init__(self, data_config, transform=None, **config):
        """
        Unified Data Module for handling both DICOM and tensor datasets.

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
        self.data_model = data_config.get("data_model")
        self.tensors = {}
        self.metadata = defaultdict(list)  # Store metadata for all files or components
        self.file_paths = []

        # Initialize transform
        self.transform = transform or HyperspectralTransform(
            output_channels=data_config.get("observed_dim", None),
            normalize=data_config.get("normalize", True),
            dataset_size=data_config.get("dataset_size", None),
        )

    def prepare_data(self):
        """
        Prepare the data by loading DICOM files or tensor files based on `data_model`.
        """
        if self.data_model == 'Newcastle_Renal':
            base_dir = 'datasets/mri/Newcastle_Renal_DCE-anonymised/Ncl_Renal_Dce_Patient_A/Unnamed-505903245/unnamed_1401/'
            base_dir = os.path.join(os.getcwd().split('src')[0], base_dir)
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(".dcm"):
                        full_path = os.path.join(root, file)
                        self.file_paths.append(full_path)

            if not self.file_paths:
                raise FileNotFoundError(f"No DICOM files found in {base_dir}.")
            print(f"Found {len(self.file_paths)} DICOM files.")

        elif self.data_model == 'DWland':
            data_path = "/Users/home/Work/Projects/nisca/datasets/mri/DWland_DCE/dwi_tensordata/dwi_tensordata.pth"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            self.raw_data = torch.load(data_path)
            if not isinstance(self.raw_data, torch.Tensor):
                raise ValueError("Expected the .pth file to contain a PyTorch tensor.")
            # print(f"Loaded tensor data with shape {self.raw_data.shape}.")

    def extract_metadata(self):
        """
        Extract metadata for DICOM or tensor datasets.
        """
        if self.data_model == 'Newcastle_Renal':
            for file_path in self.file_paths:
                dicom_data = pydicom.dcmread(file_path)
                orientation = tuple(map(str, dicom_data.get("ImageOrientationPatient", [None])))
                self.metadata[orientation].append(file_path)
            # print(f"Extracted metadata for {len(self.metadata)} orientations.")

        elif self.data_model == 'DWland':
            num_samples, *shape = self.raw_data.shape
            self.metadata["num_samples"] = num_samples
            self.metadata["shape"] = shape
            self.metadata["num_channels"] = shape[0] if len(shape) > 1 else 1
            # print(f"Extracted metadata: {self.metadata}")

    def import_data(self, key):
        """
        Import data for DICOM orientation or tensor component.

        Args:
            key: Orientation tuple (DICOM) or component index (tensor).
        Returns:
            observed_data (torch.Tensor): Processed tensor data.
        """
        if self.data_model == 'Newcastle_Renal':
            filtered_files = self.metadata[key]
            slices = []
            for file_path in filtered_files:
                dicom_data = pydicom.dcmread(file_path)
                image_array = dicom_data.pixel_array.astype(np.float32)
                slice_position = getattr(dicom_data, "InstanceNumber", None)
                slices.append((slice_position, image_array))
            slices = sorted(slices, key=lambda x: x[0])
            observed_data = [(s[1] - np.min(s[1])) / (np.max(s[1]) - np.min(s[1])) for s in slices]
            observed_data = torch.tensor(np.stack(observed_data), dtype=torch.float32)
            return observed_data

        elif self.data_model == 'DWland':
            data = self.raw_data[key]
            data = (data - data.min()) / (data.max() - data.min())
            data = data.clone().detach().to(dtype=torch.float32)
            return data

    def setup(self, stage=None):
        """
        Setup datasets for DICOM orientations or tensor components.
        """
        self.extract_metadata()

        if self.data_model == 'Newcastle_Renal':
            self.datasets = {}
            for orientation in self.metadata.keys():
                observed_data = self.import_data(orientation)
                transformed_data = self.transform(observed_data)
                self.datasets[orientation] = {
                    "transformed_data": transformed_data,
                    "original_slices": observed_data
                }
            self.current_key = list(self.datasets.keys())[0]

        elif self.data_model == 'DWland':
            self.datasets = {}
            idx = 0
            observed_data = self.import_data(idx)
            transformed_data = self.transform(observed_data)
            self.datasets[f"component_{idx}"] = {
                "transformed_data": transformed_data,
                "original_data": observed_data
            }
            self.current_key = list(self.datasets.keys())[0]

        self.dataset = self.datasets[self.current_key]["transformed_data"]

    def train_dataloader(self):
        return DataLoader(
            UnifiedDataset(data=self.datasets[self.current_key]["transformed_data"]),
            batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            UnifiedDataset(data=self.datasets[self.current_key]["transformed_data"]),
            batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            UnifiedDataset(data=self.datasets[self.current_key]["transformed_data"]),
            batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers
        )

    def plot(self, key, num_images=5, save_plot=False, plot_dir="./plots"):
        """
        Plot slices for DICOM orientation or tensor component.
        """
        if key not in self.datasets:
            raise ValueError(f"Key {key} not found in processed datasets.")
        original_data = self.datasets[key]["original_slices" if self.data_model == 'Newcastle_Renal' else "original_data"]

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
            plot_path = os.path.join(plot_dir, f"slices_{key}.png")
            plt.savefig(plot_path)
            print(f"Saved plot to {plot_path}")
        plt.show()


class UnifiedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return {
            "data": self.data[idx],
            "idxes": idx
        }


if __name__ == "__main__":
    data_config = {
        'data_model': 'DWland',  # or 'Newcastle_Renal'
        'data_dir': '/path/to/data',  # Update path for DICOM or tensor
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

    data_module = DataModule(data_config, transform=None, **config)
    data_module.prepare_data()
    data_module.setup()
    for key in data_module.datasets.keys():
        print(f"Plotting slices for key: {key}")
        data_module.plot(key=key, num_images=5, save_plot=True)

# fixme: refactor to one or split into single patient, multi-patient
