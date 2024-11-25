import numpy as np

import torch
import torch.nn as nn


class HyperspectralTransform(nn.Module):
    def __init__(self, output_channels=None, normalize=True, dataset_size=None):
        super().__init__()

        self.output_channels = output_channels
        self.normalize = normalize
        self.dataset_size = dataset_size

        self.min_val = None
        self.max_val = None

    def forward(self, x):
        input_channels, height, width = x.shape
        self.input_channels = input_channels
        self.height = height
        self.width = width

        if self.dataset_size:
            x = self.crop_image(x)

        if self.output_channels:
            x = self.select_bands_with_highest_variance(x)

        if self.normalize:
            x = self.normalize_to_range(x)

        x = self.flatten(x)

        return x

    def inverse(self, x):
        x = self.unflatten(x)

        if self.normalize:
            x = self.inverse_normalize(x)

        return x

    def crop_image(self, x):
        crop_pixels = int(self.dataset_size)
        crop_height = int(np.sqrt(crop_pixels * self.height / self.width))
        crop_width = crop_pixels // crop_height
        top = (self.height - crop_height) // 2
        left = (self.width - crop_width) // 2
        x = x[:, top:top + crop_height, left:left + crop_width]
        self.height = crop_height
        self.width = crop_width
        return x

    def select_bands_with_highest_variance(self, x):
        band_variances = torch.var(x, dim=(1, 2))
        k = self.output_channels if self.output_channels else x.shape[0]  # Default to input channels if not set
        _, selected_indices = torch.topk(band_variances, k, largest=True)
        return x[selected_indices]

    def normalize_to_range(self, tensor, min_val=0, max_val=1):
        self.min_val = torch.min(tensor)
        self.max_val = torch.max(tensor)
        return (tensor - self.min_val) / (self.max_val - self.min_val) * (max_val - min_val) + min_val

    def inverse_normalize(self, tensor):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalization parameters not set. Ensure forward normalization is applied first.")
        return tensor * (self.max_val - self.min_val) + self.min_val

    def flatten(self, data):
        data = torch.flatten(data, start_dim=1).T

        return data

    def unflatten(self, data):
        data = data.T.view(-1, self.height, self.width)
        return data

    def calculate_transformed_dimensions(self, height, width):
        crop_pixels = int(self.dataset_size)
        crop_height = int(np.sqrt(crop_pixels * height / width))
        crop_width = crop_pixels // crop_height
        return crop_height, crop_width
