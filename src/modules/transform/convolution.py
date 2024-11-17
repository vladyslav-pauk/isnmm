import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
        if self.output_channels is None:
            self.output_channels = input_channels
        self.height = height
        self.width = width

        if self.normalize:
            x = self.normalize_to_range(x)

        if self.dataset_size:
            self.crop_fraction = (self.dataset_size / height / width) ** 0.5
            x = self.crop_image(x)

        if self.output_channels is not None:
            x = self.select_bands_with_highest_variance(x)

        x = x.reshape(self.output_channels, -1)

        return x

    def backward(self, x):
        flattened_size = self.height * self.width

        if x.numel() != self.output_channels * flattened_size:
            raise ValueError(
                f"Cannot reshape tensor. Expected {self.output_channels * flattened_size} elements, but got {x.numel()}.")

        x = x.view(self.output_channels, self.height, self.width)

        if self.normalize:
            x = self.inverse_normalize(x)

        return x

    def crop_image(self, x):
        crop_height = int(self.height * self.crop_fraction)
        crop_width = int(self.width * self.crop_fraction)
        top = (self.height - crop_height) // 2
        left = (self.width - crop_width) // 2
        x = x[:, top:top + crop_height, left:left + crop_width]
        self.height = crop_height
        self.width = crop_width
        return x

    def select_bands_with_highest_variance(self, x):
        if self.output_channels is not None:
            band_variances = torch.var(x, dim=(1, 2))
            _, selected_indices = torch.topk(band_variances, self.output_channels, largest=True)
            x = x[selected_indices]
        return x

    def normalize_to_range(self, tensor, min_val=0, max_val=1):
        self.min_val = torch.min(tensor)
        self.max_val = torch.max(tensor)
        return (tensor - self.min_val) / (self.max_val - self.min_val) * (max_val - min_val) + min_val

    def inverse_normalize(self, tensor):
        if self.min_val is None or self.max_val is None:
            raise ValueError("Normalization parameters not set. Ensure forward normalization is applied first.")
        return tensor * (self.max_val - self.min_val) + self.min_val

    def plot_image(self, x):
        plt.imshow(x.numpy(), cmap='gray')
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    data = 100 * torch.randn(300, 100, 100)

    transform = HyperspectralTransform(output_channels=2, normalize=True, dataset_size=100)

    print(f"Input: {data.shape}")
    transformed_data = transform(data)
    print(f"Transformed: {transformed_data.shape}")

    reconstructed_data = transform.backward(transformed_data)
    print(f"Reconstructed: {reconstructed_data.shape}")

    transform.plot_image(data[0])
    transform.plot_image(reconstructed_data[0])