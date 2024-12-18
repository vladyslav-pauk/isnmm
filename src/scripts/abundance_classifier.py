# from torch.utils.data import Subset
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import torch.nn.functional as F
# import numpy as np
#
# from src.modules.network.vision import LGCAN
#
#
# def train(model, dataloader, optimizer, epochs=5):
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = F.cross_entropy(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
#
#
# if __name__ == "__main__":
#     # Device setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Hyperparameters
#     BATCH_SIZE = 8
#     EPOCHS = 5
#     IN_CHANNELS = 6  # 6-channel input
#     NUM_CLASSES = 10  # CIFAR-10
#     NUM_SAMPLES = 500  # Number of samples to use
#
#     # DataLoader - CIFAR10
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         lambda x: torch.cat([x, x], dim=0)  # Duplicate channels to create 6-channel input
#     ])
#     full_dataset = datasets.CIFAR10(root="../../datasets/torchvision/", train=True, transform=transform, download=True)
#
#     # Use only a subset of the dataset
#     indices = np.random.choice(len(full_dataset), NUM_SAMPLES, replace=False)
#     subset_dataset = Subset(full_dataset, indices)
#
#     dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
#     # Model, Optimizer
#     model = LGCAN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
#     # Train the model
#     print(f"Training the LG-CAN model on a subset of {NUM_SAMPLES} samples...")
#     train(model, dataloader, optimizer, epochs=EPOCHS)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import numpy as np

from src.modules.network.vision import LGCAN


# Custom Dataset for Tensor Files
class TensorDataset(Dataset):
    def __init__(self, data_tensor_path, label_tensor_path):
        self.data = torch.load(data_tensor_path)  # Load data tensor
        self.labels = torch.load(label_tensor_path)  # Load label tensor
        assert self.data.shape[0] == self.labels.shape[0], "Data and labels must have the same number of samples."

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train(model, dataloader, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 5
    IN_CHANNELS = 6  # 6-channel input
    NUM_CLASSES = 10  # Number of classes
    NUM_SAMPLES = 500  # Number of samples to use

    # Paths to the tensor files
    data_tensor_path = "../../datasets/mri/DWland_DCE/dwi_tensordata/dwi_tensordata.pth"
    label_tensor_path = "../../datasets/mri/DWland_DCE/dwi_tensordata/labels_tensordata/labels_tensordata.pth"

    # Load the dataset from tensor files
    full_dataset = TensorDataset(data_tensor_path, label_tensor_path)

    # Use only a subset of the dataset
    indices = np.random.choice(len(full_dataset), NUM_SAMPLES, replace=False)
    subset_dataset = Subset(full_dataset, indices)

    dataloader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Optimizer
    model = LGCAN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    print(f"Training the LG-CAN model on a subset of {NUM_SAMPLES} samples...")
    train(model, dataloader, optimizer, epochs=EPOCHS)