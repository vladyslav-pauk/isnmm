import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import numpy as np

from src.modules.network.vision import LGCAN


# Custom Dataset for Tensor Files
class TensorDataset(Dataset):
    def __init__(self, data_tensor_path, label_tensor_path):
        self.data = torch.load(data_tensor_path).float()
        self.labels = torch.load(label_tensor_path).long()
        self.labels = self.labels[:len(self.data), ...]
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


def test(model, dataloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max logit
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 10
    EPOCHS = 200

    NUM_CLASSES = 4  # Number of classes
    NUM_SAMPLES = 1110  # Number of samples to use
    TEST_SPLIT = 0.2  # Fraction of data used for testing

    # Paths to the tensor files
    data_tensor_path = "../../datasets/mri/DWland_DCE/dwi_tensordata/dwi_latent_tensordata.pth"
    label_tensor_path = "../../datasets/mri/DWland_DCE/labels_tensordata/labels_tensordata.pth"

    # Load the dataset from tensor files
    full_dataset = TensorDataset(data_tensor_path, label_tensor_path)

    IN_CHANNELS = full_dataset.data.shape[1]  # Number of input channels

    # Split into train and test subsets
    indices = np.random.permutation(len(full_dataset))
    split_idx = int(len(indices) * (1 - TEST_SPLIT))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, Optimizer
    model = LGCAN(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    print(f"Training the LG-CAN model on a subset of {len(train_indices)} samples...")
    train(model, train_loader, optimizer, epochs=EPOCHS)

    # Test the model
    print("Evaluating the LG-CAN model on the test set...")
    test(model, test_loader)