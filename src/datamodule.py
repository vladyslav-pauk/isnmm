from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path='datasets', batch_size=128):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size

        self.mnist_train = None
        self.mnist_val = None

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_train = MNIST(self.dataset_path, train=True, download=True, transform=transform)
        self.mnist_val = MNIST(self.dataset_path, train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=11, persistent_workers=True)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=11, persistent_workers=True)
