from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config['dataset']
        self.batch_size = config['train']['batch_size']

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.setup()

    def setup(self, stage=None):
        match self.dataset:
            case 'MNIST':
                from torchvision.datasets import MNIST
                transform = transforms.Compose([transforms.ToTensor()])
                self.data_train = MNIST('datasets', train=True, download=True, transform=transform)
                self.data_val = MNIST('datasets', train=False, download=True, transform=transform)

            case _:
                from src.modules.data_model import SyntheticDataset
                self.dataset = SyntheticDataset(self.config)
                self.data_train, self.data_val, self.data_test = random_split(self.dataset, [0.8, 0.1, 0.1])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=0, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=0, persistent_workers=False)

    def test_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=0, persistent_workers=False)
