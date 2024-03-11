import torch
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

    def forward(self, x):
        mean, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterization(mean, log_var)
        return self.decoder(z), mean, log_var

    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mean, log_var = self(x)
        loss = self.loss_function(x.view(-1, 784), x_hat, mean, log_var)
        self.log('train_loss', loss)
        return loss

    def loss_function(self, x, recon_x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return BCE + KLD

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)