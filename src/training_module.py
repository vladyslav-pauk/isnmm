import wandb
import torch
from torch.optim import Adam
import pytorch_lightning as pl


class VAE(pl.LightningModule):
    def __init__(self, encoder, decoder, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.save_hyperparameters(ignore=['encoder', 'decoder', 'data_model'])
        self.lr_th = lr["th"]
        self.lr_ph = lr["ph"]

    def forward(self, x):
        posterior_params = self.encoder(x.view(-1, x[0].numel()))
        z_samples = self.reparameterize(posterior_params)
        x_recon_samples, likelihood_params = self.decoder(z_samples)

        return x_recon_samples, z_samples, posterior_params, likelihood_params

    def training_step(self, batch, batch_idx):
        x, z = batch
        x_recon_samples, z_samples, posterior_params, likelihood_params = self(x)

        loss = self.loss_function(x, x_recon_samples, z_samples, posterior_params, likelihood_params)
        self.log_dict(loss)

        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        # todo: make validation every epoch and x-axis epoch instead of global step
        if batch_idx == 0:
            x, z = batch
            x_recon_samples, z_samples, posterior_params, likelihood_params = self(x)

            loss = self.loss_function(x, x_recon_samples, z_samples, posterior_params, likelihood_params)
            metric = self.metric(posterior_params, likelihood_params, x, z, x_recon_samples)

            self.log_dict({"validation_loss": sum(loss.values())})
            self.log_dict({k: v for k, v in metric.items()})
        else:
            pass

    def configure_optimizers(self):
        optimizer = Adam([
            {'params': self.encoder.parameters(), 'lr': self.lr_ph},
            {'params': self.decoder.parameters(), 'lr': self.lr_th},
        ])
        return optimizer

        # loss = self.loss_function(x.view(-1, x[0].numel()), x_hat, z_hat, encoder_params, sigma)
        # todo: move x = x.view(-1, x[0].numel()) to MNIST transform
