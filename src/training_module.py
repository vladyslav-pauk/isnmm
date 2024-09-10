import wandb
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
        variational_parameters = self.encoder(x)
        z_mc_sample = self.reparameterize(variational_parameters)
        x_mc_sample = self.decoder(z_mc_sample)

        return (x_mc_sample, z_mc_sample), variational_parameters

    def training_step(self, batch, batch_idx):
        data, _ = batch if len(batch) == 2 else (batch, None)
        x_mc_sample, z_mc_sample, variational_parameters = self(data)

        loss = self.loss_function(data, x_mc_sample, z_mc_sample, variational_parameters)
        self.log_dict(loss)
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            x, z = batch if len(batch) == 2 else (batch, None)
            x_mc_sample, z_mc_sample, variational_parameters = self(x)

            loss = self.loss_function(x, x_mc_sample, z_mc_sample, variational_parameters)
            metric = self.metric(x, z, x_mc_sample, z_mc_sample, variational_parameters)

            wandb.log({"validation_loss": sum(loss.values())})
            wandb.log({k: v for k, v in metric.items()})
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
# posterior_params = self.encoder(x.view(-1, x[0].numel()))
# todo: make validation every epoch and x-axis epoch instead of global step
# todo: include A into likelihood_params, as well as nonlinearity
