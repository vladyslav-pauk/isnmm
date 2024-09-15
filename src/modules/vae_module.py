import wandb

from torch.optim import Adam
import pytorch_lightning as pl


class VAEModule(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=None, monitor=None, mc_samples=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.lr_th = lr["th"]
        self.lr_ph = lr["ph"]
        self.mc_samples = mc_samples
        self.monitor = monitor

        # todo: all hyperparameters that affect outcome of training are passed in **config["train"].
        #  Ideally, I save these as hyperparameters, and the rest as config.
        #  self.save_hyperparameters(ignore=['encoder', 'decoder', 'ground_truth_model'])

    def forward(self, x):

        variational_parameters = self.encoder(x)
        z_mc_sample = self.reparameterize(variational_parameters, self.mc_samples)
        x_mc_sample = self.decoder(z_mc_sample)

        return x_mc_sample, z_mc_sample, variational_parameters

    def on_train_start(self) -> None:

        wandb.define_metric(name=self.monitor["monitor"], summary=self.monitor["mode"])
        # todo: move it out so i don't drag monitor config through classes

    def training_step(self, batch, batch_idx):

        data, _ = batch
        loss = self.loss_function(data, self(data))

        self.log_dict(loss)
        wandb.log(loss)
        # todo: should work with only self.log_dict (same validation)

        return sum(loss.values())

    def validation_step(self, batch, batch_idx):

        data, labels = batch

        validation_loss = {"validation_loss": sum(self.loss_function(data, self(data)).values())}
        metrics = self.compute_metrics(data, self(data), labels)

        wandb.log(validation_loss)
        wandb.log(metrics)
        self.log_dict(validation_loss)
        self.log_dict(self.metrics)

        return validation_loss, metrics

    def test_step(self, batch, batch_idx):

        data, labels = batch

        metrics = self.compute_metrics(data, self(data), labels, print_results=True)
        print(metrics)

        # todo: A (independent on data) is different from the best value in the validation wandb
        return metrics

    def configure_optimizers(self):

        optimizer = Adam([
            {'params': self.encoder.parameters(), 'lr': self.lr_ph},
            {'params': self.decoder.parameters(), 'lr': self.lr_th},
        ])

        return optimizer


# loss = self.loss_function(x.view(-1, x[0].numel()), x_hat, z_hat, encoder_params, sigma)
# posterior_params = self.encoder(x.view(-1, x[0].numel()))
# todo: move x = x.view(-1, x[0].numel()) to MNIST transform
