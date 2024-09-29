import wandb
import pytorch_lightning as pl


class AutoEncoderModule(pl.LightningModule):
    # todo: AutoEncoder
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            train_loader = self.trainer.datamodule.train_dataloader()
            sample_batch = next(iter(train_loader))
            data_sample = sample_batch
            observed_dim = data_sample[0].shape[1]
            if self.latent_dim is None:
                self.latent_dim = data_sample[1][0].shape[1]

            self.encoder.construct(self.latent_dim, observed_dim)
            self.decoder.construct(self.latent_dim, observed_dim)

    def forward(self, observed_batch):
        latent_parameterization_batch = self.encoder(observed_batch)
        latent_sample = self.reparameterize(latent_parameterization_batch)
        observed_sample = self.decoder(latent_sample)
        return observed_sample, latent_sample, latent_parameterization_batch

    def reparameterize(self, latent_parameterization_batch):
        raise latent_parameterization_batch

    def on_train_start(self) -> None:
        wandb.define_metric(name=self.log_monitor["monitor"], summary=self.log_monitor["mode"])
        # todo: move it out so i don't drag monitor config through classes

    def training_step(self, batch, batch_idx):
        data, labels = batch
        loss = self.loss_function(data, self(data))
        self.log_dict(loss)
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        validation_loss = {"validation_loss": sum(self.loss_function(data, self(data)).values())}
        self.update_metrics(data, self(data), labels)
        self.log_dict({**validation_loss, **self.metrics.compute()})


# loss = self.loss_function(x.view(-1, x[0].numel()), x_hat, z_hat, encoder_params, sigma)
# posterior_params = self.encoder(x.view(-1, x[0].numel()))
# todo: move x = x.view(-1, x[0].numel()) to MNIST transform
# todo: log or monitor averaged loss? is it accumulating?
# todo: all hyperparameters that affect outcome of training are passed in **config["train"].
#  Ideally, I save these as hyperparameters, and the rest as config.
#  self.save_hyperparameters(ignore=['encoder', 'decoder', 'ground_truth_model'])