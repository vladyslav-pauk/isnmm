import wandb
import pytorch_lightning as pl


class AutoEncoderModule(pl.LightningModule):
    # todo: AutoEncoder
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = None
        self.observed_dim = None

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            train_loader = self.trainer.datamodule.train_dataloader()
            sample_batch = next(iter(train_loader))
            data_sample = sample_batch
            self.observed_dim = data_sample["data"].shape[1]
            if self.latent_dim is None:
                self.latent_dim = data_sample["labels"]["latent_sample"].shape[1]

            self.encoder.construct(self.latent_dim, self.observed_dim)
            self.decoder.construct(self.latent_dim, self.observed_dim)

    def forward(self, observed_batch):
        latent_parameterization_batch = self.encoder(observed_batch)
        latent_sample = self.reparameterize(latent_parameterization_batch)
        reconstructed_sample = self.decoder(latent_sample)

        model_output = {
            "reconstructed_sample": reconstructed_sample,
            "latent_sample": latent_sample,
            "latent_parameterization_batch": latent_parameterization_batch
        }
        return model_output

    def reparameterize(self, latent_parameterization_batch):
        return latent_parameterization_batch[0]

    def on_train_start(self) -> None:
        wandb.define_metric(name=self.log_monitor["monitor"], summary=self.log_monitor["mode"])
        # todo: move it out so i don't drag monitor config through classes

    def training_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        loss = self.loss_function(data, self(data), idxes)
        self.log_dict(loss)
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        validation_loss = {"validation_loss": sum(self.loss_function(data, self(data), idxes).values())}
        self.update_metrics(data, self(data), labels, idxes)
        self.log_dict({**validation_loss, **self.metrics.compute()})

    def test_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        model_outputs = self(data)
        self.update_metrics(data, model_outputs, labels, idxes)
        # self.metrics['evaluate_metric'].toggle_show_plot(True)
        print(self.metrics.compute())


# todo: refactor data_model so it has a forward method so i can run inference like on model
# todo: check if (independent on data) is the same as the best value in the validation wandb
# loss = self.loss_function(x.view(-1, x[0].numel()), x_hat, z_hat, encoder_params, sigma)
# posterior_params = self.encoder(x.view(-1, x[0].numel()))
# todo: move x = x.view(-1, x[0].numel()) to MNIST transform
# todo: log or monitor averaged loss? is it accumulating?
# todo: all hyperparameters that affect outcome of training are passed in **config["train"].
#  Ideally, I save these as hyperparameters, and the rest as config.
#  self.save_hyperparameters(ignore=['encoder', 'decoder', 'ground_truth_model'])
