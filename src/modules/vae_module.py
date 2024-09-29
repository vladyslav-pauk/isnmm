from src.modules.ae_module import AEModule


class VAEModule(AEModule):
    def __init__(self, encoder, decoder, mc_samples=None, latent_dim=None, optimizer=None, **kwargs):
        super().__init__(encoder, decoder)

        self.latent_dim = latent_dim
        self.number_mc_samples = mc_samples
        self.optimizer = optimizer

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

    def forward(self, x):
        variational_parameters = self.encoder(x)
        latent_mc_sample = self.reparameterize(variational_parameters, self.number_mc_samples)
        observed_mc_sample = self.decoder(latent_mc_sample)
        return observed_mc_sample, latent_mc_sample, variational_parameters


# loss = self.loss_function(x.view(-1, x[0].numel()), x_hat, z_hat, encoder_params, sigma)
# posterior_params = self.encoder(x.view(-1, x[0].numel()))
# todo: move x = x.view(-1, x[0].numel()) to MNIST transform
# todo: log or monitor averaged loss? is it accumulating?
# todo: all hyperparameters that affect outcome of training are passed in **config["train"].
#  Ideally, I save these as hyperparameters, and the rest as config.
#  self.save_hyperparameters(ignore=['encoder', 'decoder', 'ground_truth_model'])
