import pytorch_lightning as pl
from torchsummary import summary
import torch
from torch.nn import functional as F


class AE(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = None
        self.observed_dim = None
        self.noise_level = 1.0

    def forward(self, observed_batch):
        posterior_parameterization_batch = self.encoder(observed_batch)
        latent_sample = self._sample_posterior(posterior_parameterization_batch)
        reconstructed_sample = self.decoder(latent_sample)

        model_output = {
            "reconstructed_sample": reconstructed_sample,
            "latent_sample": latent_sample,
            "latent_parameterization_batch": posterior_parameterization_batch
        }
        return model_output

    def _sample_posterior(self, variational_parameters):
        mean, std = variational_parameters
        eps = torch.randn(self.mc_samples, *std.shape).to(std.device)
        normal_sample = eps.mul(std.unsqueeze(0)).add_(mean.unsqueeze(0))
        return self._reparameterization(normal_sample)

    @staticmethod
    def _reparameterization(sample):
        return sample

    @staticmethod
    def _reconstruction(data, reconstructed_sample):
        recon_loss = data.size(-1) / 2 * F.mse_loss(
            reconstructed_sample, data.expand_as(reconstructed_sample), reduction='mean'
        )
        # recon_loss += self.sigma ** 2 * data.size(-1) / 2 * torch.log(torch.tensor(2 * torch.pi))
        return recon_loss

    def _loss_function(self, observed_batch, model_output, idxes):
        reconstructed_sample = model_output["reconstructed_sample"].mean(0)
        loss = {"reconstruction": F.mse_loss(reconstructed_sample, observed_batch)}
        loss.update(self._regularization_loss(model_output, observed_batch, idxes))
        return loss

    def _regularization_loss(self, model_output, observed_batch, idxes):
        return {}

    def training_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        loss = self._loss_function(data, self(data), idxes)
        self.log_dict(loss)
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        validation_loss = {"validation_loss": sum(self._loss_function(data, self(data), idxes).values())}
        self._update_metrics(data, self(data), labels, idxes)
        self.log_dict({**validation_loss, **self.metrics.compute()})

    def test_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        model_outputs = self(data)
        self._update_metrics(data, model_outputs, labels, idxes)
        # self.metrics['evaluate_metric'].toggle_show_plot(True)

        final_metrics = self.metrics.compute()
        print("Final metrics:")
        for key, value in final_metrics.items():
            print(f"\t{key} = {value.detach().cpu().numpy()}")

    def on_after_backward(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            self.zero_grad()

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            datamodule = self.trainer.datamodule

            data_sample = next(iter(datamodule.train_dataloader()))
            self.observed_dim = data_sample["data"].shape[1]
            if self.latent_dim is None:
                # self.latent_dim = self.observed_dim
                self.latent_dim = data_sample["labels"]["latent_sample"].shape[1]

            self.encoder.construct(self.latent_dim, self.observed_dim)
            self.decoder.construct(self.latent_dim, self.observed_dim)

            if data_sample["labels"]:
                ground_truth = datamodule
                print("Ground truth model found.")
            else:
                ground_truth = None
            self._setup_metrics(ground_truth)

    def _summary(self):
        summary(self.encoder, (self.observed_dim,))
        summary(self.decoder, (self.latent_dim,))


# todo: refactor data_model so it has a forward method so i can run inference like on model
# todo: check if (independent on data) is the same as the best value in the validation wandb
# loss = self.loss_function(x.view(-1, x[0].numel()), x_hat, z_hat, encoder_params, sigma)
# posterior_params = self.encoder(x.view(-1, x[0].numel()))
# todo: all hyperparameters that affect outcome of training are passed in **config["train"].
#  Ideally, I save these as hyperparameters, and the rest as config.
#  self.save_hyperparameters(ignore=['encoder', 'decoder', 'ground_truth_model'])
# todo: i can also check variance of the residual, that's more statistical metric
# todo: use expectation of the reconstructed x for each z_mc instead of using z_mean
# todo: assign self.latent_dim and self.sigma in ae_module.setup
