import torch.nn.functional as functional
from pytorch_lightning import LightningModule

import src.modules.transform as t


class Module(LightningModule):
    def __init__(self):
        super().__init__()

    def _reparameterization(self, sample):
        print(sample.shape)
        if self.encoder_transform is None:
            self.transform = t.Identity()
        else:
            self.transform = getattr(t, self.encoder_transform)()

        return self.transform(sample).unsqueeze(0)

    def _loss_function(self, observed_batch, model_output, idxes):
        reconstructed_sample = model_output["reconstructed_sample"]
        observed_batch = observed_batch.expand_as(reconstructed_sample)
        loss = self._reconstruction_loss(observed_batch, reconstructed_sample)
        loss.update(self._regularization_loss(model_output, observed_batch, idxes))
        return loss

    def _reconstruction_loss(self, data, reconstructed_sample):
        recon_loss = data.size(-1) * getattr(functional, self.distance)(
            reconstructed_sample, data.expand_as(reconstructed_sample), reduction='mean'
        )
        # recon_loss += self.sigma ** 2 * data.size(-1) / 2 * torch.log(torch.tensor(2 * torch.pi * self.sigma ** 2))
        return {"reconstruction": recon_loss}

    def _regularization_loss(self, model_output, observed_batch, idxes):
        return {}


# loss = self.loss_function(x.view(-1, x[0].numel()), x_hat, z_hat, encoder_params, sigma)
# posterior_params = self.encoder(x.view(-1, x[0].numel()))
# task: all hyperparameters that affect outcome of training are passed in **config["train"].
#  Ideally, I save these as hyperparameters, and the rest as config.
#  self.save_hyperparameters(ignore=['encoder', 'decoder', 'ground_truth_model'])
# task: use expectation of the reconstructed x for each z_mc instead of using z_mean
# task: add logging
