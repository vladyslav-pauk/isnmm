from torch import zeros, ones, randn

import src.modules.transform as t
from src.modules.distribution.location_scale import LocationScale
import src.modules.distribution.standard as distribution
from src.model.modules.ae import Module as AutoEncoder


class Module(AutoEncoder):
    def __init__(self):
        super().__init__()

    def _reparameterization(self, variational_parameters):
        loc, scale = variational_parameters

        eps = randn(self.mc_samples, *scale.shape).to(scale.device)
        base_sample = eps.mul(scale.unsqueeze(0)).add_(loc.unsqueeze(0))

        if self.encoder_transform is None:
            self.transform = t.Identity()
        else:
            self.transform = getattr(t, self.encoder_transform)()
        return self.transform(base_sample), self.transform(loc.unsqueeze(0))

    def _regularization_loss(self, model_output, data, idxes):
        latent_sample = model_output["latent_sample"]
        prior, posterior = self._model(*model_output["posterior_parameterization"])

        neg_entropy_posterior = posterior.log_prob(latent_sample)
        log_prior = prior.log_prob(latent_sample)
        kl_posterior_prior = neg_entropy_posterior - log_prior

        return {"kl_posterior_prior": 2 * self.sigma**2 * kl_posterior_prior.mean()}
        # task: add beta

    def _model(self, loc, scale):
        prior = getattr(distribution, self.prior_config['base_distribution'])(zeros(self.latent_dim), ones(self.latent_dim))
        base_distribution = getattr(distribution, self.posterior_config['base_distribution'])

        posterior = LocationScale(
            base_distribution(loc, scale), self.transform
        )
        return prior, posterior
