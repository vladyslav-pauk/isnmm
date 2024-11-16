import torch
from torch import nn

from src.modules.optimizer.augmented_lagrange import AugmentedLagrangeMultiplier
import src.modules.network as network
from src.model.modules.lightning import Module as LightningModule
from src.model.modules.ae import Module as Autoencoder


class Model(LightningModule, Autoencoder):
    def __init__(self, encoder, decoder, model_config, optimizer_config, metrics=None):
        super().__init__(encoder, decoder)

        self.optimizer = None
        self.optimizer_config = optimizer_config

        self.metrics = metrics

        self.latent_dim = None
        self.mc_samples = 1
        self.sigma = 0
        self.unmixing = model_config["unmixing"]

        self.distance = model_config["distance"]
        self.encoder_transform = model_config["reparameterization"]

    def _regularization_loss(self, model_output, observed_batch, idxes):
        latent_sample = model_output["latent_sample"]
        return self.optimizer.compute_regularization_loss(latent_sample.mean(dim=0), observed_batch.mean(dim=0), idxes)

    def configure_optimizers(self):
        self.optimizer = AugmentedLagrangeMultiplier(
            params=list(self.parameters()),
            constraint_fn=self._constraint,
            optimizer_config=self.optimizer_config
        )
        return self.optimizer

    @staticmethod
    def _constraint(latent_sample):
        return torch.sum(latent_sample, dim=-1) - 1.0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.optimizer.update_buffers(batch["idxes"], self(batch["data"])["latent_sample"])


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture_inv = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        # if latent_dim != observed_dim:
        #     self.linear_mixture_inv = network.LinearPositive(
        #         torch.eye(latent_dim, observed_dim), **self.config
        #     )
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

    def forward(self, x):
        x = self.nonlinear_transform(x)
        x = self.linear_mixture_inv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.constructor = getattr(network, config["constructor"])
        self.linear_mixture = nn.Identity()
        self.nonlinear_transform = nn.Identity()

    def construct(self, latent_dim, observed_dim):
        # if latent_dim != observed_dim:
        #     self.linear_mixture = network.LinearPositive(
        #         torch.eye(observed_dim, latent_dim), **self.config
        #     )
        self.nonlinear_transform = self.constructor(observed_dim, observed_dim, **self.config)

    def forward(self, x):
        x = self.linear_mixture(x)
        x = self.nonlinear_transform(x)
        return x

# todo: unequal dimensions by stacking and separate optimization
# fixme: compare CNAE paper synthetic results with mine

# def _create_random_subsets(self, total_dim, latent_dim, num_subsets):
    #     import torch, random, numpy
    #     indices = list(torch.arange(total_dim))
    #     random.shuffle(indices)
    #
    #     if num_subsets is None:
    #         num_subsets = int(numpy.ceil(total_dim / latent_dim))
    #
    #     subsets = [[] for _ in range(num_subsets)]
    #
    #     for i, idx in enumerate(indices):
    #         subsets[i % num_subsets].append(idx)
    #
    #     while any(len(subset) < latent_dim for subset in subsets):
    #         for subset in subsets:
    #             if len(subset) < latent_dim:
    #                 remaining_indices = [i for i in indices if i not in subset]
    #                 if not remaining_indices:
    #                     remaining_indices = indices
    #                 subset.append(random.choice(remaining_indices))
    #
    #     return [torch.tensor(subset, dtype=torch.long) for subset in subsets]

    # def configure_optimizers(self):
    #     optimizer_class = getattr(optim, self.optimizer_config["name"])
    #     optimizers = []
    #
    #     for i, subset in enumerate(self.subsets):
    #         encoder_params = [
    #             p for name, p in self.encoder.named_parameters()
    #             if any(str(sub.item()) in name for sub in subset)
    #         ]
    #         decoder_params = []
    #         for sub_part in ['linear_mixture', 'nonlinear_transform']:
    #             decoder_params += [
    #                 p for name, p in getattr(self.decoder, sub_part).named_parameters()
    #                 if any(str(sub.item()) in name for sub in subset)
    #             ]
    #
    #         subset_params = encoder_params + decoder_params
    #
    #         if subset_params:
    #             if 'linear_mixture' in subset_params:
    #                 lr = self.optimizer_config["lr"]["decoder"]["linear_mixture"]
    #             elif 'nonlinear_transform' in subset_params:
    #                 lr = self.optimizer_config["lr"]["decoder"]["nonlinear_transform"]
    #             else:
    #                 lr = self.optimizer_config["lr"]["encoder"]
    #
    #             optimizer_params = {'params': subset_params, 'lr': lr}
    #             optimizer = optimizer_class([optimizer_params], **self.optimizer_config["params"])
    #             optimizers.append(optimizer)
    #
    #     if self.optimizer_config["scheduler"]:
    #         schedulers = []
    #         for optimizer in optimizers:
    #             scheduler_class = getattr(optim.lr_scheduler, self.optimizer_config["scheduler"])
    #             scheduler = scheduler_class(optimizer, **self.optimizer_config["scheduler_params"])
    #             schedulers.append(scheduler)
    #         return optimizers, schedulers
    #     else:
    #         return optimizers