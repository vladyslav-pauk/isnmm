import torch.optim as optim
from torch import isnan, isinf
from pytorch_lightning import LightningModule

import src.experiments as exp_module


class Module(LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.latent_dim = None
        self.observed_dim = None
        self.encoder = encoder
        self.decoder = decoder
        self.metrics = None

    def forward(self, observed_batch):
        posterior_parameterization = self.encoder(observed_batch)
        latent_sample = self._reparameterization(posterior_parameterization)
        reconstructed_sample = self.decoder(latent_sample)

        model_output = {
            "reconstructed_sample": reconstructed_sample,
            "latent_sample": latent_sample,
            "posterior_parameterization": posterior_parameterization
        }
        return model_output

    def training_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        loss = self._loss_function(data, self(data), idxes)
        self.log_dict(loss)
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        validation_loss = {"validation_loss": sum(self._loss_function(data, self(data), idxes).values())}
        self.metrics._update(data, self(data), labels, idxes, self)
        self.log_dict({**validation_loss, **self.metrics.compute()})

    def test_step(self, batch, batch_idx):
        data, labels, idxes = batch["data"], batch["labels"], batch["idxes"]
        model_outputs = self(data)
        self.metrics._update(data, model_outputs, labels, idxes, self)

        final_metrics = self.metrics.compute()
        print("Final metrics:")
        for key, value in final_metrics.items():
            print(f"\t{key} = {value.detach().cpu().numpy()}")

    def val_dataloader(self):
        return self.train_dataloader()

    def on_after_backward(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (isnan(param.grad).any() or isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            self.zero_grad()

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            datamodule = self.trainer.datamodule
            data_sample = next(iter(datamodule.train_dataloader()))

            self.observed_dim = data_sample["data"].shape[1]
            if self.latent_dim is None and data_sample["labels"]:
                self.latent_dim = data_sample["labels"]["latent_sample"].shape[-1]
                print("Labelled data found.")

            if self.sigma is None:
                self.sigma = datamodule.sigma
                print("Ground truth model found.")

            self.subsets = self._create_random_subsets(self.observed_dim, self.latent_dim, self.num_subsets)

            self.encoder.construct(self.latent_dim, self.observed_dim)
            self.decoder.construct(self.latent_dim, self.observed_dim)
            self.metrics = getattr(exp_module, self.experiment_metrics).ModelMetrics(datamodule).eval()

    def _create_random_subsets(self, total_dim, latent_dim, num_subsets):
        import torch, random, numpy
        indices = list(torch.arange(total_dim))
        random.shuffle(indices)

        if num_subsets is None:
            num_subsets = int(numpy.ceil(total_dim / latent_dim))

        subsets = [[] for _ in range(num_subsets)]

        for i, idx in enumerate(indices):
            subsets[i % num_subsets].append(idx)

        while any(len(subset) < latent_dim for subset in subsets):
            for subset in subsets:
                if len(subset) < latent_dim:
                    remaining_indices = [i for i in indices if i not in subset]
                    if not remaining_indices:
                        remaining_indices = indices
                    subset.append(random.choice(remaining_indices))

        return [torch.tensor(subset, dtype=torch.long) for subset in subsets]

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.optimizer_config["name"])
        optimizer_params = []

        for model_part, lr_value in self.optimizer_config["lr"].items():
            if isinstance(lr_value, dict):
                for sub_part, sub_lr_value in lr_value.items():
                    model_params = getattr(getattr(self, model_part), sub_part).parameters()
                    optimizer_params.append({'params': model_params, 'lr': sub_lr_value})
            else:
                model_params = getattr(self, model_part).parameters()
                optimizer_params.append({'params': model_params, 'lr': lr_value})

        optimizer = optimizer_class(optimizer_params, **self.optimizer_config["params"])

        if self.optimizer_config["scheduler"]:
            scheduler = getattr(optim.lr_scheduler, self.optimizer_config["scheduler"])
            scheduler = scheduler(optimizer, gamma=0.99)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    **self.optimizer_config["scheduler_params"]
                }
            }
        else:
            return {"optimizer": optimizer}

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