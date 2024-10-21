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
        self.unmixing = False

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
            # if param is not None and param.grad is not None:
            #     if isnan(param.grad).any() or isinf(param.grad).any():
            #         valid_gradients = False
            #         break
            # else:
            #     valid_gradients = False
            #     break

        if not valid_gradients:
            self.zero_grad()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            datamodule = self.trainer.datamodule
            data_sample = next(iter(datamodule.train_dataloader()))

            self.observed_dim = data_sample["data"].shape[1]

            if data_sample["labels"]:
                print("Labelled data found")

                if self.latent_dim is None:
                    self.latent_dim = data_sample["labels"]["latent_sample"].shape[-1]

                if self.sigma is None:
                    self.sigma = datamodule.sigma

            if self.unmixing:
                print(f"Unmixing latent sample with {self.unmixing}")

            self.save_hyperparameters({"data_config": datamodule.data_config})

            self.encoder.construct(self.latent_dim, self.observed_dim)
            self.decoder.construct(self.latent_dim, self.observed_dim)
            self.metrics = getattr(exp_module, self.experiment_metrics).ModelMetrics(datamodule).eval()

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
