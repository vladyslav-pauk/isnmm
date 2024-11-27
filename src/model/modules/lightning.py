import torch.optim as optim
from torch import isnan, isinf
from pytorch_lightning import LightningModule


class Module(LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.latent_dim = None
        self.observed_dim = None
        self.metrics = None
        self.unmixing = False

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, observed_batch):
        posterior_parameterization = self.encoder(observed_batch)
        latent_sample, latent_sample_mean = self._reparameterization(posterior_parameterization)
        reconstructed_sample = self.decoder(latent_sample)

        model_output = {
            "reconstructed_sample": reconstructed_sample,
            "latent_sample": latent_sample,
            "latent_sample_mean": latent_sample_mean,
            "posterior_parameterization": posterior_parameterization
        }
        return model_output

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

            if "labels" in data_sample.keys():
                print("Labelled data found")

                if self.latent_dim is None and "latent_sample" in data_sample["labels"].keys():
                    self.latent_dim = data_sample["labels"]["latent_sample"].shape[-1]

                if self.sigma is None:
                    self.sigma = datamodule.sigma

            if self.trainer.model.model_config["latent_dim"]:
                self.latent_dim = self.trainer.model.model_config["latent_dim"]

            if self.unmixing:
                print(f"Unmixing latent sample with {self.unmixing}")

            self.save_hyperparameters({"data_config": datamodule.data_config})

            self.encoder.construct(self.latent_dim, self.observed_dim)
            self.decoder.construct(self.latent_dim, self.observed_dim)

            self.metrics.true_model = self.trainer.datamodule
            self.metrics.latent_dim = self.latent_dim

        if stage == 'predict':
            if self.trainer.model.model_config["latent_dim"]:
                self.latent_dim = self.trainer.model.model_config["latent_dim"]
            self.metrics.true_model = self.trainer.datamodule
            self.metrics.latent_dim = self.trainer.model.latent_dim
            self.metrics.unmixing = self.model_config["unmixing"]

    def on_train_start(self) -> None:
        if self.metrics.log_wandb:
            for metric_name in self.metrics:
                if metric_name == self.metrics.monitor:
                    import wandb
                    wandb.define_metric(name=metric_name, summary='min')

    def training_step(self, batch, batch_idx):
        data, idxes = batch["data"], batch["idxes"]
        loss = self._loss_function(data, self(data), idxes)
        self.log_dict(loss)
        return sum(loss.values())

    def val_dataloader(self):
        return self.train_dataloader()

    def on_validation_start(self):
        self.metrics.log_wandb = True
        self.metrics.log_plots = False
        self.metrics.show_plots = False
        self.metrics.save_plot = False
        self.metrics.setup_metrics(metrics_list=[])

    def validation_step(self, batch, batch_idx):
        data, idxes = batch["data"], batch["idxes"]
        if "labels" in batch.keys():
            labels = batch["labels"]
        else:
            labels = None

        validation_loss = {"validation_loss": sum(self._loss_function(data, self(data), idxes).values())}
        self.metrics.update(data, self(data), labels, idxes, self)
        self.log_dict(validation_loss)
        return validation_loss

    def validation_end(self, batch, outs) -> None:
        self.log_dict({**self.metrics.compute()})

    def on_test_start(self):
        self.metrics.log_wandb = False
        self.metrics.log_plots = False
        self.metrics.show_plots = False
        self.metrics.save_plot = False
        self.metrics.setup_metrics(metrics_list=[])

    def test_step(self, batch, batch_idx):
        data, idxes = batch["data"], batch["idxes"]
        if "labels" in batch.keys():
            labels = batch["labels"]
        else:
            labels = None

        self.metrics.update(data, self(data), labels, idxes, self)

    def on_test_end(self) -> None:
        final_metrics = self.metrics.compute()
        self.metrics.save(final_metrics)

    def on_predict_start(self) -> None:
        self.metrics.log_wandb = False
        self.metrics.log_plot = False
        self.metrics.show_plot = True
        self.metrics.save_plot = True
        self.metrics.setup_metrics(metrics_list=[])

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data, idxes = batch["data"], batch["idxes"]
        if "labels" in batch.keys():
            labels = batch["labels"]
        else:
            labels = None

        self.metrics.update(data, self(data), labels, idxes, self)

    def on_predict_end(self) -> None:
        final_metrics = self.metrics.compute()
        self.metrics.save(final_metrics, save_dir=f'predictions')

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
