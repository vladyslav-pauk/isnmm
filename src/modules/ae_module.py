import wandb
import pytorch_lightning as pl


class AEModule(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

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

    def test_step(self, batch, batch_idx):
        # data, labels = batch
        # self.update_metrics(data, self.ground_truth.data_model(data), labels)
        # print("Ground truth metric values", self.metrics.compute())
        # todo: refactor data_model so it has a forward method so i can run inference like on model
        import torch
        matrix = self.ground_truth.linear_mixture
        gamma = torch.lgamma(torch.tensor(matrix.size(1))).exp()
        vol = 1 / gamma * torch.det(matrix.T @ matrix).sqrt()
        print(vol.log())
        print("Ground truth mixture matrix:\n", self.ground_truth.linear_mixture)
        print("Decoder mixture matrix:\n", self.decoder.linear_mixture.matrix.numpy())
        # todo: check if (independent on data) is the same as the best value in the validation wandb
