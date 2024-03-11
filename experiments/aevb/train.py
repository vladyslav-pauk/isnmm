from pytorch_lightning import Trainer

from src.utils import init_logger
from src.datamodule import MNISTDataModule
from src.networks import Encoder, Decoder
from src.model.aevb import AEVB


if __name__ == '__main__':

    encoder = Encoder(input_dim=784, hidden_dim=400, latent_dim=200)
    decoder = Decoder(latent_dim=200, hidden_dim=400, output_dim=784)

    model = AEVB(encoder=encoder, decoder=decoder)

    datamodule = MNISTDataModule()

    logger = init_logger(project="nica-vae", experiment="aevb")
    logger.watch(model, log='parameters')

    trainer = Trainer(max_epochs=10,
                      logger=logger,
                      accelerator="mps")
    trainer.fit(model, datamodule)

# feat: data generator
# feat: probabilistic model
# feat: metrics
# feat: plots
