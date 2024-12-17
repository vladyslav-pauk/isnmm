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

        self.save_hyperparameters(ignore=['encoder', 'decoder', 'metrics'])

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
        # print(f"- {stage.split('.')[-1].capitalize()} stage")
        datamodule = self.trainer.datamodule
        self.save_hyperparameters({"data_config": datamodule.data_config})
        data_sample = next(iter(datamodule.train_dataloader()))
        self.observed_dim = data_sample["data"].shape[1]

        if 'labels' in data_sample.keys() and data_sample['labels']:
            print("Labelled data found")

            if self.latent_dim is None and "latent_sample" in data_sample["labels"].keys():
                self.latent_dim = data_sample["labels"]["latent_sample"].shape[-1]

            if self.sigma is None:
                self.sigma = datamodule.sigma

        if self.trainer.model.model_config["latent_dim"]:
            self.latent_dim = self.trainer.model.model_config["latent_dim"]

        self.metrics.eval()
        if stage == 'fit':
            # datamodule = self.trainer.datamodule
            self.model_config["latent_dim"] = self.latent_dim
            self.model_config["observed_dim"] = self.observed_dim
            self.save_hyperparameters({"data_config": datamodule.data_config})
            # data_sample = next(iter(datamodule.train_dataloader()))
            # self.observed_dim = data_sample["data"].shape[1]
            #
            # if "labels" in data_sample.keys():
            #     print("Labelled data found")
            #
            #     if self.latent_dim is None and "latent_sample" in data_sample["labels"].keys():
            #         self.latent_dim = data_sample["labels"]["latent_sample"].shape[-1]
            #
            #     if self.sigma is None:
            #         self.sigma = datamodule.sigma
            #
            # if self.trainer.model.model_config["latent_dim"]:
            #     self.latent_dim = self.trainer.model.model_config["latent_dim"]

            if self.unmixing:
                print(f"Unmixing latent sample with {self.unmixing}")

                # _, A_gt, S_gt, _, _ = loadhsi('urban')
                # mixing_init = svmax(A_gt @ S_gt, self.latent_dim)
                # mixing_init, pos_init = reorder_columns_angle(A_gt, mixing_init)

            self.encoder.construct(self.latent_dim, self.observed_dim)
            self.decoder.construct(self.latent_dim, self.observed_dim)#, mixing_init)

        # if stage == 'predict':
        #     self.metrics.true_model = self.trainer.datamodule
        #     self.metrics.model = self
        #     # self.metrics.latent_dim = self.trainer.model.latent_dim
        #     # self.metrics.unmixing = self.model_config["unmixing"]

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
        self.metrics.true_model = self.trainer.datamodule
        self.metrics.model = self

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
        # print("Testing")
        self.metrics.true_model = self.trainer.datamodule
        self.metrics.model = self

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
        # print("Predicting")
        self.metrics.true_model = self.trainer.datamodule
        self.metrics.model = self

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


import torch
def pca(Y, N):
    d = torch.mean(Y, dim=1).unsqueeze(1)
    Y_cen = Y - d
    _, C = torch.linalg.eigh(Y_cen @ Y_cen.T)
    ls = list(range(Y.shape[0]))[::-1]
    C = C[:, ls][:, :N-1]
    Y_red = torch.pinverse(C) @ Y_cen
    return Y_red, C, d


def svmax(Y, N):
    # pca
    X, C, d = pca(Y, N)
    # svmax
    X_bar = torch.cat((X, torch.ones((1, X.shape[1]), device=X.device)), dim=0)
    A_est = torch.tensor([], device=X.device)
    idx = []
    P = torch.eye(N, device=X.device)
    for i in range(N):
        _, idx_i = torch.max(torch.sum((P @ X_bar) ** 2, dim=0), 0)
        idx.append(idx_i.item())
        A_est = torch.cat((A_est, X[:, idx_i].unsqueeze(1)), dim=1)
        F = torch.cat((A_est, torch.ones((1, A_est.shape[1]), device=X.device)), dim=0)
        P = torch.eye(N, device=X.device) - F @ torch.pinverse(F)
    A_est = C @ A_est + d
    return A_est


from scipy.optimize import linear_sum_assignment
def reorder_columns_angle(A1, A2):
    # match how first can be derived by second, i.e., A1 = A2[:, order]
    # A1 should be ground truth; A2 should be the estimated matrix to be reordered

    # Normalize the columns of A1and A2
    A1_norm = A1.float() / torch.linalg.norm(A1, dim=0)
    A2_norm = A2.float() / torch.linalg.norm(A2, dim=0)
    # Calculate the cosine of the angle between each column in A1 and A2
    cos_angle_matrix = torch.mm(A1_norm.T, A2_norm).detach()
    angle_matrix = torch.acos(torch.clamp(cos_angle_matrix, -1, 1))
    # Use the Hungarian algorithm to find the optimal assignment of columns
    _, col_ind = linear_sum_assignment(angle_matrix.cpu().numpy())
    A_reordered = A2[:, col_ind]

    return A_reordered, col_ind


import torch
import numpy as np
import scipy.io as scio
def loadhsi(case):
    '''
    :input: case: for different datasets,
                 'toy' and 'usgs' are synthetic datasets
    :return: Y : HSI data of size [Bands,N]
             A_ture : Ground Truth of abundance map of size [P,N]
             P : nums of endmembers signature
    '''

    if case == 'ridge':
        file = 'PGMSU/dataset/JasperRidge2_R198.mat'
        data = scio.loadmat(file)
        Y = data['Y']
        nRow, nCol = data['nRow'][0][0], data['nCol'][0][0]
        if np.max(Y) > 1:
            Y = Y / np.max(Y)
        Y = np.reshape(Y, [198, 100, 100])
        for i, y in enumerate(Y):
            Y[i] = y.T
        Y = np.reshape(Y, [198, 10000])

        GT_file = 'PGMSU/dataset/JasperRidge2_end4.mat'
        S_gt = scio.loadmat(GT_file)['A']
        A_gt = scio.loadmat(GT_file)['M']
        S_gt = np.reshape(S_gt, (4, 100, 100))
        for i, A in enumerate(S_gt):
            S_gt[i] = A.T
        S_gt = np.reshape(S_gt, (4, 10000))

    elif case == 'cuprite':
        file = 'dataset/Cuprite/CupriteS1_R188.mat'
        data = scio.loadmat(file)
        Y = data['Y']
        SlectBands = data['SlectBands'].squeeze()
        nRow, nCol = data['nRow'][0][0], data['nCol'][0][0]

        GT_file = 'dataset/Cuprite/groundTruth_Cuprite_nEnd12.mat'
        A_gt = scio.loadmat(GT_file)['M'][SlectBands, :]
        # GT_file = 'dataset/Cuprite/AVIRIS_corrected (MoffettField).mat'

        Y = np.delete(Y, [0, 1, 135], axis=0)
        A_gt = np.delete(A_gt, [0, 1, 135], axis=0)
        if np.max(Y) > 1:
            Y = Y / np.max(Y)


    elif case == 'urban':
        file = '../datasets/hyperspectral/Urban_R162/data.mat'
        data = scio.loadmat(file)
        Y = data['Y']  # (C,w*h)
        nRow, nCol = data['nRow'][0][0], data['nCol'][0][0]

        GT_file = '../datasets/hyperspectral/Urban_R162/end4_groundTruth.mat'
        S_gt = scio.loadmat(GT_file)['A']
        A_gt = scio.loadmat(GT_file)['M']
        if np.max(Y) > 1:
            Y = Y / np.max(Y)

    Y = torch.tensor(Y).float()
    A_gt = torch.tensor(A_gt).float()
    if 'S_gt' in locals():
        S_gt = torch.tensor(S_gt).float()
    else:
        S_gt = torch.ones((A_gt.shape[1], Y.shape[1]))

    return Y, A_gt, S_gt, nRow, nCol