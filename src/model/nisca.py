import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from src.modules.network import LinearPositive, FCNConstructor
# from src.modules.vae_module import VAEModule
from src.model.vasca import Model as VASCA
from src.model.vasca import Encoder
import src.modules.metric as metric


class Model(VASCA):
    def __init__(self, ground_truth_model=None, encoder=None, decoder=None, train_config=None):
        super().__init__(ground_truth_model=ground_truth_model, encoder=encoder, decoder=decoder, train_config=train_config)

        self.ground_truth = ground_truth_model

        self.metrics = torchmetrics.MetricCollection({
            'mixture_mse_db': metric.MatrixMse(),
            'mixture_sam': metric.SpectralAngle(),
            'z_subspace': metric.SubspaceDistance(),
            # 'h_r_square': metric.ResidualNonlinearity()
        })
        # self.metrics.eval()

        # self.latent_prior_distribution = Dirichlet  # (torch.ones(self.latent_dim))
        # self.noise_distribution = Normal
        # self.variational_posterior_distribution = LogNormal

    def update_metrics(self, data, model_output, labels):
        self.metrics['mixture_mse_db'].update(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        self.metrics['mixture_sam'].update(
            self.ground_truth.data_model.linear_mixture.matrix, self.decoder.linear_mixture.matrix
        )
        self.metrics['z_subspace'].update(
            labels[0], model_output[1].mean(dim=0)
        )
        # self.metrics['h_r_square'].update(
        #     labels[0],
        #     self.ground_truth.data_model.linear_mixture.matrix,
        #     self.ground_truth.data_model.nonlinear_transform, self.decoder.nonlinear_transform
        # )

    # def metrics(self, x, z, x_mc_sample, z_mc_sample, variational_parameters, decoder):
    #
    #     return self.metrics.compute_metrics(x, z, x_mc_sample, z_mc_sample, variational_parameters, decoder)

    # todo: create metric class and initialize on datamodule and pass into model class.
    #  method metric is then calling the metric class methodss
    # def metrics(self, x, z, x_mc_sample, z_mc_sample, variational_parameters=None):
    #
    #     # true_mixing_A = self.data_model.lin_transform.to(x.device)
    #     # model_mixing_A = self.decoder.lin_transform.matrix.to(x.device)
    #     #
    #     # true_nonlinearity = self.data_model.nonlinear_transform
    #     # model_nonlinearity = self.decoder.nonlinear_transform
    #     #
    #     # metrics = {}
    #     # for metric in self.metrics:
    #     #
    #     #     if metric == "A_MSE_dB":
    #     #         A_mse_dB = mse_matrix_db(true_mixing_A, model_mixing_A)
    #     #         metrics["A_MSE_dB"] = A_mse_dB
    #     #
    #     #     elif metric == "A_SAM":
    #     #         A_sam = spectral_angle_distance(true_mixing_A, model_mixing_A)
    #     #         metrics["A_SAM"] = A_sam
    #     #
    #     #     elif metric == "z_MSE_dB":
    #     #         z_mse_dB = mse_matrix_db(z, z_mc_sample.mean(dim=0))
    #     #         metrics["z_mse_dB"] = z_mse_dB
    #     #
    #     #     elif metric == "z_subspace":
    #     #         z_subspace = subspace_distance(z, z_mc_sample.mean(dim=0))
    #     #         metrics["z_subspace"] = z_subspace
    #     #
    #     #     elif metric == "h_R-squared":
    #     #         h_rsquared = residual_nonlinearity(
    #     #             z @ true_mixing_A.T,
    #     #             true_nonlinearity,
    #     #             model_nonlinearity,
    #     #             show_plot=False
    #     #         ).rsquared.mean()
    #     #         metrics["h_R-squared"] = h_rsquared
    #     # metrics = self.metrics_module.compute_metrics(x, z, x_mc_sample, z_mc_sample, variational_parameters, decoder=self.decoder)
    #     self.metric_a_mse_db.update()
    #     self.metric_a_sam.update()
    #     self.metric_z_subspace.update()
    #
    #     # wandb.define_metric(self.monitor, summary="max")
    #
    #     metrics = {
    #         "a_mse_db": self.a_mse_metric.compute(),
    #         "a_sam": self.spectral_angle_metric.compute(),
    #         "subspace_distance": self.subspace_distance_metric.compute()
    #     }
    #     return metrics

    # def metric(self, x, z, x_mc_sample, z_mc_sample, variational_parameters=None):
    #
    #     true_mixing_A = self.data_model.dataset.lin_transform.to(x.device)
    #     model_mixing_A = self.decoder.lin_transform.matrix.to(x.device)
    #
    #     true_nonlinearity = self.data_model.dataset.nonlinear_transform
    #     model_nonlinearity = self.decoder.nonlinear_transform
    #
    #     A_mse_dB = mse_matrix_db(true_mixing_A, model_mixing_A)
    #     A_sam = spectral_angle_distance(true_mixing_A, model_mixing_A)
    #
    #     z_mse_dB = mse_matrix_db(z, z_mc_sample.mean(dim=0))
    #     z_subspace = subspace_distance(z, z_mc_sample.mean(dim=0))
    #     h_rsquared = self.residual_nonlinearity(
    #         z @ true_mixing_A.T,
    #         true_nonlinearity,
    #         model_nonlinearity,
    #         show_plot=False
    #     ).rsquared.mean()
    #
    #     wandb.define_metric(self.config.train.monitor, summary="max")
    #     return {
    #         "A_mse_dB": A_mse_dB,
    #         # "A_SAM": A_sam,
    #         # "z_mse_dB": z_mse_dB,
    #         # "z_subspace": z_subspace,
    #         # "h_rsquared": h_rsquared
    #     }

        # wandb.define_metric("subspace_distance", summary="min")
        # r_squared = torch.tensor([0.0])

        # z_recon = (torch.linalg.pinv(true_mixing_A) @ residual_nonlinearity((model_mixing_A @ z.T).T).T).T
        # reconstruction_subspace_distance = subspace_distance(z, z_recon)
        # reconstruction_subspace_distance = torch.norm(z - z_recon, dim=-1).mean()

        # print(residual_nonlinearity(y_true), fitter(y_true))

        # nonlinearity_plot = plot_components(
        #     y_true,
        #     # true_nonlinearity=true_nonlinearity,
        #     # inferred_nonlinearity=model_nonlinearity,
        #     residual_nonlinearity=residual_nonlinearity,
        #     linear_fit=fitter
        # )
        # nonlinearity_plot.show()
        # nonlinearity_plot = plot_components(
        #     y_true,
        #     residual_nonlinearity=residual_nonlinearity,
        #     # Create a component-wise lambda for the linear fit
        #     linear_fit=lambda x: torch.cat(
        #         [fitter.slopes[i] * x[:, i:i + 1] + fitter.intercepts[i] for i in range(x.shape[-1])], dim=-1
        #     )
        # )

        # print(reconstruction_subspace_distance)

        # reconstruction_subspace_distance = torch.norm(z - z_recon, dim=-1).mean()

        # data_nonlinearity = lambda x: self.data_model.dataset.nonlinear_transform(x)
        # data_nonlinearity_i = lambda x: self.data_model.dataset.nonlinear_transform.inverse(x)
        # model_nonlinearity = lambda x: self.decoder.nonlinear_transform(x)

        # W = torch.linalg.pinv(A0) @ A_hat
        # print(A0 @ torch.linalg.pinv(A0))
        # print(A0 @ W, A_hat)

        # print(torch.linalg.det(A0), torch.linalg.det(A_hat))
        # Rr = torch.rand(self.latent_dim, self.latent_dim)
        # R = torch.linalg.pinv(A0) @ A_hat
        # R = A_hat @ A_hat.T
        # print(W.T @ A0.T - A_hat.T)
        # z_true = (A0 @ z.T).T
        # print((z @ A0.T).shape)
        # print(remainining_nonlinearity(A0 @ z[0]))
        # sys.exit()
        # print(model_nonlinearity((A_hat @ z.T).T).shape)

        # z_recon = (R @ z.T).T
        # z_reconr = (Rr @ z.T).T

        # print(ssd)

        # ssdr = subspace_distance(z, z_reconr)
        # print(ssd, ssdr)
        # print(R)
        # print(torch.linalg.det(R))

        # ssd = subspace_distance(z @ A0.T, z @ A0.T @ R.T)


# class Encoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, config_encoder):
#         super().__init__()
#
#         self.mu_network = FCNConstructor(input_dim, latent_dim - 1, **config_encoder)
#         self.log_var_network = FCNConstructor(input_dim, latent_dim - 1, **config_encoder)
#
#     def forward(self, x):
#         mu = self.mu_network.forward(x)
#         log_var = self.log_var_network.forward(x)
#         return mu, log_var
# fixme: fix slow training, initialization (loss infinity), mc and batch in fcnconstructor, activation argument (to config?)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, config_decoder):
        super(Decoder, self).__init__()

        self.linear_mixture = LinearPositive(torch.ones(output_dim, latent_dim), **config_decoder)

        # self.nonlinear_transform = ComponentWiseNonlinear(output_dim, **config_decoder)

        self.nonlinear_transform = nn.ModuleList([
            FCNConstructor(
                input_dim=1, output_dim=1, **config_decoder
            ) for _ in range(output_dim)
        ])
        # todo: no need for a class, compose from constructors here

    def forward(self, z):
        x = self.linear_mixture(z)
        # x = self.nonlinear_transform(y)

        x = torch.cat([
            self.nonlinear_transform[i](x[..., i:i + 1].view(-1, 1)).view_as(x[..., i:i + 1])
            for i in range(x.shape[-1])
        ], dim=-1)

        return x
        # todo: check the networks once again, make sure everything is consistent and implemented right, ask gpt to improve

# fixme: clean up and test nisca model, initialize on top of vasca super. just modify the metric and decoder
#  neural network output is horizontal!!! (nearly constant)  something is wrong


# class Network(nn.Module):
#     def __init__(self, output_dim, hidden_layers, activation=None, output_activation=None, weight_initialization=None, **kwargs):
#         super(Network, self).__init__()
#         self.component_wise_nets = nn.ModuleList([
#             self._build_component_wise_net(hidden_layers, activation)
#             for _ in range(output_dim)
#         ])
#         self._initialize_weights(activation, weight_initialization) if len(hidden_layers) != 0 else None
#
#     def _build_component_wise_net(self, hidden_layers, activation):
#         if len(hidden_layers) == 0:
#             return nn.Identity()
#         layers = []
#         input_size = 1
#         for hidden_size in hidden_layers.values():
#             layers.append(nn.Linear(input_size, hidden_size))
#             if activation:
#                 layers.append(getattr(nn, activation)())  # Adding activation
#             input_size = hidden_size
#
#         layers.append(nn.Linear(input_size, 1))
#         return nn.Sequential(*layers)
#
#     def _initialize_weights(self, activation, init_weights):
#         for net in self.component_wise_nets:
#             for layer in net:
#                 if isinstance(layer, nn.Linear):
#                     if init_weights:
#                         getattr(nn.init, init_weights)(layer.weight, nonlinearity=activation.lower())
#                     if layer.bias is not None:
#                         nn.init.zeros_(layer.bias)
#
#     def forward(self, x):
#         transformed_components = [
#             self.component_wise_nets[i](x[..., i:i + 1]) for i in range(x.shape[-1])
#         ]
#         return torch.cat(transformed_components, dim=-1).abs()