import torchmetrics
import matplotlib.pyplot as plt
import torch
import wandb
import numpy as np


class EvaluateMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.show_plot = show_plot

    def update(self, reconstructed_sample, data, linearly_mixed_sample):
        reconstructed_sample = reconstructed_sample.detach().cpu()
        linearly_mixed_sample = linearly_mixed_sample.detach().cpu()
        self.reconstructed_sample = reconstructed_sample
        self.data = data
        self.linearly_mixed_sample = linearly_mixed_sample

    def compute(self):
        self.plot(self.reconstructed_sample, self.data, self.linearly_mixed_sample)
        return torch.tensor(0.0)

    def toggle_show_plot(self, enable):
        self.show_plot = enable

    def plot(self, reconstructed_sample, data, linearly_mixed_sample):
        plt.figure(figsize=(10, 5))

        for i in range(linearly_mixed_sample.shape[1]):
            plt.subplot(1, linearly_mixed_sample.shape[1], i + 1)
            plt.scatter(linearly_mixed_sample[:, i], self.visual_normalization(reconstructed_sample[:, i]),
                        label=rf'$\hat{{f}}_{i + 1}\circ g_{i + 1}$', alpha=0.5, color='blue')
            plt.scatter(linearly_mixed_sample[:, i], self.visual_normalization(data[:, i]),
                        label=rf'$g_{i + 1}$', alpha=0.5, color='orange')

            plt.xlabel('input', fontsize=12)
            if i == 0:
                plt.ylabel('output', fontsize=12)

            plt.legend(fontsize=12)

        if self.show_plot:
            plt.show()
        else:
            wandb.log({"Evaluate Metric Plot": plt})
            plt.close()

    def visual_normalization(self, x):
        bound = 10
        x = x.cpu() - torch.min(x.cpu())
        x = x.cpu() / torch.max(x.cpu()) * bound
        return x