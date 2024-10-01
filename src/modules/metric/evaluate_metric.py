import torchmetrics
import matplotlib.pyplot as plt
import torch
import wandb
import numpy as np

from src.helpers.plotter import plot_components


class EvaluateMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False, show_plot=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.show_plot = show_plot

    def update(self, data, linearly_mixed_sample, reconstructed_sample):
        reconstructed_sample = reconstructed_sample.detach().cpu()
        linearly_mixed_sample = linearly_mixed_sample.detach().cpu()
        self.reconstructed_sample = reconstructed_sample
        self.data = data
        self.linearly_mixed_sample = linearly_mixed_sample

    def compute(self):
        plot = plot_components(
            self.linearly_mixed_sample,
            labels=None,
            scale=True,
            Reconstructed=self.reconstructed_sample,
            Data=self.data
        )
        if self.show_plot:
            plot.show()
        else:
            wandb.log({"Evaluate Metric Plot": plot})
            plot.close()
        return torch.tensor(0.0)

    def toggle_show_plot(self, enable):
        self.show_plot = enable

    def plot(self, reconstructed_sample, data, linearly_mixed_sample):
        plt.figure(figsize=(10, 5))

        for i in range(linearly_mixed_sample.shape[1]):
            plt.subplot(1, linearly_mixed_sample.shape[1], i + 1)
            plt.scatter(linearly_mixed_sample[:, i], self.visual_normalization(reconstructed_sample[:, i]), alpha=0.5)
            plt.scatter(linearly_mixed_sample[:, i], self.visual_normalization(data[:, i]), alpha=0.5)

            plt.xlabel('input', fontsize=12)
            if i == 0:
                plt.ylabel('output', fontsize=12)

            plt.legend(fontsize=12)

    def visual_normalization(self, x):
        bound = 10
        x = x.cpu() - torch.min(x.cpu())
        x = x.cpu() / torch.max(x.cpu()) * bound
        return x