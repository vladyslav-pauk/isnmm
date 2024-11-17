import os
import json
import wandb

import torch
from torchmetrics import MetricCollection

import src.modules.metric as metric
import src.model as model_package


class ModelMetrics(MetricCollection):
    def __init__(self, true_model=None, monitor=None):
        self.metrics_list = [monitor]
        self.monitor = monitor
        self.show_plots = False
        self.log_plots = False
        self.log_wandb = True
        self.save_plots = False
        self.true_model = true_model

        self._setup_metrics()

    def _setup_metrics(self):
        all_metrics = {
        }

        if not self.metrics_list:
            self.metrics_list = all_metrics.keys()

        metrics = {name: m for name, m in all_metrics.items() if name in self.metrics_list}

        self.linear_mixture_true = self.true_model.linear_mixture if self.true_model else None

        super().__init__(metrics)
        return metrics

    def _update(self, observed_sample, model_output, labels, idxes, model):
        pass

    def save_metrics(self, metrics, save_dir=None):
        pass

    def unmix(self, latent_sample, model):
        pass