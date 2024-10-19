import wandb
from torchmetrics import MetricCollection


class ModelMetrics(MetricCollection):
    def __init__(self, model):
        metrics = self._setup_metrics(model)
        super().__init__(metrics)

    def _setup_metrics(self, model=None):
        metrics = {}
        wandb.define_metric(name="validation_loss", summary='min')
        return metrics

    def _update(self, data, model_output, labels, idxes, model):
        pass
