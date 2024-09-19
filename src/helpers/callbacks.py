from torch import isnan
from pytorch_lightning.callbacks import Callback


class EarlyStoppingThreshold(Callback):
    def __init__(self, monitor: str, patience: int, mode: str, min_delta: float):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.stopping_threshold = min_delta
        self.wait_count = 0
        self.stopped_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if isnan(current):
            trainer.should_stop = True
            print(f"Early stopping: {self.monitor} is NaN.")

        if self._is_over_threshold(current):
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
        else:
            self.wait_count = 0

    def on_train_end(self, trainer, pl_module):
        comparison = "less than" if self.mode == 'min' else "more than"
        if self.stopped_epoch > 0:
            print(
                f"Early stopping: {self.monitor} is {comparison} {self.stopping_threshold} for {self.patience} epochs."
            )

    def _is_over_threshold(self, current):
        if self.mode == 'min':
            return current <= self.stopping_threshold
        else:
            return current >= self.stopping_threshold

