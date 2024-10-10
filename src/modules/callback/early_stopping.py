from pytorch_lightning.callbacks import EarlyStopping, Callback
import torch
from math import isnan


class EarlyStoppingCallback(Callback):
    def __init__(self, monitor: str, patience: int = 3, mode: str = 'min', min_delta: float = 0.0, verbose=False):
        super().__init__()

        stopping_threshold = min_delta

        if mode in ['min', 'max']:
            self.early_stopping = EarlyStopping(
                monitor=monitor, patience=patience, mode=mode, min_delta=min_delta, verbose=verbose
            )
        else:
            self.monitor = monitor
            self.patience = patience
            self.mode = mode
            self.stopping_threshold = stopping_threshold
            self.wait_count = 0
            self.stopped_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        if hasattr(self, 'early_stopping'):
            return self.early_stopping.on_validation_end(trainer, pl_module)

        current = trainer.callback_metrics.get(self.monitor)

        if current is None:
            return

        if isnan(current):
            trainer.should_stop = True
            print(f"Early stopping: {self.monitor} is NaN.")
            return

        if self._is_over_threshold(current):
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
                # print(
                #     f"Early stopping: {self.monitor} has reached the threshold of {self.stopping_threshold} for {self.patience} epochs."
                # )
                # todo: add logging
        else:
            self.wait_count = 0

    def _is_over_threshold(self, current):
        return current <= self.stopping_threshold