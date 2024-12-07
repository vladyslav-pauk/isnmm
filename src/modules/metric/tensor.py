import torch
import torchmetrics

from src.utils.plot_tools import plot_image, plot_components


class Tensor(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.state_data = {}
        self.tensor = None
        self.tensors = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.state_data:
                self.state_data[key] = []
            self.state_data[key].append(value.clone().detach().cpu())

    def compute(self):
        state_data = {}
        for key, value in self.state_data.items():
            state_data[key] = torch.cat(value, dim=0)

        self.tensors = state_data
        self.state_data.clear()
        return None

    def plot(self, image_dims, show_plot=False, save_plot=False):
        plot_image(
            tensors=self.tensors,
            image_dims=image_dims,
            show_plot=show_plot,
            save_plot=save_plot
        )

        if len(self.tensors) == 2:
            plot_components(
                components=self.tensors,
                labels=None,
                scale=False,
                max_points=300,
                show_plot=show_plot,
                save_plot=save_plot,
                diagonal=True,
                name="Component"
            )

# todo: rewrite using torch add_state interface
