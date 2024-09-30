import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluate(F, dataloader, mixture):

    F = F.cpu().detach().numpy()
    mixture = mixture.cpu().detach().numpy()
    x_list = []
    for batch in dataloader:
        x_batch, _ = batch
        x_list.append(x_batch)

    x = torch.cat(x_list, dim=0).to('cpu').detach().numpy()

    plt.figure(figsize=(10, 5))
    for i in range(mixture.shape[1]):
        plt.subplot(1, mixture.shape[1], i + 1)
        plt.scatter(mixture[:, i], visual_normalization(F[:, i]),
                    label=rf'$\hat{{f}}_{i+1}\circ g_{i+1}$', alpha=0.5, color='blue')
        plt.scatter(mixture[:, i], visual_normalization(x[:, i]), label=rf'$g_{i+1}$', alpha=0.5, color='orange')

        plt.xlabel('input', fontsize=12)
        if i == 0:
            plt.ylabel('output', fontsize=12)

        plt.legend(fontsize=12)

    plt.show()


def visual_normalization(x):
    bound = 10
    x = x - np.min(x)
    x = x / np.max(x) * bound
    return x
