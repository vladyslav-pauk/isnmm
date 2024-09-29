import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluate(F, dataloader, mixture, x):
    # Check if F is an empty list or tensor
    if isinstance(F, list) and len(F) == 0:
        raise ValueError("The list F is empty.")

    if isinstance(F, torch.Tensor) and F.numel() == 0:
        raise ValueError("The tensor F is empty.")

    # Ensure that F now has the same number of samples as mixture
    if F.shape[0] != mixture.shape[0]:
        raise ValueError(f"Shape mismatch: mixture has {mixture.shape[0]} samples, but F has {F.shape[0]} samples.")

    # Convert tensors to numpy arrays for plotting
    F = F.cpu().detach().numpy()
    mixture = mixture.cpu().detach().numpy()
    x = x.cpu().detach().numpy()

    # Plot the results
    plt.figure(figsize=(10, 5))  # Adjust size as needed
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
    x = x - np.min(x)  # Min normalization
    x = x / np.max(x) * bound  # Scale to the range [0, bound]
    return x