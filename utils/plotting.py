"""Description
This script contains the function to plot the relationship between catastrophic forgetting and sensitivity.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import textwrap

def tensor_to_numpy(data):
    """
    Converts a PyTorch tensor to a NumPy array if necessary.
    
    Args:
        data: The input, which can be a tensor, NumPy array, or other types.
    
    Returns:
        A NumPy array or the original data if no conversion is needed.
    """
    if isinstance(data, torch.Tensor):
        # Move to CPU and detach if required, then convert to NumPy
        return data.cpu().detach().numpy()
    elif isinstance(data, np.ndarray):
        # Already a NumPy array, return as-is
        return data
    else:
        raise TypeError(f"Unsupported type: {type(data)}. Expected a PyTorch tensor or NumPy array.")

def plot(sensitivity, caf, saving_path: str, title: str, plot_type='log'):
    sensitivity = tensor_to_numpy(sensitivity)
    caf = tensor_to_numpy(caf)
    # Analyze relation between D_cf and D_sens
    plt.figure(figsize=(8,6))
    plt.scatter(sensitivity, caf, alpha=0.5)
    plt.ylabel('Catastrophic Forgetting (D_cf)')
    plt.xlabel('Sensitivity (D_sens)')
    plt.title("\n".join(textwrap.wrap(title, 80)))
    plt.grid(True)
    plt.xscale(plot_type)  # Log scale for x-axis
    plt.yscale(plot_type)  # Log scale for y-axis
    plt.savefig(saving_path)
    plt.clf()