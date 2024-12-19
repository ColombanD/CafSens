import matplotlib.pyplot as plt
import torch
import numpy as np

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


def plot(sensitivity, caf, saving_path: str):
    sensitivity = tensor_to_numpy(sensitivity)
    caf = tensor_to_numpy(caf)
    print(sensitivity.shape)
    print(caf.shape)
    # Analyze relation between D_cf and D_sens
    plt.figure(figsize=(8,6))
    plt.scatter(sensitivity, caf, alpha=0.5)
    plt.xlabel('Catastrophic Forgetting (D_cf)')
    plt.ylabel('Sensitivity (D_sens)')
    plt.title('D_cf vs D_sens')
    plt.grid(True)
    plt.xscale('log')  # Log scale for x-axis
    plt.yscale('log')  # Log scale for y-axis
    plt.show()
    plt.savefig(saving_path)