"""
Description:
This script estimates the sensitivity of a model to the inputs of a dataset, 
based on the residuals, prediction variance (via variational inference), and 
the softmax Hessian diagonal approximation. It follows the formula:
    Sensitivity = Lambda * Variance * Residual (Eq. 12 from [1]).

[1] Reference:
    - Paper: "The Memory Perturbation Equation: Understanding Model's Sensitivity to Data"
      (https://arxiv.org/abs/2310.19273)
    - Github Repository: https://github.com/team-approx-bayes/memory-perturbation
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from laplace import Laplace
from laplace.curvature import AsdlGGN


class Sensitivity:
    """
    Computes model sensitivity to inputs using variational inference 
    and softmax Hessian approximation.

    Attributes:
        model (torch.nn.Module): Model to compute the sensitivity on.
        dataloader (DataLoader): Dataloader providing the dataset for sensitivity computation.
        device (str): The device to perform computations on ('cuda' or 'cpu').
    """
    def __init__(self, model: torch.nn.Module, dataloader: DataLoader, device: str = 'cuda'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device


    def compute_variance_infer(self) -> np.ndarray:
        """
        Estimates prediction variance using variational inference with Laplace approximation.

        Returns:
            np.ndarray: Prediction variance for all data points in the dataloader.
                        Shape: (num_samples, num_classes)
        """
        # Initialize Laplace approximation
        laplace_object = Laplace(
            self.model, 'classification',
            subset_of_weights='last_layer',  # Focus on last-layer weights
            hessian_structure='diag',       # Diagonal Hessian approximation
            backend=AsdlGGN
        )

        # Fit Laplace approximation using the training dataloader
        laplace_object.fit(self.dataloader)

        # Collect variance estimates for each input
        variances = []
        for inputs, _ in self.dataloader:
            inputs = inputs.to(self.device)
            _, fvar = laplace_object._glm_predictive_distribution(inputs)
            variances.append(np.diagonal(fvar.cpu().numpy(), axis1=1, axis2=2))

        del laplace_object
        torch.cuda.empty_cache()

        return np.vstack(variances)


    def softmax_hessian(self, probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Computes the diagonal approximation of the Hessian for the softmax function.

        Args:
            probs (torch.Tensor): Predicted probabilities from the model. Shape: (batch_size, num_classes)
            eps (float).

        Returns:
            torch.Tensor: The Hessian diagonal approximation. Shape: (batch_size, num_classes)
        """
        return torch.clamp(probs * (1 - probs), min=eps)


    def get_sensitivities(self) -> np.ndarray:
        """
        Computes sensitivities for the entire dataset based on the residuals, prediction variance (via variational inference), and 
        the Hessian diagonal approximation.

        Returns:
            np.ndarray: Sensitivity values for each input sample. Shape: (num_samples,)
        """
        self.model.eval()  # Set model to evaluation mode

        residuals_list = []
        lambdas_list = []

        for X, y in self.dataloader:
            X, y = X.to(self.device), y.to(self.device)

            # Model predictions
            logits = self.model(X)
            probs = F.softmax(logits, dim=-1)

            # Compute residuals for the true class
            one_hot_y = F.one_hot(y, probs.shape[1]).float()
            residuals = probs - one_hot_y
            residuals_list.append(residuals.cpu().detach().numpy())

            # Compute softmax Hessian diagonal approximation
            lambdas = self.softmax_hessian(probs).cpu().detach().numpy()
            lambdas_list.append(lambdas)

        residuals = np.vstack(residuals_list)
        lambdas = np.vstack(lambdas_list)

        # Compute prediction variance
        variances = self.compute_variance_infer()

        # Compute sensitivity: lambda * variance * residual
        sensitivities = lambdas * variances * residuals

        # Aggregate sensitivities for all classes
        return np.sum(np.abs(sensitivities), axis=-1)