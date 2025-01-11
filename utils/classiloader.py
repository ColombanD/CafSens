"""
Description:
This script facilitates custom grouping of classes for splitting datasets. 
"""
import numpy as np
import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset, DataLoader


class ClassiLoader():
    """
    A utility class for splitting datasets by class labels.

    Attributes:
        dataset (torchvision.datasets): The input dataset to be split.
        classes (np.ndarray): Array of unique class labels in the dataset.
        split_by_classes (list): List of subsets of the dataset, each corresponding to a single class.
    """
    def __init__(self, dataset: torchvision.datasets):
        self.dataset = dataset
        self.classes = np.unique(self.dataset.targets)
        self.split_by_classes = self._split_by_classes()


    def _split_by_classes(self) -> list:
      """
        Split the dataset into separate subsets for each class and return a list of subsets.

        Return:
            split_by_class (list): A list of subsets, each corresponding to one specific class.
      """
      indices = np.arange(len(self.dataset))
      split_by_class = []

      for classe in self.classes:
        class_bool = (self.dataset.targets == classe)
        class_indices = indices[class_bool]
        split_by_class.append(Subset(self.dataset, class_indices))

      return split_by_class


    def split(self, class_groups: list) -> list:
      """
        Split the dataset according to the given 'class_groups' of classes.

        Args:
            class_groups (list): Each element in 'class_groups' contains indices representing the desired grouping of classes.

        Returns:
            D (list): A list of ConcatDatasets, each corresponding to a 
                  group of classes as defined in 'class_groups'.
      """
      D = []

      for classes in class_groups:
        combined_subsets = [self.split_by_classes[i] for i in classes]
        combined_datasets = ConcatDataset(combined_subsets)
        D.append(combined_datasets)

      return D
