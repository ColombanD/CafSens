import numpy as np
import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset, DataLoader


class ClassiLoader():
    def __init__(self, dataset: torchvision.datasets):
        self.trainset = dataset
        self.classes = np.unique(self.trainset.targets)
        self.split_by_classes = self._split_by_classes()

    def _split_by_classes(self):
      """
        Split the dataset into separate subsets for each class and return a list of subsets.
      """
      indices = np.arange(len(self.trainset))
      split_by_class = []

      for classe in self.classes:
        class_bool = (self.trainset.targets == classe)
        class_indices = indices[class_bool]
        split_by_class.append(Subset(self.trainset, class_indices))

      return split_by_class
    
    def split(self, liste):
      """
        Split the dataset according to the given 'liste' of classes.
        Each element in 'liste' contains indices representing the desired classes.
        Returns a list of DataLoaders, one for each group of classes.
      """
      D = []

      for classes in liste:
        combined_subsets = [self.split_by_classes[i] for i in classes]
        combined_datasets = ConcatDataset(combined_subsets)
        dataloader = DataLoader(combined_datasets, batch_size=4, shuffle=True, num_workers=2)
        D.append(dataloader)

      return D