import numpy as np
import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset, DataLoader

class ClassiLoader():
    def __init__(self, dataset: torchvision.datasets):
        self.trainset = dataset
    
    def split(self, liste):
      """
        Split the dataset according to the given 'liste' of classes.
        Each element in 'liste' contains indices representing the desired classes.
        Returns a list of DataLoaders, one for each group of classes.
      """
      D = []
      indices = np.arange(len(self.trainset))

      for classes in liste:
        combined_subsets = []
        
        for classe in classes:
          if isinstance(classe, str):
            class_bool = (self.trainset.targets == classe)
          elif isinstance(classe, int):
            class_bool = (self.trainset.targets == np.int_(classe))
          else:
            raise ValueError(f"Invalid class type. for \"{classe}\" Expected str or int.")

          if np.sum(class_bool) == 0:
            raise ValueError(f"Class \"{classe}\" not found in the dataset.")

          class_indices = indices[class_bool]
          subset = Subset(self.trainset, class_indices)
          combined_subsets.append(Subset(self.trainset, class_indices))
        
        combined_datasets = ConcatDataset(combined_subsets)
        dataloader = DataLoader(combined_datasets, batch_size=4, shuffle=True, num_workers=2)
        D.append(combined_datasets)

      return D