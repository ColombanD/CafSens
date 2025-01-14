"""
Description: 
This file contains the function to load the datasets.
"""

from torchvision import datasets, transforms
from utils.classiloader import ClassiLoader
import ast
    
def load_datasets(dataset_names, split_indices: str, transform):
    "returns (list of datasets, grayscale)"

    list_datasets = []
    grayscale = False
    num_classes = 10
    for dataset_name in dataset_names:
        if dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        if dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
            num_classes = 100
        elif dataset_name == 'MNIST':
            dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
            grayscale = True
        elif dataset_name == 'FashionMNIST':
            dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
            grayscale = True
        list_datasets.append(dataset)
    
    if split_indices is not None:
        if len(list_datasets) != 1:
            raise ValueError("split_indices can only be used when only one dataset is given")
        split_indices = expand_ranges(split_indices)
        split_indices = [ast.literal_eval(indices) for indices in split_indices]
        classi = ClassiLoader(list_datasets[0])
        D = classi.split(split_indices)
        return D, grayscale, num_classes
    
    else:
        if len(list_datasets) == 1:
            raise ValueError("split_indices must be used when only one dataset is given")
        return list_datasets, grayscale, num_classes

def expand_ranges(input_list):
    """
    Expand ranges in the form of strings to lists of numbers.
    
    Args:
        input_list (list): A list of strings representing ranges, e.g., ["[0,9]", "[10,19]"].
    
    Returns:
        list: A list of strings where each range is expanded, e.g., ["[0, 1, ..., 9]"].
    """
    expanded_list = []
    
    for range_str in input_list:
        # Convert the string range to a Python list (e.g., "[0,9]" -> [0, 9])
        range_values = ast.literal_eval(range_str)
        
        # Generate the list of numbers for the range
        expanded_range = list(range(range_values[0], range_values[1] + 1))
        
        # Convert the expanded range back to a string representation
        expanded_list.append(str(expanded_range))
    
    return expanded_list