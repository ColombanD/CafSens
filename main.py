"""Description:
Pipeline to run the CAFSENS experiment
"""

import yaml
import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

import torch
from torch.utils.data import random_split, DataLoader

from utils.caf import Caf
from utils.sensitivity import Sensitivity
from utils.plotting import plot
from utils.datasets import load_datasets
from utils.models import load_model
from utils.transforms import load_transform


# Function to load the configuration file
def load_config(config_path='./configs/default_config.yaml'):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading the configuration file {config_path}: {e}")
        sys.exit(1)

# Function to parse the command-line arguments
# eg: python main.py --model CNN --datasets Cifar10 --split-indices "[0,2]" "[3,5]" "[6,9]"
def parse_args():
    parser = argparse.ArgumentParser(description='Run the whole CAFSENS pipeline')
    choices_model = ['CNN', 'Transformer', 'AlexNet', 'Resnet18', 'Resnet50']
    choices_dataset = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--model', default='CNN', type=str, choices=choices_model, help='model to use')
    parser.add_argument('--split-indices', default=None, nargs="+", type=str, help='If only one dataset was given, list of indices representing the classes to split the dataset into, e.g. "[0,2]" "[3,5]" "[6,9]"')
    parser.add_argument('--datasets', nargs="+", type=str, choices=choices_dataset, help='datasets to use, can be multiple')
    parser.add_argument('--exp-tag', type=str, default="Last Experiment", help="Experiment tag.")
    args = parser.parse_args()
    return args

# Function to setup logging
def setup_logging(logging_path="./logs", logging_info="info"):
    logger = logging.getLogger()
    logger.setLevel(logging_info)

    os.makedirs(os.path.dirname(logging_path), exist_ok=True)
    # Delete the log file if it exists
    if os.path.exists(logging_path):
        os.remove(logging_path)

    # Create a file handler
    file_handler = RotatingFileHandler(logging_path, maxBytes=10000, backupCount=1)
    file_handler.setLevel(logging_info)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

# Main function
def main():

    # Parse the command-line arguments
    args = parse_args()

    # Load the configuration file
    if args.config is not None:
        config = load_config(args.config)
        # Apply config to arguments if a config file was given
        for key, value in config.items():
            if key in vars(args): # Only use the keys that are in the arguments
                setattr(args, key, value)
    else:
        config = load_config()

    # Setup logging
    logger = setup_logging(config['logging_path'], config['logging_level'])

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    
     # Determine which transform to use for the pair (model, dataset)
    transform = load_transform(args.model)

    # Load the datasets
    datasets_list, grayscale, num_classes = load_datasets(args.datasets, args.split_indices, transform=transform)

    # Load the model
    model = load_model(args.model, gray_scale=grayscale, num_classes=num_classes)

    # Split the datasets into train/test parts
    train_list = []
    test_list = []
    for dataset in datasets_list:
        train_size = int(config['train_ratio'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_list.append(train_dataset)
        test_list.append(test_dataset)

    # Initialize the loaders
    train_loaders = [DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True) for train_dataset in train_list]
    test_loaders = [DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False) for test_dataset in test_list]

    # Initialize the CAF and Sensi experiment
    caf = Caf(model, list_train_loaders=train_loaders, list_test_loaders=test_loaders, device=device, logger=logger)
    acc = []
    true_train_logits = []
    true_test_logits = []


    # Train/Test the model and retrieve the logits for Caf analysis
    # For naming in printing and saving, we assume that the datasets are given in the same order as the train_loaders
    if len(args.datasets) == 1:
        args.datasets = args.datasets * len(train_loaders)
    for i, train_loader in enumerate(train_loaders):

        # For logging the names of the datasets
        dataset_name = args.datasets[i]
        if args.split_indices is not None:
            classes_name = args.split_indices[i].strip("[]").split(",")
            dataset_name = f"{dataset_name}, classes {classes_name[0]} to {classes_name[1]}"

        # Train on dataset i
        logger.info(f"Training on dataset {dataset_name} ...")
        caf.train(epochs=config['epochs'], lr=config['learning_rate'], train_nbr=i)
        logger.info(f"Training completed.")

        saving_path = f"./models_weights/{args.model}_{dataset_name}.pth"
        os.makedirs(os.path.dirname(saving_path), exist_ok=True) 
        torch.save(model.state_dict(), saving_path)

        # Test on dataset i
        logger.info(f"Testing on dataset {dataset_name}...")
        acc.append(caf.test(test_nbr=i))
        logger.info(f"Accuracy on dataset {dataset_name}: {acc[-1]}")
        caf.history[f"test_acc_{dataset_name}"] = acc[-1]

        # Get the true logits for dataset 0 in j-th training
        true_test_logits.append(caf.get_true_probs(train=False, dataset_nbr=0))

        if i == 0:
            # Compute the sensibility of the 1st dataset
            logger.info(f"Computing sensitivities before training on other tasks for dataset {dataset_name} ...")
            sensey_test = Sensitivity(model=model, dataloader=test_loaders[0], device=device)
            sensi_test = sensey_test.get_sensitivities()

    # Compute the sensibility of dataset 0 after training on all datasets
    logger.info(f"Computing sensitivities after training for dataset {args.datasets[0]} ...")
    post_sensi_test = Sensitivity(model=model, dataloader=test_loaders[0], device=device)
    post_sensi_test = post_sensi_test.get_sensitivities()
    
        
    # Get CAF scores for the dataset 0 in each training
    caf_scores_test_ratio = []
    caf_scores_test_diff = []
    logger.info(f'Computing CAF score...')
    for i in range(len(train_loaders) - 1):
        caf_scores_test_ratio.append(caf.get_caf(old_true_probs=true_test_logits[i], new_true_probs=true_test_logits[i+1], caf_type='ratio'))
        caf_scores_test_diff.append(caf.get_caf(old_true_probs=true_test_logits[i], new_true_probs=true_test_logits[i+1], caf_type='difference'))


    # Plot the results
    result_directory = f'./results/{args.exp_tag}'
    os.makedirs(result_directory, exist_ok=True)
    test_name_0 = args.datasets[0]
    if args.split_indices is not None:
        classes_name = args.split_indices[0].strip("[]").split(",")
        test_name_0 = f"{test_name_0}_{classes_name[0]}_{classes_name[1]}"

    for i in range(len(caf_scores_test_ratio)):        
        test_name_i = args.datasets[i+1]
        if args.split_indices is not None:
            classes_name_i = args.split_indices[i].strip("[]").split(",")
            test_name_i = f"{test_name_i}_{classes_name_i[0]}_{classes_name_i[1]}"

        plot_path_test_ratio = os.path.join(result_directory, f"test_{test_name_0}_{test_name_i}_pre_ratio")
        plot_path_test_post_ratio = os.path.join(result_directory, f"test_{test_name_0}_{test_name_i}_post_ratio")

        plot_path_test_diff = os.path.join(result_directory, f"test_{test_name_0}_{test_name_i}_pre_diff")
        plot_path_test_post_diff = os.path.join(result_directory, f"test_{test_name_0}_{test_name_i}_post_diff")

        plot(sensitivity=sensi_test, caf=caf_scores_test_ratio[i], saving_path=plot_path_test_ratio, title=f"Test Dataset {test_name_0} after training on {test_name_i} with sensitivity computed just after training on {test_name_0} and Caf measured with the ratio of logits")
        plot(sensitivity=post_sensi_test, caf=caf_scores_test_ratio[i], saving_path=plot_path_test_post_ratio, title=f"Test Dataset {test_name_0} after training on {test_name_i} with sensitivity computed after training on all datasets and Caf measured with the ratio of logits")

        plot(sensitivity=sensi_test, caf=caf_scores_test_diff[i], saving_path=plot_path_test_diff, title=f"Test Dataset {test_name_0} after training on {test_name_i} with sensitivity computed just after training on {test_name_0} and Caf measured with the difference of logits")
        plot(sensitivity=post_sensi_test, caf=caf_scores_test_diff[i], saving_path=plot_path_test_post_diff, title=f"Test Dataset {test_name_0} after training on {test_name_i} with sensitivity computed after training on all datasets and Caf measured with the difference of logits")

if "__main__" == __name__:
    main()