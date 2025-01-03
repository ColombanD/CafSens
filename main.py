import yaml
import argparse
import logging
from logging.handlers import RotatingFileHandler
import ast

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets, models

from utils.caf import Caf
from utils.Sensi import Sensitivity
from utils.plotting import plot
from utils.ClassiLoader import ClassiLoader
from models.CNN import CNN
from models.Transformer import Transformer
from scipy.stats import pearsonr, spearmanr, kendalltau


# Function to load the configuration file
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to parse the command-line arguments
# eg: python main.py --model CNN --datasets Cifar10 --split-info "[[0, 1, 2], [3, 4, 5]]"
def parse_args():
    parser = argparse.ArgumentParser(description='Run the whole CAFSENS pipeline')
    choices_model = ['CNN', 'Transformer', 'AlexNet', 'Resnet18']
    choices_dataset = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
    parser.add_argument('--model', default='CNN', type=str, choices=choices_model, help='model to use')
    parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
    parser.add_argument('--split-indices', default=None, nargs="+", type=str, help='If only one dataset was given, list of indices representing the classes to split the dataset into, e.g. "[0, 1, 2]" "[3, 4, 5]" "[6, 7, 8, 9]"')
    parser.add_argument('--datasets', nargs="+", type=str, choices=choices_dataset, help='datasets to use, can be multiple')
    args = parser.parse_args()
    return args

# Function to setup logging
def setup_logging(logging_path="logs/logs.log", logging_info="info"):
    logger = logging.getLogger()
    logger.setLevel(logging_info)

    # Create a file handler
    file_handler = RotatingFileHandler(logging_path, maxBytes=10000, backupCount=1)
    file_handler.setLevel(logging_info)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

def load_model(model_name, gray_scale, num_classes):
    if model_name == 'CNN':
        return CNN(gray_scale=gray_scale, num_classes=num_classes)
    elif model_name == 'Transformer':
        return Transformer(grayscale=gray_scale)
    elif model_name == 'AlexNet':
        return models.alexnet(pretrained=False, num_classes=num_classes)
    elif model_name == 'Resnet18':
        return models.rensnet18(pretrained=False, num_classes=num_classes)
    
def load_datasets(dataset_names, split_indices: str):
    "returns (list of datasets, grayscale)"
    transform = transforms.Compose([
        #for compatibility between datasets and models
        transforms.Resize((28,28)),
        transforms.ToTensor()
        ])
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
        split_indices = [ast.literal_eval(indices) for indices in split_indices]
        classi = ClassiLoader(list_datasets[0])
        D = classi.split(split_indices)
        return D, grayscale, num_classes
    
    else:
        if len(list_datasets) == 1:
            raise ValueError("split_indices must be used when only one dataset is given")
        return list_datasets, grayscale, num_classes


# Main function
def main():

    # Parse the command-line arguments
    args = parse_args()

    # Load the configuration file
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(config['logging']['path'], config['logging']['level'])

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device}")
    
    # Load the datasets
    datasets_list, grayscale, num_classes = load_datasets(args.datasets, args.split_indices)

    # Load the model
    model = load_model(args.model, gray_scale=grayscale, num_classes=num_classes)

    # Split the datasets into train/test parts
    train_list = []
    test_list = []
    for dataset in datasets_list:
        train_size = int(config['dataset']['train_ratio'] * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_list.append(train_dataset)
        test_list.append(test_dataset)

    # Initialize the loaders
    train_loaders = [DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True) for train_dataset in train_list]
    test_loaders = [DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False) for test_dataset in test_list]

    # Initialize the CAF experiment
    caf = Caf(model, list_train_loaders=train_loaders, list_test_loaders=test_loaders, device=device, logger=logger)
    acc = []
    true_train_logits = []
    true_test_logits = []


    # Train/Test the model and retrieve the logits for Caf analysis
    # For naming in printing and saving, we assume that the datasets are given in the same order as the train_loaders
    if len(args.datasets) == 1:
        args.datasets = args.datasets * len(train_loaders)
    for i, train_loader in enumerate(train_loaders):

        # Train on dataset i
        logger.info(f"Training {i} on dataset {args.datasets[i]} ...")
        caf.train(epochs=config['training']['epochs'], lr=config['training']['learning_rate'], train_nbr=i)
        logger.info(f"Training {i} on dataset {args.datasets[i]} completed.")
        torch.save(model.state_dict(), f"./models/{args.model}_{i}_train.pth")

        # Test on dataset i
        logger.info(f"Testing {i} on dataset {args.datasets[i]}...")
        acc.append(caf.test(test_nbr=i))
        logger.info(f"Accuracy {i} on dataset {args.datasets[i]}: {acc[-1]}")
        caf.history[f"test_acc_{i}"] = acc[-1]

        # Get the true logits for each dataset up to the i-th dataset
        # Each element in true_train_logits and true_test_logits is a list of all logits for each dataset up to the i-th dataset
        true_train_logits_i = []
        true_test_logits_i = []
        for j in range(i+1):
            true_train_logits_i.append(caf.get_true_probs(train=True, dataset_nbr=j))
            true_test_logits_i.append(caf.get_true_probs(train=False, dataset_nbr=j))

        true_train_logits.append(true_train_logits_i)
        true_test_logits.append(true_test_logits_i)
        
    # Get CAF scores for each dataset: [i][j] corresponds to the CAF score for dataset i, comparing state at training j with state at training j+1
    caf_scores_train = []
    caf_scores_test = []
    logger.info(f'Computing CAF score...')
    for i in range(len(train_loaders) - 1):
        caf_score_train_i = []
        caf_score_test_i = []
        for j in range(i, len(train_loaders) - 1):
            caf_score_train_i.append(caf.get_caf(old_true_probs=true_train_logits[j][i], new_true_probs=true_train_logits[j+1][i]))
            caf_score_test_i.append(caf.get_caf(old_true_probs=true_test_logits[j][i], new_true_probs=true_test_logits[j+1][i]))
        caf_scores_train.append(caf_score_train_i)
        caf_scores_test.append(caf_score_test_i)

    # Compute sensitivity: one for each dataset
    sensi_train = []
    sensi_test = []
    logger.info(f"Computing sensitivities...")
    for i in range(len(train_loaders) - 1):
        sensey_train = Sensitivity(model=model, dataloader=train_loaders[i], device=device)
        sensey_test = Sensitivity(model=model, dataloader=test_loaders[i], device=device)
        sensi_train.append(sensey_train.get_sensitivities())
        sensi_test.append(sensey_test.get_sensitivities())

    # Plot the results
    for i in range(len(caf_scores_train)):
        for j in range(len(caf_scores_train[i])):
            plot(sensitivity=sensi_train[i], caf=caf_scores_train[i][j], saving_path=f"./results/{args.model}_{args.datasets}_{i}_{j+1}_train.png", title=f"Train Dataset {i} after training on {j+1}")
            plot(sensitivity=sensi_test[i], caf=caf_scores_test[i][j], saving_path=f"./results/{args.model}_{args.datasets}_{i}_{j+1}_test.png", title=f"Test Dataset {i} after training on {j+1}")
    
    """# Compute some summary statistics (correlation)
    corr_train, p_value_train = pearsonr(caf_score_train, sensitivities_train)
    corr_test, p_value_test = pearsonr(caf_score_test, sensitivities_test)
    logger.info(f"Pearson correlation during training between CF and Sensitivity: {corr_train:.4f} (p-value: {p_value_train:.4e})")
    logger.info(f"Pearson correlation during testing between CF and Sensitivity: {corr_test:.4f} (p-value: {p_value_test:.4e})")

    # Spearman’s rank correlation
    rho_train, pval_spear_train = spearmanr(caf_score_train, sensitivities_train)
    rho_test, pval_spear_test = spearmanr(caf_score_test, sensitivities_test)
    logger.info(f"Spearman correlation during training between CF an Sensitivity: {rho_train:.4f}, (p value {pval_spear_train:4e})")
    logger.info(f"Spearman correlation during testing between CF an Sensitivity: {rho_test:.4f}, (p value {pval_spear_test:4e})")

    # Kendall’s tau
    tau_train, pval_kend_train = kendalltau(caf_score_train, sensitivities_train)
    tau_test, pval_kend_test = kendalltau(caf_score_test, sensitivities_test)
    logger.info(f"Kendall correlation during training between CF an Sensitivity: {tau_train:.4f}, (p value {pval_kend_train:4e})")
    logger.info(f"Kendall correlation during testing between CF an Sensitivity: {tau_test:.4f}, (p value {pval_kend_test:4e})")"""




if "__main__" == __name__:
    main()