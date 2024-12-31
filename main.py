import yaml
import argparse
import logging
from logging.handlers import RotatingFileHandler

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets, models

from utils.caf import Caf
from utils.Sensi import Sensitivity
from utils.plotting import plot
from models.CNN import CNN
from models.Transformer import Transformer
from scipy.stats import pearsonr, spearmanr, kendalltau


# Function to load the configuration file
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to parse the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run the whole CAFSENS pipeline')
    choices_model = ['CNN', 'Transformer', 'AlexNet', 'Resnet18']
    choices_dataset = ['MNIST', 'FashionMNIST', 'CIFAR10']
    parser.add_argument('--model', default='CNN', type=str, choices=choices_model, help='model to use')
    parser.add_argument('--old-dataset', default='MNIST', type=str, choices=choices_dataset, help='old dataset to use')
    parser.add_argument('--new-dataset', default='FashionMNIST', type=str, choices=choices_dataset, help='new dataset to use')
    parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
    args = parser.parse_args()
    return args

# Function to setup logging
def setup_logging(logging_path="logs/logs.log", logging_info="info"):
    logger = logging.getLogger()
    logger.setLevel(logging_info)

    # Create a file handler
    file_handler = RotatingFileHandler(logging_path, maxBytes=10000, backupCount=1)
    file_handler.setLevel(logging_info)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_info)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_model(model_name, gray_scale):
    if model_name == 'CNN':
        return CNN(gray_scale=gray_scale)
    elif model_name == 'Transformer':
        return Transformer(grayscale=gray_scale)
    elif model_name == 'AlexNet':
        return models.alexnet(pretrained=False)
    elif model_name == 'Resnet18':
        return models.rensnet18(pretrained=False)

def load_dataset(dataset_name):
    "returns (dataset, greyscale)"

    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'MNIST':
        return datasets.MNIST(root="./data", train=True, download=True, transform=transform), True
    elif dataset_name == 'FashionMNIST':
        return datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform), True
    elif dataset_name == 'CIFAR10':
        return datasets.CIFAR10(root="./data", train=True, download=True, transform=transform), False
    

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
    old_dataset, gray_scale = load_dataset(args.old_dataset)
    new_dataset, _ = load_dataset(args.new_dataset)

    # Load the model
    model = load_model(args.model, gray_scale=gray_scale)

    # Split the datasets into train/test parts
    old_train_size = int(config['dataset']['old_train_ratio'] * len(old_dataset))
    old_test_size = len(old_dataset) - old_train_size
    new_train_size = int(config['dataset']['new_train_ratio'] * len(new_dataset))
    new_test_size = len(new_dataset) - new_train_size
    old_train_dataset, old_test_dataset = random_split(old_dataset, [old_train_size, old_test_size])
    new_train_dataset, new_test_dataset = random_split(new_dataset, [new_train_size, new_test_size])

    # Initialize the loaders
    train_old_loader = DataLoader(old_train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_old_loader = DataLoader(old_test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    train_new_loader = DataLoader(new_train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_new_loader = DataLoader(new_test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Initialize the CAF experiment
    caf = Caf(model, train_old_loader, test_old_loader, train_new_loader, test_new_loader, device=device, logger=logger)
    
    # 1. Train on old
    logger.info("Training on old dataset...")
    caf.train(epochs=config['training']['epochs_old'], lr=config['training']['learning_rate'], train_old=True)
    logger.info("Training on old dataset completed.")
    torch.save(model.state_dict(), f"{args.model}_old.pth")

    # 2. Test on old
    logger.info("Testing on old dataset...")
    acc_old_before = caf.test(test_old=True)
    logger.info(f"Accuracy on old dataset before training on new: {acc_old_before}")
    caf.history["test_old_acc_before"] = acc_old_before

    # 3. Get old_true_logits
    old_true_logits_train = caf.get_true_probs(train=True)
    old_true_logits_test = caf.get_true_probs(train=False)

    # 4. Train on new
    logger.info("Training on new dataset...")
    caf.train(epochs=config['training']['epochs_new'], lr=config['training']['learning_rate'], train_old=False)
    logger.info("Training on new dataset completed.")
    torch.save(model.state_dict(), f"{args.model}_new.pth")

    # 5. Test on new
    logger.info("Testing on new dataset...")
    acc_new = caf.test(test_old=False)
    logger.info(f"Accuracy on new dataset: {acc_new}")
    caf.history["test_new_acc"] = acc_new

    # 6. Test on old again to see if forgetting occurred
    logger.info("Testing on old dataset after training on new...")
    acc_old_after = caf.test(test_old=True)
    logger.info(f"Accuracy on old dataset after training on new: {acc_old_after}")
    caf.history["test_old_acc_after"] = acc_old_after

    # 7. Get new_true_logits
    new_true_logits_train = caf.get_true_probs(train=True)
    new_true_logits_test = caf.get_true_probs(train=False)

    # Get CAF score
    caf_score_train = caf.get_caf(old_true_probs=old_true_logits_train, new_true_probs=new_true_logits_train)
    logger.info(f"CAF train score mean: {caf_score_train.mean()}, CAF train score std: {caf_score_train.std()}")
    caf_score_test = caf.get_caf(old_true_probs=old_true_logits_test, new_true_probs=new_true_logits_test)
    logger.info(f"CAF test score mean: {caf_score_test.mean()}, CAF test score std: {caf_score_test.std()}")

    # Compute sensitivity
    sensitivity_train = Sensitivity(model=model, dataloader=train_old_loader, device=device)
    sensitivity_test = Sensitivity(model=model, dataloader=test_old_loader, device=device)
    sensitivities_train = sensitivity_train.get_sensitivities()
    sensitivities_test = sensitivity_test.get_sensitivities()
    logger.info(f"Sensitivity mean: {sensitivities_train.mean()}, Sensitivity std: {sensitivities_train.std()}")
    logger.info(f"Sensitivity mean: {sensitivities_test.mean()}, Sensitivity std: {sensitivities_test.std()}")

    # Plot the results
    plot(sensitivity=sensitivities_train, caf=caf_score_train, saving_path=f"/results/{args.model}_{args.old_dataset}_{args.new_dataset}_train.png")
    plot(sensitivity=sensitivities_test, caf=caf_score_test, saving_path=f"/results/{args.model}_{args.old_dataset}_{args.new_dataset}.png")
    
    # Compute some summary statistics (correlation)
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
    logger.info(f"Kendall correlation during testing between CF an Sensitivity: {tau_test:.4f}, (p value {pval_kend_test:4e})")

    


if "__main__" == __name__:
    main()