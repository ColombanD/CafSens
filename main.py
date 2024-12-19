import yaml
import argparse
import logging
from logging.handlers import RotatingFileHandler

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets

from utils.caf import Caf
from utils.Sensi import Sensitivity
from utils.plotting import plot
from models.CNN import CNN
from models.Transformer import Transformer

# Function to load the configuration file
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to parse the command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run the whole CAFSENS pipeline')
    choices_model = ['CNN', 'Transformer']
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
    old_true_logits = caf.get_true_probs()

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
    new_true_logits = caf.get_true_probs()

    # Get CAF score
    caf_score = caf.get_caf(old_true_probs=old_true_logits, new_true_probs=new_true_logits)
    logger.info(f"CAF score mean: {caf_score.mean()}, CAF score std: {caf_score.std()}")

    # Compute sensitivity
    sensitivity = Sensitivity(model=model, dataloader=train_old_loader, device=device)
    sensitivities = sensitivity.get_sensitivities()
    logger.info(f"Sensitivity mean: {sensitivities.mean()}, Sensitivity std: {sensitivities.std()}")

    # Plot the results
    plot(sensitivity=sensitivities, caf=caf_score, saving_path=f"/results/{args.model}_{args.old_dataset}_{args.new_dataset}.png")


if "__main__" == __name__:
    main()