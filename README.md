# CafSens: Catastrophic Forgetting & Sensitivity

Welcome to **CafSens**, a codebase designed to study catastrophic forgetting and model sensitivity in deep learning. This repository provides a modular way to:

- Train models on a sequence of datasets,
- Measure catastrophic forgetting,
- Evaluate model sensitivity.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Directory Structure](#directory-structure)
4. [Installation & Environment](#installation--environment)
5. [Usage](#usage)
6. [Configuration Files](#configuration-files)

## Overview

Catastrophic Forgetting is the phenomenon where a neural network forgets previously learned knowledge upon learning new tasks. This codebase helps you:

- Train a model on a sequence of datasets
- Measure how much it forgets each previously known dataset once trained on a new one,
- Compute model sensitivity via a `Sensitivity` class,
- Correlate forgetting with sensitivity to see if samples that are “more sensitive” are also “forgotten more.”

## Features

- **Config-Driven Workflow**: Use YAML files to configure datasets, models, and hyperparameters.
- **Flexible Dataset Handling**: Supports datasets like MNIST, FashionMNIST, CIFAR10, CIFAR100, and Tiny ImageNet.
- **Custom Model Creation**: Use architectures like ResNet18, AlexNet, VGG16, or any model from `torchvision.models`.

## Directory Structure

```
CafSens/
├── main.py                          # Main logic of project
├── analyze.py                       # Analysis script
├── configs/                         # YAML config files
│   ├── mnist_fashion_resnet18.yaml
│   ├── mnist_fashion_alexnet.yaml
│   ├── cifar10_cifar100_resnet18.yaml
│   ├── cifar10_cifar100_vgg16.yaml
│   └── ...
├── sensitivities
│   ├── test_sensi.py                # Test script to show sensitivity by plotting some images
├── utils/
│   ├── caf.py                       # Catastrophic Forgetting pipeline
│   ├── classiloader.py              # Dataset partitioner for training on sequencial datasets
│   ├── datasets.py                  # Dataset loading
│   ├── models.py                    # Model creation utilities
│   ├── plotting.py                  # Plotting utils
│   ├── sensitivity.py               # Sensitivity computation class
│   ├── transforms.py                # Helper functions (e.g., random seeds)
├── models/                          # Local models
├── model_weights/                   # Saved model weights
├── results
└── README.md                        # This file
```

## Installation & Environment

1. Clone this repository:

   ```bash
   git clone https://github.com/YourUsername/CafSens.git
   cd CafSens
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv_cfsens
   source venv_cfsens/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   
## Usage

### A. Run one of the existing experiments

Run `main.py` with a YAML config:

```bash
python main.py --config configs/mnist_fashionmnist_resnet18.yaml
```

For this example, the code will:

- Train on MNIST (1st dataset),
- Train on FashionMNIST (2nd dataset),
- Compute Sensitivity and Catastrophic Forgetting for MNIST (old dataset)
- plot the results

### B. Run a custom experiment

Run `main.py` with specific arguments:
- `--model`: model name
- `--datasets`: name of the datasets to test. If and only if one is given, must specify the split-indices argument
- `--split-indices`: if only one dataset is given, specify the partition of the dataset into the different classes, e.g: `"[0,2]" "[3,5]" "[6,9]"`
- `--exp-tag`: experiment tag

Examples:

```bash
python main.py --model CNN --datasets MNIST --split-indices "[0,2]" "[3,5]" "[6,9]" --exp-tag Basic_exp
```

```bash
python main.py --model CNN --datasets MNIST FashionMNIST --exp-tag Another_exp
```

Type `python main.py -h` for help.


## Configuration Files

YAML files in `configs/` define experiments. Key fields:

- `datasets`: Dataset names (e.g., MNIST, CIFAR10).
- `model`: Model name (e.g., ResNet18, AlexNet).

### Example (mnist_fashionmnist_alexnet.yaml):

```yaml
# Datasets
train_ratio: 0.8
datasets: ["MNIST", "FashionMNIST"]

# Model
model: "Resnet18"

# Training
batch_size: 64
epochs: 10
learning_rate: 0.001

# Logging
logging_path: ./logs/log
logging_level: INFO

# Results
exp_tag: "MNIST FashionMNIST resnet18 Experiment"
```
