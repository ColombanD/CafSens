import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader


class CataForgetter():
    def __init__(self, model: nn.Module, old_data: dataloader, new_data: dataloader) -> None:
        self.model = model
        self.old_data = old_data
        self.new_data = new_data

    # get the CF for each data point
    def get_CF(self) -> list:
        self.old_label = []
        self.new_label = []

        # train the model on the old data
        self.model.train(self.old_data)
        with torch.no_grad():
            for i, data in enumerate(self.old_data):
                inputs, labels = data
                # get the probability of the true class
                self.old_label.append(self.model.get_probability_for_true_class(inputs, labels))
        
        # train the model on the new data
        self.model.train(self.new_data)
        with torch.no_grad():
            for i, data in enumerate(self.new_data):
                inputs, labels = data
                # get the probability of the true class
                self.new_label.append(self.model.get_probability_for_true_class(inputs, labels))
        
        # calculate the CF
        self.CF = []
        for i in range(len(self.old_label)):
            self.CF.append(self.new_label[i] / self.old_label[i])
        
        return self.CF



    