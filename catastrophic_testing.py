import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader


class CataForgetter():
    def __init__(self, model: nn.Module, old_data: dataloader, new_data: dataloader, classification: bool) -> None:
        self.model = model
        self.old_data = old_data
        self.new_data = new_data
        self.classification = classification

    # get the CF for each data point
    def get_CF(self) -> list:
        self.old_label_on_old_data = []
        self.new_label_on_old_data = []

        # train the model on the old data
        print("Training the model on the old data")
        self.model.train(self.old_data)
        print("Computing the probability of the true class on the old data")
        with torch.no_grad():
            for i, data in enumerate(self.old_data):
                inputs, labels = data
                # get the probability of the true class
                batch_prob_list = self.model.get_probability_for_true_class(inputs, labels)
                for prob in batch_prob_list:
                    self.old_label_on_old_data.append(prob)
        
        # train the model on the new data
        print("Training the model on the new data")
        self.model.train(self.new_data)
        print("Computing the probability of the true class on the old data")
        with torch.no_grad():
            for i, data in enumerate(self.old_data):
                inputs, labels = data
                # get the probability of the true class
                batch_prob_list = self.model.get_probability_for_true_class(inputs, labels)
                for prob in batch_prob_list:
                    self.new_label_on_old_data.append(prob)
        
        # calculate the CF
        self.CF = []
        for i in range(len(self.old_label_on_old_data)):
            if self.classification == True:
                self.CF.append(self.new_label_on_old_data[i] - self.old_label_on_old_data[i])
            else:
                return "Regression is not supported"
        return self.CF



    