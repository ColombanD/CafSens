import torch
import torch.nn as nn

class Classifier():
    def __init__(self, model : nn.Module) -> None:
        self.model = model

    # Based on the input x and the true label y, return the probability the model assigns to the true class
    def get_probability_for_true_class(self, x, y):
        # Softmax the output of the model to get the probabilities
        prob_list = torch.softmax(self.model.forward(x), dim=1)
        prob_for_true_class = []
        # Since we have a batch of inputs, we need to get the probability of the true class for each input of the batch
        for i in range(len(prob_list)):
            prob_for_true_class.append(prob_list[i][y[i]])
        return prob_for_true_class
    