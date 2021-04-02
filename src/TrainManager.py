"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This class will provide us everything we need to train correctly 
        our different models
"""

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.data.DataManager import get_final_data

class TrainManager():
    """
        Class used to Manage the training of our different models
    """

    def __init__(self, model, train_set, test_set, batch_size, num_epochs, validation_size, use_cuda):
        """
            Args:
                model: The model that we want to train
                train_set: The data used for the training
                test_set: The data used for the tests
                batch_size: The batch size that we want to use for the training
                num_epochs: the number of epochs for the training
                validation_size: Pourcentage of data to use for validation
                use_cada: If we want to use the gpu for the training
        """
        # If you don't 
        if use_cuda :
            device_name = 'cuda:0'
            if not torch.cuda.is_available():
                print("You are not able to run on a GPU")
                device_name = 'cpu'
        else :
            device_name = 'cpu'

        self.device = torch.device(device_name)
        self.train_set, self.val_set, self.test_set = get_final_data(train_set, test_set, validation_size, batch_size)
        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def train(self):
        """
            Train the models
        """
        for epoch in range(self.num_epochs):
            print("Epoch: " + str(epoch + 1) + " of " + str(self.num_epochs))
            train_loss = 0.0

            with tqdm(range(len(self.train_set))) as t:
                train_losses = []
                train_accuracies = []

                for i, data in enumerate(self.train_set, 0):
                    X, labels = data[0].to(self.device), data[1].to(self.device)

                    outputs = self.model(X)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()

                    train_losses.append(loss.item())
                    train_accuracies.append(self.accuracy(outputs, labels))

                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i+1)))
                    t.update()

            self.train_loss.append(np.mean(train_losses))
            self.train_accuracy.append(np.mean(train_accuracies))
            self.evaluate_validation()
            
        print("Finished training")

    def evaluate_validation(self):
        """
            This method will, at every epoch, evaluate the results on the validation set
        """
        self.model.eval()

        val_loss = 0.0
        val_losses = []
        val_accuracies = []

        with torch.no_grad():
            for data in self.val_set:
                X, labels = data[0].to(self.device), data[1].to(self.device)

                outputs = self.model(X)
                loss = self.loss_function(outputs, labels)
                val_losses.append(loss.item())
                val_accuracies.append(self.accuracy(outputs, labels))
                val_loss += loss.item()

        self.val_loss.append(np.mean(val_losses))
        self.val_accuracy.append(np.mean(val_accuracies))

        print('Validation loss %.3f' % (val_loss / len(self.val_set)))

        self.model.train()

    def evaluate_test(self):
        """
            This method will evaluate the model on the test set at the end
            Returns:
                The accuracy of the model on the test set
        """
        accuracy = 0.0
        with torch.no_grad():
            for data in self.test_set:
                X, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(X)
                accuracy += self.accuracy(outputs, labels)

        print("Accuracy of the model on the test set : " + str(100 * accuracy / len(self.test_set)) + " %")

    def accuracy(self, outputs, labels):
        """
            This method calculate the accuracy of the model
            Args:
                outputs : the output of our model
                labels : the value we should obtain

            Returns:
                Accuracy of the model
        """
        predicted = outputs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)