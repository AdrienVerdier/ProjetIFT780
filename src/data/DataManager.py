"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This file gather different methods useful on our datasets
"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

def get_data(dataset, train_transform, test_transform):
    """
        This method download the dataset and apply the transform on it

        This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
        Args:
            dataset: The name of the dataset to load
            train_transform: the transform to use on the train data
            test_transform : the transform to use on the test data

        Returns:
            train_set : train set to train our model
            test_set : test set to validate our model
    """

    if dataset == 'cifar10' :
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    elif dataset == 'mnist' : 
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    return train_set, test_set

def get_final_data(train_set, test_set, validation):
    """
        This methods gets all the data that we need, ready to train our model

        This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
        Args:
            train_set : The data that we will use to train the model
            test_set : The data that we will use to test the model
            validation : the pourcentage of training data used for validation

        Returns:
            final_train_set : final train set to train our model
            final_val_set : final validation set to validate our model
            final_test_set : final test set to validate our model
    """
    if type(validation) is float:
        shuffle_ids = torch.randperm(len(train_set)).long()
        size = int(len(train_set) * validation)
        ids = shuffle_ids[:size].tolist()
        other_ids = shuffle_ids[size:].tolist()

        final_train_set = Subset(train_set, other_ids)
        val_set = Subset(train_set, ids)
        final_val_set = DataLoader(val_set, shuffle=True)
        final_test_set = DataLoader(test_set, shuffle=True)
    else:
        final_val_set = DataLoader(validation, shuffle=True)
        final_test_set = DataLoader(test_set, shuffle=True)

    return final_train_set, final_val_set, final_test_set