"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This file gather different methods useful on our datasets
"""

import torch
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler

def get_data(dataset, train_transform, test_transform):
    """
        This method download the dataset and apply the transform on it
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

    elif dataset == 'celebA' :
        train_set = datasets.CelebA(root='./data', split='train', download=True, transform=train_transform)
        test_set = datasets.CelebA(root='./data', split='test', download=True, transform=test_transform)

    elif dataset == 'mnist' : 
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    return train_set, test_set

def get_final_data(train_set, test_set, validation, batch_size):
    """
        This methods gets all the data that we need, ready to train our model
        Args:
            train_set : The data that we will use to train the model
            test_set : The data that we will use to test the model
            validation : the pourcentage of training data used for validation
            batch_size : the size of the batches

        Returns:
            final_train_set : final train set to train our model
            final_val_set : final validation set to validate our model
            final_test_set : final test set to validate our model
    """
    if type(validation) is float:
        train_sampler, val_sampler = __splitter(len(train_set), validation)
        final_train_set = DataLoader(train_set, batch_size, sampler=train_sampler)
        final_val_set = DataLoader(train_set, batch_size, sampler=val_sampler)
        final_test_set = DataLoader(test_set, batch_size, shuffle=True)
    else:
        final_train_set = DataLoader(train_set, batch_size, shuffle=True)
        final_val_set = DataLoader(validation, batch_size, shuffle=True)
        final_test_set = DataLoader(test_set, batch_size, shuffle=True)

    return final_train_set, final_val_set, final_test_set

def __splitter(dataset_size, validation):
    """
        This methods returns samplers to separate training data in
        a validation and a training set
        Args:
            dataset_size : size of the dataset
            validation : pourcentage of data to use in validation

        Returns:
            train_sampler : train sampler
            val_sampler : validation sampler
    """
    torch.manual_seed(0)
    shuffle_ids = torch.randperm(dataset_size).long()
    val_size = int(dataset_size * validation)
    train_ids = shuffle_ids[val_size:]
    val_ids = shuffle_ids[:val_size]
    return SubsetRandomSampler(train_ids), SubsetRandomSampler(val_ids)