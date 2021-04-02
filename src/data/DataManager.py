"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This file gather different methods useful on our datasets
"""

from torchvision import datasets

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