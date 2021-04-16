"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This file provide us with method useful for active learning
"""

import numpy as np
from enum import Enum
import random


class StoppingCriteria(Enum):
    STEP = 1
    QUERIES = 2
    THRESHOLD = 3


def random_sampling(predictions, number):
    """
        This method will return us the next values that we need to labelise
        for our training with a random prioritisation
        Args:
            predictions : A matrix of probabilities with all the predictions
                            for the unlabelled data
            number : The number of indexes that we need to return

        Returns:
            The indexes that we need to labelised and enter in the training set
    """
    return random.sample(range(len(predictions)), number)

def least_confidence(predictions, number):
    """
        This method will return us the next values that we need to labelise
        for our training with the prioritisation score metric
        Args:
            predictions : A matrix of probabilities with all the predictions
                            for the unlabelled data
            number : The number of indexes that we need to return

        Returns:
            The indexes that we need to labelised and enter in the training set
    """
    maxes = []
    for i in range(0, predictions.shape[0]):
        maxes.append(np.max(predictions[i]))
    return __get_min_indexes(maxes, number)


def margin_sampling(predictions, number):
    """
        This method will return us the next values that we need to labelise
        for our training with the margin sample metric
        Args:
            predictions : A matrix of probabilities with all the predictions
                            for the unlabelled data
            number : The number of indexes that we need to return

        Returns:
            The indexes that we need to labelised and enter in the training set
    """
    maxes = []
    maxesBis = []
    tmp = []
    for i in range(0, predictions.shape[0]):
        maxes.append(np.max(predictions[i]))
        tmp.append(np.delete(predictions[i], np.where(predictions[i] == np.max(predictions[i]))[0]))
        maxesBis.append(np.max(tmp[i]))

    val = np.array(maxes) - np.array(maxesBis)

    return __get_min_indexes(val, number)


def entropy(predictions, number):
    """
        This method will return us the next values that we need to labelise
        for our training with the entropy metric
        Args:
            predictions : A matrix of probabilities with all the predictions
                            for the unlabelled data
            number : The number of indexes that we need to return

        Returns:
            The indexes that we need to labelised and enter in the training set
    """
    val = []
    for i in range(0, predictions.shape[0]):
        tmp = 0
        for j in range(0, len(predictions[i])):
            if predictions[i][j] > 0:
                tmp -= predictions[i][j] * np.log(predictions[i][j])
            else:
                tmp -= 0.000001 * np.log(0.000001)
        val.append(tmp)

    return __get_max_indexes(val, number)


def __get_min_indexes(num_list, number):
    """
        This method returns us the n minimums indexes from a list
        Args :
            num_list : our list of values
            number : the number of values we want

        Returns:
            result : a list of the number minimum indexes
    """
    result = []

    num_list = np.array(num_list)
    result = np.argpartition(num_list, number)[:number]

    return result


def __get_max_indexes(num_list, number):
    """
        This method returns us the n maximums indexes from a list
        Args :
            num_list : our list of values
            number : the number of values we want

        Returns:
            result : a list of the number maximum indexes
    """
    result = []

    num_list = np.array(num_list)
    result = num_list.argsort()[-number:][::-1]

    return result
