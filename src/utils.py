"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This file provide us with method useful for active learning
"""

import numpy as np

def prioritisation_scores(probabilities, number):
    """
        This method will return us the next values that we need to labelise
        for our training with the prioritisation score metric
        Args:
            probabilities : A matrix of probabilities with all the predictions 
                            for the unlabelled data
            number : The number of indexes that we need to return

        Returns:
            The indexes that we need to labelised ant enter in the training set
    """
    maxes = []
    for i in range (0, probabilities.shape[0]):
        maxes.append(np.max(probabilities[i]))
    return __get_min_indexes(maxes, number)

def margin_sampling(probabilities, number):
    """
        This method will return us the next values that we need to labelise
        for our training with the margin sample metric
        Args:
            probabilities : A matrix of probabilities with all the predictions 
                            for the unlabelled data
            number : The number of indexes that we need to return

        Returns:
            The indexes that we need to labelised ant enter in the training set
    """
    maxes = []
    maxesBis = []
    tmp = []
    for i in range (0, probabilities.shape[0]):
        maxes.append(np.max(probabilities[i]))
        tmp.append(np.delete(probabilities[i], np.where(probabilities[i] == np.max(probabilities[i]))[0]))
        maxesBis.append(np.max(tmp[i]))
        
    val = np.array(maxes) - np.array(maxesBis)

    return __get_min_indexes(val, number)

def entropy(probabilities, number):
    """
        This method will return us the next values that we need to labelise
        for our training with the entropy metric
        Args:
            probabilities : A matrix of probabilities with all the predictions 
                            for the unlabelled data
            number : The number of indexes that we need to return

        Returns:
            The indexes that we need to labelised ant enter in the training set
    """
    val = []
    for i in range (0, probabilities.shape[0]):
        tmp = 0
        for j in range (0, len(probabilities[i])):
            if probabilities[i][j] != 0 :
                tmp -= probabilities[i][j] * np.log(probabilities[i][j])
            else : 
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

    while len(result) < number :
        index = np.where(num_list == np.min(num_list))
        result.append(index)
        num_list.pop(index)

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

    while len(result) < number :
        index = np.where(num_list == np.max(num_list))
        result.append(index)
        num_list.pop(index)

    return result