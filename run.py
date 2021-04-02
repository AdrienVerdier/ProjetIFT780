"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux et Adrien Verdier
License: Opensource, free to use
Other: Run File to run the project with the wanted parameters
"""

import argparse

def argument_parser():
    """
        A parser to verify parameters and help the user use the program
        Args:

        Returns:
            parser : element that will contain all the user's arguments to run the program
    """

    parser = argparse.ArgumentParser(usage='\n python3 run.py [model] [dataset] [paramaters]'
                                    description="This program allows to run different neural networks on"
                                    " different datasets.")
    
    parser.add_argument('--model', type=str, default='MLP',
                        choices=["MLP", "to define"], help='The model to use')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "celebA", "mnist"],
                        help='The dataset to train on')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help='The optimizer that you want to use to train the model')
    parser.add_argument('--validation_size', type=float, default=0.1, 
                        help='Percentage of the data to use for the validation')
    parser.add_argument('--batch_size', type=int, default=20, help='Size of each training batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epoch for the training')
    parser.add_argument('--data_aug', action='store_true', help='If we want to use data_augmentation')

    return parser.parse_args()

if __name__ == "__main__":

    # First, we get all of the user's arguments
    args = argument_parser()

    val_size = args.validation_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    data_aug = args.data_aug

    print("Model = ", args.model)
    print("Dataset = ", args.dataset)
    print("optimizer = ", args.optimizer)
    print("validation_size = ", val_size)
    print("batch_size = ", batch_size)
    print("learning_rate = ", learning_rate)
    print("num_epochs = ", num_epochs)
    print("data_aug = ", data_aug)