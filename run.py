"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: Run File to run the project with the wanted parameters
"""

import argparse
from src.features.DataTransforms import DataTransforms
from src.data.DataManager import get_data
from src.TrainManager import TrainManager
from src.models.CNNVanilla import CnnVanilla

def argument_parser():
    """
        A parser to verify parameters and help the user use the program
        Args:

        Returns:
            parser : element that will contain all the user's arguments to run the program
    """

    parser = argparse.ArgumentParser(usage='\n python3 run.py [model] [dataset] [paramaters]',
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

    validation_size = args.validation_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    data_aug = args.data_aug

    # Here, we get the transform to use on our images depending on the parameters data_aug
    transforms = DataTransforms(data_aug)
    train_transform, test_transform = transforms.get_transforms()

    # We get the dataset we want and apply the transforms on it
    train_set, test_set = get_data(args.dataset, train_transform, test_transform)

    ##### TO DO #####
    # We set the optimizer
    # The name is set in args.optimizer

    #################

    ##### TO DO #####
    # We create our models
    # The name of our model is in args.model
    # TEMPORARY  (10 because CIFAR10 is 10 classes to watch for the others (10177 for celebA and mnist 10 classes)): 
    # Very long because there is no optimizer yet 
    model = CnnVanilla(num_classes=10)

    #################

    # We create the training manager (WARNING : if you don't have a GPU, put use_cuda to False)
    trainer = TrainManager(model=model,
                        train_set=train_set,
                        test_set=test_set,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        validation_size=validation_size,
                        use_cuda=True)

    print("Training {} on {} for {} epochs".format(args.model, args.dataset, args.num_epochs))

    trainer.train()
    trainer.evaluate_test()

    ##### TO DO #####
    # We displays our graphs (and save our graphics)