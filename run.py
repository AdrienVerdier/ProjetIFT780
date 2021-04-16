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
from src.graphs.GraphGenerator import GraphGenerator
from src.TrainManager import TrainManager
from torch.optim import Adam, SGD
from src.models.AlexNet import AlexNet
from src.models.VGGNet import VGGNet
from src.models.ResNet import ResNet
from src.models.ResNext import ResNext
from src.models.LeNet import LeNet
from src.utils import StoppingCriteria

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

    parser.add_argument('--model', type=str, default='AlexNet',
                        choices=["AlexNet", "VGGNet", "ResNet", "LeNet", "ResNext"], help='The model to use')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=["cifar10", "mnist"],
                        help='The dataset to train on')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help='The optimizer that you want to use to train the model')
    parser.add_argument('--validation_size', type=float, default=0.1,
                        help='Percentage of the data to use for the validation')
    parser.add_argument('--batch_size', type=int, default=20, help='Size of each training batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epoch for the training')
    parser.add_argument('--metric', type=str, default='entropy', choices=["entropy", "margin", "lc", "random"],
                        help='The metric you want to use for active learning')
    parser.add_argument('--data_aug', action='store_true', help='If we want to use data_augmentation')
    parser.add_argument('--active_learning', action='store_true', help='If we want to use active learning')
    parser.add_argument('--generate_graphs1', action='store_true', help='If we want to generate all of our graphics')
    parser.add_argument('--generate_graphs2', action='store_true', help='If we want to generate all of our graphics')
    parser.add_argument('--query_size', type=int, default=1000, help='Size of data that we query for active learning')
    parser.add_argument('--size_data_first_learning', type=int, default=5000, help='Size of data of first dataset on active learning')
    parser.add_argument('--pool_base', type=int, default=0, help='Size of pooling on active learning')

    opts, _ = parser.parse_known_args()

    if opts.active_learning:
        group = parser.add_mutually_exclusive_group(required=True)

        group.add_argument('--stop_criteria_step', type=int,
                           help='The step criterion you use for stopping active learning')
        group.add_argument('--stop_criteria_threshold', type=float,
                           help='The accuracy threshold criterion you use for stopping active learning')
        group.add_argument('--stop_criteria_queries', type=int,
                           help='The query criterion use for stopping active learning')

    args = parser.parse_args()
    if args.pool_base < args.query_size and args.pool_base > 0 :
        parser.error('--pool_base can only be >= --query_size')

    return parser.parse_args()


if __name__ == "__main__":

    # First, we get all of the user's arguments
    args = argument_parser()

    validation_size = args.validation_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    data_aug = args.data_aug
    dataset = args.dataset

    # Here, we get the transform to use on our images depending on the parameters data_aug
    transforms = DataTransforms(data_aug, args.model, dataset)
    train_transform, test_transform = transforms.get_transforms()

    # We get the dataset we want and apply the transforms on it
    train_set, test_set = get_data(dataset, train_transform, test_transform)

    # We create our model (we take care of witch dataset we're using for the input channel)
    if dataset == 'mnist':
        in_channels = 1
    elif dataset == 'cifar10':
        in_channels=3

    if args.model == 'AlexNet':
        model = AlexNet(in_channels=in_channels,num_classes=10)
    elif args.model == 'VGGNet':
        model = VGGNet(in_channels=in_channels,num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet(in_channels=in_channels, num_classes=10)
    elif args.model == 'ResNext':
        model = ResNext(in_channels=in_channels, num_classes=10)
    elif args.model == 'LeNet':
        model = LeNet(in_channels=in_channels, num_classes=10)

    # We set the optimizer
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)

    # We create the training manager (WARNING : if you don't have a GPU, put use_cuda to False)
    trainer = TrainManager(model=model,
                           optimizer=optimizer,
                           train_set=train_set,
                           test_set=test_set,
                           batch_size=batch_size,
                           num_epochs=num_epochs,
                           validation_size=validation_size,
                           metric=args.metric,
                           use_cuda=True,
                           query_size=args.query_size,
                           size_data_first_learning=args.size_data_first_learning,
                           pool_size=args.pool_base)

    print("Training {} on {} for {} epochs".format(args.model, dataset, args.num_epochs))

    if args.generate_graphs1 :
        # If we want to generate our graphs for the report, we use this
        graphs_creator = GraphGenerator(model=model,
                           optimizer=optimizer,
                           train_set=train_set,
                           test_set=test_set,
                           batch_size=batch_size,
                           num_epochs=num_epochs,
                           validation_size=validation_size,
                           metric=args.metric,
                           use_cuda=True,
                           query_size=args.query_size,
                           size_data_first_learning=args.size_data_first_learning,
                           pool_size=args.pool_base,
                           dataset=dataset)
        graphs_creator.generate_graphs()
    elif args.generate_graphs2 :
        # If we want to generate our graphs for the report, we use this
        graphs_creator = GraphGenerator(model=model,
                           optimizer=optimizer,
                           train_set=train_set,
                           test_set=test_set,
                           batch_size=batch_size,
                           num_epochs=num_epochs,
                           validation_size=validation_size,
                           metric=args.metric,
                           use_cuda=True,
                           query_size=args.query_size,
                           size_data_first_learning=args.size_data_first_learning,
                           pool_size=args.pool_base,
                           dataset=dataset)
        graphs_creator.generate_graphs2()
    else :
        if args.active_learning:
            # We give the stopping criteria for active learning and we train the model
            if args.stop_criteria_step is not None:
                trainer.active_train(StoppingCriteria.STEP, args.stop_criteria_step)
            elif args.stop_criteria_queries is not None:
                trainer.active_train(StoppingCriteria.QUERIES, args.stop_criteria_queries)
            elif args.stop_criteria_threshold is not None:
                trainer.active_train(StoppingCriteria.THRESHOLD, args.stop_criteria_threshold)
        else:
            # We train the model for a classic learning
            trainer.train()

        trainer.evaluate_test()

        # We create our graph generator 
        graphs_creator = GraphGenerator(trainer, dataset=dataset)
        # We displays our graphs (and save our graphics)
        graphs_creator.plot_metrics(args.active_learning)
