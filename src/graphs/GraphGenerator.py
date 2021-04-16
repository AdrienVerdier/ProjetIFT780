"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This class will generate graphs
"""

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.utils import StoppingCriteria
from src.TrainManager import TrainManager
from datetime import datetime


class GraphGenerator():
    """
        Class used to create different graphs
    """

    def __init__(self, trainer=None, model=None, optimizer=None, train_set=None, test_set=None,
                 batch_size=None, num_epochs=None, validation_size=None, metric=None, use_cuda=None, query_size=None, size_data_first_learning=None, pool_size=None, dataset=None):
        """
            Args:
                trainer: The trainer manager for our model
                model: The model that we want to train
                train_set: The data used for the training
                test_set: The data used for the tests
                batch_size: The batch size that we want to use for the training
                num_epochs: the number of epochs for the training
                validation_size: Pourcentage of data to use for validation
                use_cada: If we want to use the gpu for the training
        """
        self.trainer = trainer
        self.model = model
        self.optimizer = optimizer
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.validation_size = validation_size
        self.use_cuda = use_cuda
        self.metric = metric
        self.size_data_first_learning = size_data_first_learning
        self.query_size = query_size
        self.pool_size = pool_size
        self.dataset = dataset

    def generate_graphs(self):
        """
            This method all of the graphs that we need for our final reports 
            and store them in the graphs folder
        """
        # We train for the random metric
        print("Training with random metric : ")
        self.loss_random, self.accuracy_random, self.random_score = self.__train('random')
        self.__reset_parameters()

        # We train for the entropy metric
        print("Training with entropy metric : ")
        self.loss_entropy, self.accuracy_entropy, self.entropy_score = self.__train('entropy')
        self.__reset_parameters()

        # We train for the lc metric
        print("Training with lc metric : ")
        self.loss_lc, self.accuracy_lc, self.lc_score = self.__train('lc')
        self.__reset_parameters()

        # We train for the margin metric
        print("Training with margin metric : ")
        self.loss_margin, self.accuracy_margin, self.margin_score = self.__train('margin')
        self.__reset_parameters()

        # We plot both graphs
        self.__plot_metric_comparision()
        self.__plot_final_pourcentages()

    def generate_graphs2(self):
        """
            This method will generate a graph of comparision between an active and
            a passive learning for a model
        """
        # We train for the classic learning
        print("Training with classic learning : ")
        self.trainer = TrainManager(model=self.model,
                           optimizer=self.optimizer,
                           train_set=self.train_set,
                           test_set=self.test_set,
                           batch_size=self.batch_size,
                           num_epochs=self.num_epochs,
                           validation_size=self.validation_size,
                           metric=self.metric,
                           use_cuda=True, 
                           query_size=self.query_size,
                           size_data_first_learning=self.size_data_first_learning,
                           pool_size=self.pool_size)

        self.trainer.train()

        self.classic_loss = self.trainer.val_loss
        self.classic_accuracy = self.trainer.val_accuracy
        self.classic_score = self.trainer.evaluate_test()
        self.__reset_parameters()

        # We train for the active learning
        print("Training with active learning : ")
        self.trainer = TrainManager(model=self.model,
                           optimizer=self.optimizer,
                           train_set=self.train_set,
                           test_set=self.test_set,
                           batch_size=self.batch_size,
                           num_epochs=self.num_epochs,
                           validation_size=self.validation_size,
                           metric=self.metric,
                           use_cuda=True, 
                           query_size=self.query_size,
                           size_data_first_learning=self.size_data_first_learning,
                           pool_size=self.pool_size)

        self.trainer.active_train(StoppingCriteria.STEP, self.num_epochs)

        self.active_loss = self.trainer.val_loss
        self.active_accuracy = self.trainer.val_accuracy
        self.active_score = self.trainer.evaluate_test()
        self.__reset_parameters()

        self.__plot_act_vs_pass()

    def __train(self, metric):
        """
            This method train a model for the metric we want and return all the values
            we need for the plots
            Args : 
                metric: The metric we want to train for

            Returns :
                All the loss that we got at each queries
                All the accuracy that we got at each queries
                The score of the model on the train set
        """
        self.trainer = TrainManager(model=self.model,
                           optimizer=self.optimizer,
                           train_set=self.train_set,
                           test_set=self.test_set,
                           batch_size=self.batch_size,
                           num_epochs=self.num_epochs,
                           validation_size=self.validation_size,
                           metric=metric,
                           use_cuda=True, 
                           query_size=self.query_size,
                           size_data_first_learning=self.size_data_first_learning,
                           pool_size=self.pool_size)

        self.trainer.active_train(StoppingCriteria.STEP, 10)
        return self.trainer.val_loss, self.trainer.val_accuracy, self.trainer.evaluate_test()

    def __reset_parameters(self):
        """
            This method will reset the parameters of our model

            This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
        """
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __plot_metric_comparision(self):
        """
            This method will plot the validation loss and accuracy for all our different metrics
        """
        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(range(1,len(self.loss_random)+1), self.loss_random, '-o', label='random')
        ax1.plot(range(1,len(self.loss_entropy)+1), self.loss_entropy, '-o', label='entropy')
        ax1.plot(range(1,len(self.loss_lc)+1), self.loss_lc, '-o', label='least confident')
        ax1.plot(range(1,len(self.loss_margin)+1), self.loss_margin, '-o', label='margin')
        ax1.set_title('validation loss per metric')
        ax1.set_xlabel('Queries')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(range(1,len(self.accuracy_random)+1), self.accuracy_random, '-o', label='random')
        ax2.plot(range(1,len(self.accuracy_entropy)+1), self.accuracy_entropy, '-o', label='entropy')
        ax2.plot(range(1,len(self.accuracy_lc)+1), self.accuracy_lc, '-o', label='least confident')
        ax2.plot(range(1,len(self.accuracy_margin)+1), self.accuracy_margin, '-o', label='margin')
        ax2.set_title('validation accuracy per metric')
        ax1.set_xlabel('Queries')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        f.text(0.1, -0.02, 'Settings used : ')
        f.text(0.1, -0.10, 'model = ' + str(type(self.model).__name__))
        f.text(0.1, -0.14, 'optimizer = ' + str(type(self.trainer.optimizer).__name__))
        f.text(0.3, -0.10, 'batch size = ' + str(self.trainer.batch_size))
        f.text(0.3, -0.14, 'num epochs = ' + str(self.trainer.num_epochs))
        f.text(0.5, -0.14, 'query size = ' + str(self.trainer.query_size))
        f.text(0.7, -0.10, 'size first learning = ' + str(self.trainer.size_data_first_learning))
        f.text(0.7, -0.14, 'dataset = ' + str(self.dataset))

        f.savefig('./graphs/metric_comparison_' + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") + '.png')
        plt.show()

    def __plot_act_vs_pass(self):
        """
            This method will plot the valisation loss and accuracy for an active and a passive
            learning of a model
        """
        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(range(1,len(self.classic_loss)+1), self.classic_loss, '-o', label='passive')
        ax1.plot(range(1,len(self.classic_loss)+1), self.active_loss, '-o', label='active')
        ax1.set_title('validation loss')
        ax1.set_xlabel('Queries or epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(range(1,len(self.classic_accuracy)+1), self.classic_accuracy, '-o', label='passive')
        ax2.plot(range(1,len(self.active_accuracy)+1), self.active_accuracy, '-o', label='active')
        ax2.set_title('validation accuracy')
        ax1.set_xlabel('Queries or epoch')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        f.text(0.1, -0.02, 'Settings used : ')
        f.text(0.1, -0.10, 'model = ' + str(type(self.model).__name__))
        f.text(0.1, -0.14, 'optimizer = ' + str(type(self.trainer.optimizer).__name__))
        f.text(0.3, -0.10, 'batch size = ' + str(self.trainer.batch_size))
        f.text(0.3, -0.14, 'num epochs = ' + str(self.trainer.num_epochs))
        f.text(0.5, -0.10, 'metric = ' + str(self.trainer.metric))
        f.text(0.5, -0.14, 'query size = ' + str(self.trainer.query_size))
        f.text(0.7, -0.10, 'size first learning = ' + str(self.trainer.size_data_first_learning))
        f.text(0.7, -0.14, 'dataset = ' + str(self.dataset))

        f.savefig('./graphs/active_vs_passive' + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") + '.png')
        plt.show()

    def __plot_final_pourcentages(self):
        """
            This method will plot the final pourcentages of our trainings on the test set
        """
        f = plt.figure()

        x = ['random', 'entropy', 'least confident', 'margin']
        height = [self.random_score, self.entropy_score, self.lc_score, self.margin_score]
        width = 0.5

        plt.bar(x, height, width, color='b')

        f.text(0.1, -0.02, 'Settings used : ')
        f.text(0.1, -0.10, 'model = ' + str(type(self.model).__name__))
        f.text(0.1, -0.14, 'optimizer = ' + str(type(self.trainer.optimizer).__name__))
        f.text(0.3, -0.10, 'batch size = ' + str(self.trainer.batch_size))
        f.text(0.3, -0.14, 'num epochs = ' + str(self.trainer.num_epochs))
        f.text(0.5, -0.14, 'query size = ' + str(self.trainer.query_size))
        f.text(0.7, -0.10, 'size first learning = ' + str(self.trainer.size_data_first_learning))
        f.text(0.7, -0.14, 'dataset = ' + str(self.dataset))

        f.savefig('./graphs/metric_score_' + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") + '.png')
        plt.show()

    def plot_metrics(self, active):
        """
            This method will plot train and validation losses and accuracies after the training phase

            This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
            Args:
                active : a boolean that say if this is active learning
        """
        x_value = range(1, len(self.trainer.train_loss) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(x_value, self.trainer.train_loss, '-o', label='Training loss')
        ax1.plot(x_value, self.trainer.val_loss, '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        if active :
            ax1.set_xlabel('Number instance queries')
        else:
            ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(x_value, self.trainer.train_accuracy, '-o', label='Training accuracy')
        ax2.plot(x_value, self.trainer.val_accuracy, '-o', label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        if active :
            ax2.set_xlabel('Number instance queries')
        else:
            ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        f.text(0.1, -0.02, 'Settings used : ')
        f.text(0.1, -0.10, 'model = ' + str(type(self.trainer.model).__name__))
        f.text(0.1, -0.14, 'optimizer = ' + str(type(self.trainer.optimizer).__name__))
        f.text(0.3, -0.10, 'batch size = ' + str(self.trainer.batch_size))
        f.text(0.3, -0.14, 'num epochs = ' + str(self.trainer.num_epochs))
        f.text(0.5, -0.10, 'metric = ' + str(self.trainer.metric))
        f.text(0.5, -0.14, 'query size = ' + str(self.trainer.query_size))
        f.text(0.7, -0.10, 'size first learning = ' + str(self.trainer.size_data_first_learning))
        f.text(0.7, -0.14, 'dataset = ' + str(self.dataset))

        f.savefig('./graphs/results_' + datetime.now().strftime("%d-%m-%Y_%Hh%Mm%Ss") + '.png', bbox_inches='tight')
        plt.show()