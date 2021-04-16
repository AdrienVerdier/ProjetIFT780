"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This class will provide us everything we need to train correctly 
        our different models
"""

import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Subset, SubsetRandomSampler, DataLoader, ConcatDataset
from src.data.DataManager import get_final_data
from src.utils import margin_sampling, least_confidence, entropy, StoppingCriteria, random_sampling


class TrainManager():
    """
        Class used to Manage the training of our different models
    """

    def __init__(self, model, optimizer, train_set, test_set, batch_size, num_epochs, validation_size, metric,
                 use_cuda, query_size, size_data_first_learning, pool_size):
        """
            Args:
                model: The model that we want to train
                optimizer: The optimizer we use for this model
                train_set: The data used for the training
                test_set: The data used for the tests
                batch_size: The batch size that we want to use for the training
                num_epochs: the number of epochs for the training
                validation_size: Pourcentage of data to use for validation
                use_cada: If we want to use the gpu for the training
                query_size: the number of data we want to add at each iteration of active learning
                size_data_first_learning: The number of data we want at the beginning of the active learning
        """
        # Set the device name if we want to use Cuda
        if use_cuda:
            device_name = 'cuda:0'
            if not torch.cuda.is_available():
                print("You are not able to run on a GPU")
                device_name = 'cpu'
        else:
            device_name = 'cpu'

        self.device = torch.device(device_name)
        self.train_set, self.val_set, self.test_set = get_final_data(train_set, test_set, validation_size)
        self.batch_size = batch_size
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.loss_function = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda

        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

        self.unknown_set = []

        self.size_data_first_learning = size_data_first_learning
        self.query_size = query_size
        self.pool_size = pool_size

    def train(self):
        """
            Train the models in a classic way

            This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
        """
        # Load the train set
        train_set = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            print("Epoch: " + str(epoch + 1) + " of " + str(self.num_epochs))
            train_loss = 0.0

            with tqdm(range(len(train_set))) as t:
                train_losses = []
                train_accuracies = []

                for i, data in enumerate(train_set, 0):
                    X, labels = data[0].to(self.device), data[1].to(self.device)

                    # clearing the Gradients of the model parameters
                    self.optimizer.zero_grad()

                    outputs = self.model(X)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()

                    self.optimizer.step()

                    train_losses.append(loss.item())
                    train_accuracies.append(self.accuracy(outputs, labels))

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

            self.train_loss.append(np.mean(train_losses))
            self.train_accuracy.append(np.mean(train_accuracies))
            self.evaluate_validation()

        print("Finished training")

    def active_train(self, criterion, value):
        """
            Train the model with active learning
        """
        # We load our starting dataset for the active learning
        load_train = self.__get_first_set()
        
        loop_number = 0
        is_criteria_not_good = True
        # We iterate while our criteria for active learning is not completed
        while is_criteria_not_good:
            print("Active learning : " + str(loop_number + 1))

            # We do some epochs like in the classic training
            for epoch in range(self.num_epochs):
                print("Epoch: " + str(epoch + 1) + " of " + str(self.num_epochs))
                train_loss = 0.0

                with tqdm(range(len(load_train))) as t:
                    train_losses = []
                    train_accuracies = []

                    for i, data in enumerate(load_train, 0):
                        X, labels = data[0].to(self.device), data[1].to(self.device)

                        # clearing the Gradients of the model parameters
                        self.optimizer.zero_grad()

                        outputs = self.model(X)
                        loss = self.loss_function(outputs, labels)
                        loss.backward()

                        self.optimizer.step()

                        train_losses.append(loss.item())
                        train_accuracies.append(self.accuracy(outputs, labels))

                        train_loss += loss.item()
                        t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                        t.update()

            loop_number += 1

            self.train_loss.append(np.mean(train_losses))
            self.train_accuracy.append(np.mean(train_accuracies))
            self.evaluate_validation()

            if criterion == StoppingCriteria.STEP:
                # We verrify if we have done the good amount of step
                if loop_number == value:
                    is_criteria_not_good = False
            elif criterion == StoppingCriteria.QUERIES:
                # We verify if we have the good amount of queries
                size_actual_dataset = len(load_train.dataset)
                if size_actual_dataset - self.size_data_first_learning >= value:
                    is_criteria_not_good = False
            elif criterion == StoppingCriteria.THRESHOLD:
                # We verify if our model still learn enough
                if loop_number >= 3:
                    if abs(self.val_accuracy[loop_number - 1] - self.val_accuracy[loop_number - 2]) < value and abs(self.val_accuracy[loop_number - 2] - self.val_accuracy[loop_number - 3]) < value:
                        is_criteria_not_good = False

            # If it is not finish, we get the new dataset for the training
            if is_criteria_not_good:
                print(len(load_train.dataset))
                load_train = self.__get_next_set()

            self.__reset_parameters()

        print("Finished training")

    def __get_first_set(self):
        """
            This method will get us our first set for the active learning
            Returns:
                first_set: The set that we will train on
        """
        torch.manual_seed(0)
        shuffle_ids = torch.randperm(len(self.train_set)).long()
        ids = shuffle_ids[:self.size_data_first_learning].tolist()
        other_ids = shuffle_ids[self.size_data_first_learning:].tolist()

        self.unknown_set = Subset(self.train_set, other_ids)
        self.train_set = Subset(self.train_set, ids)

        first_set = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        return first_set

    def __get_next_set(self):
        """
            This method will get us the next set that we are gonna
            use for the active learning
            Returns :
                next_set : The set that we are going to train on
        """
        # We put the model in eval mode to go faster
        self.model.eval()
        # We get the output for all the unknowns data

        if self.pool_size > 0 :
            if self.pool_size > len(self.unknown_set) :
                self.pool_size = len(self.unknown_set)
            pool_idx = random.sample(list(range(0, len(self.unknown_set))), self.pool_size)
            sampler = SubsetRandomSampler(pool_idx)
            dataset = DataLoader(self.unknown_set, batch_size=128, num_workers=4, sampler=sampler)
            idx = list(iter(sampler))
        else:
            dataset = DataLoader(self.unknown_set, batch_size=128, num_workers=4)
            idx = list(range(0, len(self.unknown_set)))

        outputs = []
        
        with torch.no_grad():
            for batch in dataset:
                data, _ = batch
                out = self.model(data.to(self.device)).tolist()
                for values in out:
                    outputs.append(values)

        outputs = np.array(outputs)
        
        # We launch the research for the new dataset
        if self.metric == "lc":
            ids = np.take(idx, least_confidence(outputs, self.query_size))
            other_ids = list(range(0, len(self.unknown_set)))
            for ind in sorted(ids, reverse=True):
                other_ids.pop(np.where(other_ids == ind)[0][0])

            new = Subset(self.unknown_set, ids)
            self.train_set = ConcatDataset([self.train_set, new])
            self.unknown_set = Subset(self.unknown_set, other_ids)
            next_set = DataLoader(self.train_set, batch_size=self.batch_size)

        elif self.metric == "margin":
            ids = np.take(idx, margin_sampling(outputs, self.query_size))
            other_ids = list(range(0, len(self.unknown_set)))
            for ind in sorted(ids, reverse=True):
                other_ids.pop(np.where(other_ids == ind)[0][0])

            new = Subset(self.unknown_set, ids)
            self.train_set = ConcatDataset([self.train_set, new])
            self.unknown_set = Subset(self.unknown_set, other_ids)
            next_set = DataLoader(self.train_set, batch_size=self.batch_size)

        elif self.metric == "entropy":
            ids = np.take(idx, entropy(outputs, self.query_size))
            other_ids = list(range(0, len(self.unknown_set)))
            for ind in sorted(ids, reverse=True):
                other_ids.pop(np.where(other_ids == ind)[0][0])

            new = Subset(self.unknown_set, ids)
            self.train_set = ConcatDataset([self.train_set, new])
            self.unknown_set = Subset(self.unknown_set, other_ids)
            next_set = DataLoader(self.train_set, batch_size=self.batch_size)

        elif self.metric == "random":
            ids = np.take(idx, random_sampling(outputs, self.query_size))
            other_ids = list(range(0, len(self.unknown_set)))
            for ind in sorted(ids, reverse=True):
                other_ids.pop(ind)

            new = Subset(self.unknown_set, ids)
            self.train_set = ConcatDataset([self.train_set, new])
            self.unknown_set = Subset(self.unknown_set, other_ids)
            next_set = DataLoader(self.train_set, batch_size=self.batch_size)

        # We put him back in training mode
        self.model.train()

        return next_set

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

    def evaluate_validation(self):
        """
            This method will, at every epoch, evaluate the results on the validation set

            This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
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

            This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
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

        return 100 * accuracy / len(self.test_set)

    def accuracy(self, outputs, labels):
        """
            This method calculate the accuracy of the model

            This method was extracted in big parts from a TP made by Mamadou Mountagha BAH & Pierre-Marc Jodoin
            Args:
                outputs : the output of our model
                labels : the value we should obtain

            Returns:
                Accuracy of the model
        """
        predicted = outputs.argmax(dim=1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)