"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This File represent the AlexNet Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.CNNBlocks import ConvBlock


class LeNet(nn.Module):
    """
        Class used to implement the LeNet model
    """

    def __init__(self, in_channels, num_classes):
        """
            Args:
                in_channels: The input channel for this model
                num_classes: The number of classes
        """
        super(LeNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )

        self.subsampling2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.subsampling4 = nn.AvgPool2d(kernel_size=2)

        self.linear_layers = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=84, out_features=num_classes)
        )

    def forward(self, x):
        """
            This method implement the forward propagation of our model
            Args :
                x: The input of the model

            Returns :
                out: The output of the model
        """
        out = self.conv1(x)
        out = self.subsampling2(out)
        out = self.conv3(out)
        out = self.subsampling4(out)
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)
        out = F.log_softmax(out, dim=1)

        return out
