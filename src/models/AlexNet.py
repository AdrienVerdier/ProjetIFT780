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


class AlexNet(nn.Module):
    """
        Class used to implement the AlexNet model
    """

    def __init__(self, in_channels, num_classes):
        """
            Args:
                in_channels: The input channel for this model
                num_classes: The number of classes
        """
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            ConvBlock(in_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            ConvBlock(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.conv3 = ConvBlock(256, 384, kernel_size=3, stride=1, padding=1)

        self.conv4 = ConvBlock(384, 384, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Sequential(
            ConvBlock(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
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
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = out.view(out.size(0), -1)

        out = self.linear_layers(out)

        out = F.log_softmax(out, dim=1)

        return out
