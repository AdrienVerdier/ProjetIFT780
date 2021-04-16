"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This File represent the VggNet Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.CNNBlocks import ConvBlock


class VGGNet(nn.Module):
    """
        Class used to implement the VggNet model
    """

    def __init__(self, in_channels, num_classes):
        """
            Args:
                in_channels: The input channel for this model
                num_classes: The number of classes
        """
        super(VGGNet, self).__init__()

        self.conv1 = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1)
        )

        self.conv3 = nn.Sequential(
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.conv4 = nn.Sequential(
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        )

        self.conv5 = nn.Sequential(
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
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
        out = self.maxPool(out)
        out = self.conv2(out)
        out = self.maxPool(out)
        out = self.conv3(out)
        out = self.maxPool(out)
        out = self.conv4(out)
        out = self.maxPool(out)
        out = self.conv5(out)
        out = self.maxPool(out)
        out = out.view(out.size(0), -1)

        out = self.linear_layers(out)

        out = F.log_softmax(out, dim=1)

        return out
