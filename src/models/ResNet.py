"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This File represent the ResNet Model
"""

import torch
from torch import nn
from src.models.CNNBlocks import ConvBlock
from src.models.CNNBlocks import ResBlock


class ResNet(nn.Module):
    """
        Class used to implement the ResNext model
    """

    def __init__(self, in_channels, num_classes):
        """
            Args: 
                in_channels: The number of input channels for the model
                num_classes: The number of classes of the dataset
        """
        super(ResNet, self).__init__()

        self.inplanes = 64

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128)
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256)
        )
        self.layer4 = nn.Sequential(
            ResBlock(256,512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, num_classes)

    def forward(self, x):
        """
            This method implement the forward propagation of our model
            Args :
                x: The input of the model

            Returns :
                out: The output of the model
        """

        out = self.init_conv(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out