"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This File provides the method to implement a convolutional block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
        This class is implementing a classique convolutional block, ready to be used in a model
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
            Args:
                in_channels: number of input channels of the block
                out_channels: number of output channels of the block
                kernel_size: size of the kernel to use for the block
                stride: Stride to use in the block
                padding: Padding to use in the block
        """
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            This method implement the forward propagation of our block
            Args :
                x: The input of the block

            Returns :
                out: The output of the block
        """
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        x =  F.relu(self.downsample(x))

        return x + out
