"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This File represent the ResNext Model
"""

import torch
from torch import nn
from src.models.CNNBlocks import ConvBlock

class BottleneckResNextBlock(nn.Module):
    """
        This class is implementing a bottleneck block, ready to be used in our ResNext model
    """

    def __init__(self, in_channels, mid_channels, stride=1):
        """
            Args: 
                in_channels: The number of input channels
                mid_channels: The number of outputs channels
                stride: The stride to use in this block
        """
        super(BottleneckResNextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=32, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * 2, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(mid_channels * 2)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * 2, 1, stride=stride, padding=0),
            nn.BatchNorm2d(mid_channels * 2),
        )
        
        self.stride = stride

    def forward(self, x):
        """
            This method implement the forward propagation of our block
            Args :
                x: The input of the block

            Returns :
                out: The output of the block
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class ResNext(nn.Module):
    """
        Class used to implement the ResNext model
    """

    def __init__(self, in_channels, num_classes):
        """
            Args: 
                in_channels: The number of input channels for the model
                num_classes: The number of classes of the dataset
        """
        super(ResNext, self).__init__()

        self.inplanes = 64

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(128, 3)
        self.layer2 = self._make_layer(256, 4, stride=2)
        self.layer3 = self._make_layer(512, 6, stride=2)
        self.layer4 = self._make_layer(1024, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)


    def _make_layer(self, mid_channels, blocks, stride=1):
        """
            This method will create a new layer with bottleneck block
            Args:
                mid_channels: The number of channels in the middle of the block
                blocks: the number of blocks to create
                stride: the stride we want in the blocks

            Returns:
                The layers we created, ready to be add in the model
        """
        layers = []
        layers.append(BottleneckResNextBlock(self.inplanes, mid_channels, stride))
        self.inplanes = mid_channels * 2
        for _ in range(1, blocks):
            layers.append(BottleneckResNextBlock(self.inplanes, mid_channels))

        return nn.Sequential(*layers)

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