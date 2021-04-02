"""
Projet de session IFT780
Date:
Authors: Alexandre Turpin, Quentin Levieux and Adrien Verdier
License: Opensource, free to use
Other: This class is used to create our different transforms to 
        modify ou dataset (for data augmentation for exemple)
"""

import torchvision.transforms as transforms

class DataTransforms():
    """
        Class used to create our different transforms for data augmentation
    """

    def __init__(self, data_aug=False):
        """
            Args:
                data_aug: Boolean that say if data augmentation is activated
        """
        self.data_aug=data_aug

    def get_transforms(self):
        """
            This method defines the transform that we will use on our datasets
            Args:

            Returns:
                train_transform : Transforms used on the train data 
                test_transform : Transforms used on the test data
        """
        ##### TO DO #####
        if self.data_aug :
            print("Data augmentation activated")
        else :
            print("Data augmentation not activated")

        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return base_transform, base_transform