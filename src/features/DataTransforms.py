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

    def __init__(self, data_aug=False, model_name="", dataset_name=""):
        """
            Args:
                data_aug: Boolean that say if data augmentation is activated
        """
        self.data_aug=data_aug
        self.model_name=model_name
        self.dataset_name=dataset_name

    def get_transforms(self):
        """
            This method defines the transform that we will use on our datasets
            Args:

            Returns:
                train_transform : Transforms used on the train data 
                test_transform : Transforms used on the test data
        """
        if self.data_aug :
            # We create the transforms if data augmentation is activated
            if self.dataset_name == "mnist":
                if self.model_name == "AlexNet":
                    test_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))
                    ])
                    train_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(10),
                        transforms.RandomCrop((227,227), padding=4)
                    ])
                elif self.model_name == "VGGNet" or self.model_name == "ResNet"  or self.model_name == "ResNext":
                    test_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))
                    ])
                    train_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(10),
                        transforms.RandomCrop((224,224), padding=4)
                    ])
                elif self.model_name == "LeNet":
                    test_transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))
                    ])
                    train_transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(10),
                        transforms.RandomCrop((32, 32), padding=4)
                    ])
            elif self.dataset_name == "cifar10": 
                if self.model_name == "AlexNet":
                    test_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    train_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=.20),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(10),
                        transforms.RandomCrop((227,227), padding=4)
                    ])
                elif self.model_name == "VGGNet" or self.model_name == "ResNet" or self.model_name == "ResNext":
                    test_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    train_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=.20),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(10),
                        transforms.RandomCrop((224,224), padding=4)
                    ])
                elif self.model_name == "LeNet":
                    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(10),
                        transforms.RandomCrop((32, 32), padding=4)
                    ])

            return train_transform, test_transform
        else :
            # We create the transforms if data augmentation is not activated
            if self.dataset_name == "mnist":
                if self.model_name == "AlexNet":
                    base_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))
                    ])
                elif self.model_name == "VGGNet" or self.model_name == "ResNet"  or self.model_name == "ResNext":
                    base_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))
                    ])
                elif self.model_name == "LeNet":
                    base_transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=0.5, std=0.5)
                    ])
            elif self.dataset_name == "cifar10": 
                if self.model_name == "AlexNet":
                    base_transform = transforms.Compose([
                        transforms.Resize((227, 227)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif self.model_name == "VGGNet" or self.model_name == "ResNet"  or self.model_name == "ResNext":
                    base_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif self.model_name == "LeNet":
                    base_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

            return base_transform, base_transform