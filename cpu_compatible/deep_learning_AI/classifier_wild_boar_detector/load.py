
#Importing all required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

from PIL import Image
import cv2

import matplotlib.pyplot as plt

import numpy as np


import json
import random

import PIL
import glob
import random
import os


#Defining the data directories
root_directory = 'folders'#'content/drive/MyDrive'
data_dir = root_directory# + '/'#20260103_vildsvinsdetektor'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
own_dir = data_dir + '/own'

#I load an image and plot it for examination.
#data = ImageFolder(train_dir)
#samples = data.samples
#image = samples[23][0]
#myimage = cv2.imread(image)
#plt.imshow(myimage)

#I also examine its shape.
#print(myimage.shape)

#Data augmentation:
#For the trainset I will perform RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation and ColorJitter.
#The purpose is to make the model generalize better and decrease overfitting.

"""
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transform = valid_transform

# TODO: Load the datasets with ImageFolder
trainset = ImageFolder(train_dir, transform=train_transform)
validset = ImageFolder(valid_dir, transform=valid_transform)
testset = ImageFolder(test_dir, transform=test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
validloader = torch.utils.data.DataLoader(validset, batch_size=64, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=4)


json_file = root_directory + '/' + 'classdict.json'
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
    classes_dict = cat_to_name


image_datasets = {
    'train': ImageFolder(train_dir, transform=train_transform),
    'valid': ImageFolder(valid_dir, transform=valid_transform),
    'test': ImageFolder(test_dir, transform=test_transform),
}

"""


# TODO: Write a function that loads a checkpoint and rebuilds the model

# I am defining the neural network here too, in case I come back to the code
# at another moment for working with the models output and I don't have time
# to run the training loop again.

classes_num = 5


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer
        self.fc1 = nn.Linear(25088, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(p=0.5)
        # Second layer
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(p=0.5)
        # Third (final) layer
        self.fc3 = nn.Linear(1024, 5)  # 5 classes

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

model = models.resnet50(pretrained=True)


model.classifier = Network()

# Load the saved checkpoint
checkpoint = torch.load('model_state.pth')

# Access the saved data
model_state_dict = checkpoint['model_state_dict']
training_loss_array = checkpoint['training loss']
valid_loss_array = checkpoint['validation loss']
epoch_array = checkpoint['epoch']
learning_rate = checkpoint['learning rate']
epochs = checkpoint['number of epochs trained']

model.class_to_idx = checkpoint['class_to_idx']

# Assuming you have already initialized your model architecture
model.load_state_dict(model_state_dict)


