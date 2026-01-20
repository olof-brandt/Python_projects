
# @title


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


own_dir = data_dir + '/own'

json_file = root_directory + '/' + 'classdict.json'
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
    classes_dict = cat_to_name



# TODO: Write a function that loads a checkpoint and rebuilds the model

# I am defining the neural network here too, in case I come back to the code
# at another moment for working with the models output and I don't have time
# to run the training loop again.

# Set the device to GPU if one is available. Transfer the model to the GPU.
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#model.to(device)
#model.to(device)

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
model.to(device)


model.classifier = Network()

# Load the saved checkpoint
checkpoint = torch.load('model_state.pth', map_location=torch.device('cpu'))

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




def image_preprocessing(filename):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    print(filename)

    # Load the image into Python
    img = Image.open(filename).convert('RGB')


    # -----------men här behåller jag inte aspect ratio?
    # Resizing the image's shortest side to 256 pixels, while maintaining the aspect ratio.
    img.thumbnail(size=(256, 256))

    # Center cropping the image to 224x224 pixels.
    width, height = img.size
    width_224 = 224
    height_224 = 224
    left = (width - width_224) / 2
    right = (width + width_224) / 2
    top = (height - height_224) / 2
    bottom = (height + height_224) / 2
    img_cropped_224 = img.crop((left, top, right, bottom))

    # Transforming the image data to a torch tensor and normalization.
    transf_tensor = transforms.ToTensor()
    transf_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_processed = transf_tensor(img_cropped_224)
    img_processed = transf_norm(img_processed)

    # Transforming the image data into a numpy array.
    img_numpy = np.array(img_processed)

    return img_numpy


# Define the local adress to an image file.
filename = own_dir + '/' + 'hast01.jpg'
# print(filename)

# TODO: Process a PIL image for use in a PyTorch model

# image = image_preprocessing(filename)



################


def imshow(image, ax=None, title=None):


    fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose(1, 2, 0)

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    # Display the image.
    ax.imshow(image)

    return ax

# imshow(image)


################



file = 'folders/classdict.json'

def class_to_label(file, classes):
    with open(file, 'r') as f:
        class_map = json.load(f)
        labels = []
        for c in classes:
            labels.append(class_map[c])
            return labels

# idx_mapping = dict(map(reversed, class_to_idx.items()))


################


def predict(label_index, image, model, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.





    '''
    list_ps = []

    # Set the device to GPU if one is available. Transfer the model to the GPU.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    model.eval()

    image = torch.from_numpy(image)
    image = torch.unsqueeze(image, 0).to(device).float()
    image = image.to(device)


    print("Running own image into neural network image classifier.")

    log_ps = model.forward(image)
    log_ps = torch.nn.functional.log_softmax(log_ps, dim=1)


    ps = torch.exp(log_ps)
    # ps = torch.nn.functional.softmax(outputs, dim=1)
    top_ps, top_indices = ps.topk(1, dim=1)

    # print(ps)


    # Create a list of the probabilities of the top classes
    list_ps = top_ps.tolist()[0]
    # Create a list of the indices of the top classes
    list_indices = top_indices.tolist()[0]

    model.train()

    # Sort the classes dictionary
    sorted_classes_dict = dict(sorted(classes_dict.items(), key=lambda item: item[0]))
    # Extract the class names to a list of strings
    classes_strings_list = list(sorted_classes_dict.values())

    #print(list_indices)



    # Create an array with the top classes.
    top_classes = []
    for x in list_indices:
        top_classes.append(classes_strings_list[x])

    print(top_classes)


    return list_ps, top_indices, top_classes




import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torchvision.transforms as transforms

# Define image transformations (matching your training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load your model (ensure the class definition matches)
# ... (include the model definition code here)
#model = MyModel()
#model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define a function to classify images
def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # create a mini-batch
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        #log_ps = model.forward(image)
        #log_ps = torch.nn.functional.log_softmax(log_ps, dim=1)
        #ps = torch.exp(log_ps)
    # Get the top prediction
    confidence, predicted_idx = torch.max(probabilities, 0)
    # Map predicted_idx to class name (you need to define your class labels)
    class_name = class_names[predicted_idx]
    return class_name, confidence.item()

# GUI functions
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        label_img.config(image=img_tk)
        label_img.image = img_tk

        # Classify and display result
        class_name, confidence = classify_image(file_path)
        label_result.config(text=f"Prediction: {class_name} ({confidence*100:.2f}%)")

# Build GUI
root = tk.Tk()
root.title("Image Classifier")

btn_browse = tk.Button(root, text="Browse Image", command=open_file)
btn_browse.pack()

label_img = tk.Label(root)
label_img.pack()

label_result = tk.Label(root, text="Select an image to classify")
label_result.pack()

# Define your class labels
class_names = ["vildsvin", "gravling", "hast", "katt", "kakadua"]  # replace with your actual class labels
#{"01": "vildsvin", "02": "gravling", "03": "hast", "04": "katt", "05": "kakadua"}

root.mainloop()









