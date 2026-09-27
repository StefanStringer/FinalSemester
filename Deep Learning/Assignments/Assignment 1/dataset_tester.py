#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#import my custom dataset class
from insect_dataset import InsectDataset

# transform = transforms.Compose([transforms.ToTensor()]) #Change this as the images come in different sizes, so we need to resize them to a common size. We will resize them to 224x224 pixels.
transform = transforms.Compose([
    transforms.Resize((224, 224)), #Resize the images to 224x224 pixels
])

batch_size = 4


# Set up the dataset.
dataset = InsectDataset(# YOUR DATASET HERE
    annotations_file='/home/stefan_stringer/FinalSemester/Deep Learning/Assignments/Assignment 1/insects.csv',
    img_dir='/home/stefan_stringer/FinalSemester/Deep Learning/Assignments/Assignment 1/Insects',
    transform=transform
)



# Set up the dataset.
trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0) #Change this from 2 to 0 as specified in the email

# get some images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# for i in range(5): #Run through 5 batches
#     images, labels = next(dataiter)
#     for image, label in zip(images,labels): # Run through all samples in a batch
#         plt.figure()
#         plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
#         plt.title(label)
#         plt.show()
for i in range(1):  # Just one batch
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for j, (image, label) in enumerate(zip(images, labels)):
        axes[j].imshow(np.transpose(image.numpy(), (1, 2, 0)))
        axes[j].set_title(label)
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
# Print sample data in the requested format
print("Double check to see if it comes out like it shows in assignment description:\n")

for i in range(5):
    image, label = dataset[i]
    print(f"<image data from {dataset.img_labels.iloc[i, 2]}>,  {label}")
    # print(f"  Tensor shape: {image.shape}, dtype: {image.dtype}")
    # print()