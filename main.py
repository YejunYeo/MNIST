import random
import numpy as np
from torchvision import datasets, transforms
import torch

# weight are assigned random floats
weights = torch.rand(10,784)


mnist = datasets.MNIST(root =" ./data", download = True, train = True )

image,label = mnist[0]

print (label)

# forward pass

# alright I asked Claude why I have to use  totensor instead of  PILToTensor it said smth abt gradient. Get back to it after u code the gradient part.

to_tensor = transforms.ToTensor()

for image,label in mnist:
    img_to_tensor = to_tensor(image)
    flattened_tensor = torch.flatten(img_to_tensor)
    for i in range 10:
        max = -9999
        scores =  torch.dot(flattened_tensor, weights[i])
        if scores > max:

    

