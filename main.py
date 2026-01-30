import random
import numpy as np
from torchvision import datasets, transforms
import torch

# weight are assigned random floats
weights = torch.rand(10,784)


mnist = datasets.MNIST(root =" ./data", download = True, train = True )

image,label = mnist[0]
# forward pass

# alright I asked Claude why I have to use  totensor instead of  PILToTensor it said smth abt gradient. Get back to it after u code the gradient part.

to_tensor = transforms.ToTensor()


for image,label in mnist:
    img_to_tensor = to_tensor(image)
    flattened_tensor = torch.flatten(img_to_tensor)
    # maximum = -999
    num_scores = np.zeros(10)

    # a number is assigned for each possible digit from 0 to 9
    for i in range (10):
        score  =  torch.dot(flattened_tensor, weights[i])
        num_scores[i] = score

    # in forward pass,a prediction is made using initial weights
    print (np.argmax(num_scores))

print ("Hello WOrld")


