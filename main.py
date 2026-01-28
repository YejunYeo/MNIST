import random
import numpy as np
from torchvision import datasets, transforms


# weight are assigned random floats
weights = np.random.rand(10,784)


mnist = datasets.MNIST(root =" ./data", download = True, train = True )

image,label = mnist[0]

print (label)

# forward pass

# alright I asked Claude why I have to use  totensor instead of  PILToTensor it said smth abt gradient. Get back to it after u code the gradient part.

for image,label in mnist:


