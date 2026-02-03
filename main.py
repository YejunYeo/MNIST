import random
import numpy as np
from torchvision import datasets, transforms
import torch

# weight are assigned random floats
weights = torch.rand(10,784)

# disabling the abbreviation of numpy arrays
np.set_printoptions(threshold=np.inf)

mnist = datasets.MNIST(root =" ./data", download = True, train = True )

# forward pass

# alright I asked Claude why I have to use  totensor instead of  PILToTensor it said smth abt gradient. Get back to it after u code the gradient part.

to_tensor = transforms.ToTensor()

# this will be 
forward_predictions = []



for image,label in mnist:
    img_to_tensor = to_tensor(image)
    flattened_tensor = torch.flatten(img_to_tensor)
    # maximum = -999
    num_scores = np.zeros(10)

    # a number is assigned for each possible digit from 0 to 9
    for i in range (10):
        # find the dot product beteween the image and the weights for each pixel
        score  =  torch.dot(flattened_tensor, weights[i])
        num_scores[i] = score

    # in forward pass,a prediction is made using initial weights
    
    #print (np.argmax(num_scores))
    
    # list of all the predictions made in the forward pass
    forward_predictions.append(num_scores)



loss = np.zeros(60000)



for i in range (len(forward_predictions)):
    correct_number = mnist[i][1]
    total_magnitude_of_all_preds = np.sum(forward_predictions[i])
    prediction_of_correct_num = forward_predictions[i][correct_number]
    prob_of_correctness = prediction_of_correct_num / total_magnitude_of_all_preds
    loss[i] = (-1 * np.log(prob_of_correctness))
        
print(loss)

# the loss will be approx 2.3 because the prob of correctness is about 0.1
# this is beacause the image is consisted of 0s and 1s that do not change its location so only weights at particular indexes are used
# For example, lets say that there are 300 1s in the 784 pixel image. 300 Weights will be multiplied by 1 and 300 * 0.5 = 150. 
# Due to Law of Large Numbers.



# Now Implement Back Propogation


