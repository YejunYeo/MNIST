import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


data = pd.read_csv("/Users/yeoyejun/Downloads/train.csv")

data = data.to_numpy()

length = len(data)
#print(data)

pixels = 28 * 28
no_of_neurons_layer_1 = 32
no_of_neurons_layer_2 = 16

#forward pass
def sigmoid(x:int):
    return 1 / (1 + np.exp(-x))

def normalize_np_array(arr):
        norm_arr = arr/255
        return norm_arr

def feed_forward(input_pixels,weights, bias):
             weighted_sum  = np.dot(input_pixels, weights) + bias
             layer_1 = sigmoid(weighted_sum)
             return layer_1


for i in range (length):
    data_pixel_only= data[:,1:] 


data_pixel_only = normalize_np_array(data_pixel_only)
#data_pixel_only = np.transpose(data_pixel_only)


weights_layer1 =np.transpose( np.random.rand(no_of_neurons_layer_1,pixels))
biases_layer1 = np.random.rand(no_of_neurons_layer_1)


weights_layer2 = np.transpose(np.random.rand(16,32))

biases_layer2 = np.random.rand(16)

weights_final = np.random.rand(16, 10)
biases_final = np.random.rand(10)

labels_only = data[:, :1]

#calculate the loss
def softmax(arr):
    denominator = np.sum (np.exp(arr))
    arr = np.exp(arr)/denominator
    return arr

def cross_entropy(arr, answer):
    return(-1 *  np.log(arr[answer]))


def calculate_loss(layer_final, answer):
    arr = softmax(layer_final)
    return cross_entropy(arr,answer)

for i in range (length):
    layer_1 = feed_forward(data_pixel_only[i],weights_layer1, biases_layer1)
    layer_2 = feed_forward(layer_1, weights_layer2, biases_layer2)
    layer_final = feed_forward(layer_2, weights_final, biases_final)
    loss = calculate_loss(layer_final,labels_only[i])
    print (loss)

#print(layer_final)



