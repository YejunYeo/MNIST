import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("/Users/yeoyejun/Downloads/train.csv")

data = data.to_numpy()

length = len(data)
print(data)
pixels = 28 * 28

#forward pass
def sigmoid(x:int):
    return 1 / (1 + np.exp(-x))

def normalize_np_array(arr):
        norm_arr = arr/255
        return norm_arr

def feed_forward(input_pixels,weights, bias):
             input_pixels = input_pixels.reshape(-1,1)
             weighted_sum  = np.dot( weights, input_pixels) + bias
             layer_1 = sigmoid(weighted_sum)
             return layer_1

def softmax(arr):
    denominator = np.sum (np.exp(arr))
    arr = np.exp(arr)/denominator
    return arr

def feed_forward_final(input_pixels,weights, bias):
             input_pixels = input_pixels.reshape(-1,1)
             weighted_sum  = np.dot(weights,input_pixels) + bias
             return softmax(weighted_sum);

data_pixel_only= data[:,1:] 

data_pixel_only = normalize_np_array(data_pixel_only)

weights_layer1 = np.random.randn(32,pixels)
biases_layer1 = np.random.rand(32, 1)

weights_layer2 = np.random.randn(16,32)

biases_layer2 = np.random.rand(16, 1)

weights_final = np.random.randn(10, 16)
biases_final = np.random.rand(10,1)

for i in range (length):
    layer_1 = feed_forward(data_pixel_only[i],weights_layer1, biases_layer1)
    layer_2 = feed_forward(layer_1, weights_layer2, biases_layer2)
    layer_final = feed_forward_final(layer_2, weights_final, biases_final)
    print(layer_final)
# Back Propagation

labels_only = data[:, :1]

#calculate the loss
def cross_entropy(arr, answer):
    return(-1 *  np.log(arr[answer]))

def calculate_loss(layer_final, answer):
    arr = softmax(layer_final)
    return cross_entropy(arr,answer)

def calculate_gradient(arr):
    arr = softmax(arr)
#backpropogation
# parameters should be weights, loss, input, bias
#def backprop(inputs,loss,weights,bias):
 #   loss = 

def make_one_hot_vector (index):
    one_hot = np.zeros(10)
    one_hot[index] = 1;
    return one_hot
    #one_hot = make_one_hot_vector(labels_only[i])

   # dl_dz3 = layer_final - one_hot
    #print (gradient)
    
    #loss = calculate_loss(layer_final,labels_only[i])
    #print (loss)

#print(layer_final)
