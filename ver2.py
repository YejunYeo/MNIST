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

#for i in range (length):
#    layer_1 = feed_forward(data_pixel_only[i],weights_layer1, biases_layer1)
#    layer_2 = feed_forward(layer_1, weights_layer2, biases_layer2)
#    layer_final = feed_forward_final(layer_2, weights_final, biases_final)
#    print(layer_final)

# Back Propagation

labels_only = data[:, :1]

#this is the cross entropy function
def calc_loss(arr, answer):
    return(-1 *  np.log(arr[answer]))

#one-hot vector of true probabilities
def true_probs(index):
    one_hot = np.zeros(10)
    one_hot[index] = 1
    one_hot =  one_hot.reshape(-1,1)
    return one_hot

def oneminus_activ(arr):
    new_arr = 1 - arr
    new_arr = np.transpose(new_arr)
    return new_arr

def make_one_hot_vector(num):
    arr = np.zeros(10)
    arr[num] = 1;
    return arr


for i in range (length):
    layer_1 = feed_forward(data_pixel_only[i],weights_layer1, biases_layer1)
    layer_2 = feed_forward(layer_1, weights_layer2, biases_layer2)
    layer_final = feed_forward_final(layer_2, weights_final, biases_final)
    loss = calc_loss(layer_final,labels_only[i])
    #dl_dz3 is a column vector
    dl_dz3 = layer_final - make_one_hot_vector(labels_only[i])
    dl_dw3 = np.dot(dl_dz3,np.transpose(layer_2))
    dl_db3 = dl_dz3
    dl_dz2 = np.dot(np.transpose(weights_final),dl_dz3) * ((layer_2) *(np.ones((16, 1)) - layer_2))
    dl_dw2 = np.dot(dl_dz2, np.transpose(layer_1))
    dl_db2 = dl_dz2
    dl_dz1 = np.dot(np.transpose(weights_layer2),dl_dz2) * ((layer_1) *(np.ones((32, 1)) - layer_1))
    dl_dw1 = np.dot(dl_dz1, np.transpose(data_pixel_only[i]))
    dl_db1 = dl_dz1
    learning_rate = 0.01
    weights_final = weights_final -  learning_rate * dl_dw3
    biases_final = biases_final - learning_rate * dl_db3
    weights_layer2 = weights_layer2 - learning_rate * dl_dw2
    biases_layer2 = biases_layer2 - learning_rate * dl_db2
    weights_layer1 = weights_layer1 - learning_rate * dl_dw1
    biases_layer1 = biases_layer1 - learning_rate * dl_db1

print(layer_final)
