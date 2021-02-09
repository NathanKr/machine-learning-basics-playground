from nn_learn_back_propagation_engine import Network
from utils import sigmoid , dsigmoid_to_dval
import numpy as np


def print_list(l,var_name):
    i=0
    while i < len(l):
        print(f"{var_name}[{i}].shape : {l[i].shape}")
        i += 1
    print(f"{var_name}\n{l}")


def learn_logical_and():
    net = Network([2, 1],sigmoid , dsigmoid_to_dval)
    # ok values are -30,20,20
    net.biases[0][0][0] = -25
    net.weights[0][0][0] = 15
    net.weights[0][0][1] = 15

    print_list(net.biases,"biases")
    print_list(net.weights,"weights")

    x1 = np.array([1 , 0 , 0 , 1])
    x2 = np.array([1 , 0 , 1 , 0])
    y =  np.array([1 , 0 , 0 , 0]) # logic and gate
    x= np.vstack((x1,x2)).T

    mini_batch = [(x_sample.reshape(x_sample.size,1),y_sample) for x_sample , y_sample in zip(x,y)]
    epochs = 30
    for j in range(epochs):
        net.update_mini_batch( mini_batch,  0.001)

    for x_sample in x:
        print(f"net.feedforward({x_sample}) : {net.feedforward(x_sample)}")

# main
learn_logical_and()    
