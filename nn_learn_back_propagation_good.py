from nn_learn_back_propagation_good_engine import Network
from utils import sigmoid , dsigmoid_to_dval
import numpy as np


def print_list(l,var_name):
    i=0
    while i < len(l):
        print(f"{var_name}[{i}].shape : {l[i].shape}")
        i += 1
    print(f"{var_name}\n{l}")


def learn_logical_and():
    obj = Network([2, 1],sigmoid , dsigmoid_to_dval)
    # these are correct values
    obj.biases[0][0][0] = -30
    obj.weights[0][0][0] = 20
    obj.weights[0][0][1] = 20

    print_list(obj.biases,"biases")
    print_list(obj.weights,"weights")

    x1 = np.array([1 , 0 , 0 , 1])
    x2 = np.array([1 , 0 , 1 , 0])
    y =  np.array([1 , 0 , 0 , 0]) # logic and gate
    x= np.vstack((x1,x2)).T
    i_sample = 0
    x_sample = x[i_sample].reshape((x[i_sample].size,1))
    y_sample = y[i_sample]
    print(f"feedforward\n{obj.feedforward(x_sample)}")
    (nabla_b , nabla_w) = obj.backprop(x_sample,y_sample)
    print(f"nabla_b\n{nabla_b}")
    print(f"nabla_w\n{nabla_w}")

# main
learn_logical_and()    
