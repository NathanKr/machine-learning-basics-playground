from nn_learn_back_propagation_engine import Network
from utils import sigmoid , dsigmoid_to_dval , softplus , dsoftplus_to_dval 
import numpy as np



def learn_logical_and():
    net = Network([2, 1],sigmoid , dsigmoid_to_dval)
    # by default it start with random numbers
    net.biases[0][0][0] = -25 # exact value is -30
    net.weights[0][0][0] = 15 # exact value is 20
    net.weights[0][0][1] = 15 # exact value is 20

    net.print_shapes()

    x1 = np.array([1 , 0 , 0 , 1])
    x2 = np.array([1 , 0 , 1 , 0])
    y =  np.array([1 , 0 , 0 , 0]) # logic and gate
    x= np.vstack((x1,x2)).T

    mini_batch = [(x_sample.reshape(x_sample.size,1),y_sample) for x_sample , y_sample in zip(x,y)]
    epochs = 30
    net.train(mini_batch,epochs,0.01)

    for x_sample in x:
        print(f"net.feedforward({x_sample}) : {net.feedforward(x_sample)}")


def learn_StatsQuest():
    # this will NOT match StatsQuest because there the activation on layer 3 is linear
    # while the activation on layer 2 is softplus . 
    # However , currently Network class has the same activation for every layer. 
    # it is not difficult to support activation per nuron but currently it is not supported
    net = Network([1 , 2, 1],softplus , dsoftplus_to_dval)

    # layer 1 -> layer 2
    net.biases[0][0][0] =  0.50560784 # StatsQuest b1 , exact is  0.50560784
    net.biases[0][1][0] = -10.69737385 # StatsQuest b2 , exact is -10.69737385

    net.weights[0][0][0] =  0.54058468 # StatsQuest w1 , exact is  0.54058468
    net.weights[0][1][0] = 12.98906751 # StatsQuest w2 , exact is 12.98906751

    # layer 2 -> layer 3
    net.biases[1][0][0] = -5.59668029 # StatsQuest b3 , exact is -5.5966802
    net.weights[1][0][0] = 5.72510687 # StatsQuest w3 , exact is 5.72510687
    net.weights[1][0][1] = -0.88626893 # StatsQuest w4 , exact is -0.88626893

    net.print_shapes()
    
    x = np.array([0 , 0.5  , 1])
    y =  np.array([0 , 1 , 0]) 

    mini_batch = [(x_sample.reshape(x_sample.size,1),y_sample) for x_sample , y_sample in zip(x,y)]
    epochs = 30
    net.train(mini_batch,epochs,0.01)

    for x_sample in x:
        print(f"net.feedforward({x_sample}) : {net.feedforward(x_sample)}")


# main
learn_logical_and()    
#learn_StatsQuest()