from os.path import join 
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import normal_dist


current_dir = os.path.abspath(".")
data_dir = join(current_dir, 'data')
file_name = join(data_dir,"ex8data1.mat")
mat_dict = sio.loadmat(file_name)
# print("mat_dict.keys() : ",mat_dict.keys())

X = mat_dict["X"]
Xval = mat_dict["Xval"]
yval = mat_dict["yval"]

x1 = X[:,0]
x2 = X[:,1]

def plots():
    plt.plot(x1,x2,'x')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throuput (mb/s)')
    plt.title("server computers train dataset - unsupervised learning")
    plt.show()

def plot_histogram():
    sns.histplot(data=x1)
    plt.xlabel("x1")
    plt.show()
    sns.histplot(data=x2)
    plt.xlabel("x2")
    plt.show()

def p_x1(sample_x):
    return normal_dist(sample_x,x1)

def p_x2(sample_x):
    return normal_dist(sample_x,x2)

def p_x1_vec():
    return np.vectorize(p_x1)(x1)

def p_x2_vec():
    return np.vectorize(p_x1)(x2)    

def plot_probabilities():
    fig, axs = plt.subplots(2)
    axs[0].plot(x1,p_x1_vec(),'.')
    axs[0].set_title('p(x1)')
    axs[1].plot(x2,p_x2_vec(),'.')
    axs[1].set_title('p(x2)')
    plt.show()

#plots()    
#plot_histogram() # check that it is normally distributed
#plot_probabilities()



# P_X1, P_X2 = np.meshgrid(p_x1_vec(),p_x2_vec())
# P = P_X1*P_X2
# print(P.shape)
# plt.contour(P_X1, P_X2, P,colors='black')
# plt.show()

X1, X2 = np.meshgrid(x1,x2)
P = np.vectorize(p_x1)(X1) * np.vectorize(p_x2)(X2)
plt.contour(X1, X2, P)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('p(x) = p(x1)*p(x2)')
plt.show()