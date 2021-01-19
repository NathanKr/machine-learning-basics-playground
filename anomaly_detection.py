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
Xcv = mat_dict["Xval"]
ycv = mat_dict["yval"]

x1 = X[:,0]
x2 = X[:,1]
x1cv = Xcv[:,0]
x2cv = Xcv[:,1]


def plot_dataset():
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

def plot_contours():
    #X1, X2 = np.meshgrid(x1,x2) -> too big
    graph_start = 10
    graph_stop = 20
    X1, X2 = np.meshgrid(np.linspace(graph_start,graph_stop,100) ,np.linspace(graph_start,graph_stop,100) )
    P = np.vectorize(p_x1)(X1) * np.vectorize(p_x2)(X2)
    fig, ax = plt.subplots()
    CS = ax.contour(X1, X2, P)
    ax.clabel(CS, inline=True, fontsize=12)
    ax.set_title('p(x) = p(x1)*p(x2)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.ylim(graph_start,graph_stop)
    plt.xlim(graph_start,graph_stop)
    plt.plot(x1,x2,'.')
    plt.show() 

def plot_cv():
    ar_index_anomaly = np.where(ycv == 1)[0]
    ar_index_normal = np.where(ycv == 0)[0]

    plt.plot(x1cv[ar_index_anomaly],x2cv[ar_index_anomaly],'rx')
    plt.plot(x1cv[ar_index_normal],x2cv[ar_index_normal],'g.')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throuput (mb/s)')
    plt.title("server computers cross validation : x - anomaly , . - normal")
    plt.show()



def plots():
    plot_dataset()    
    plot_histogram() # check that it is normally distributed
    plot_probabilities()
    plot_contours()
    plot_cv()

# main
# plots()
plot_cv()



