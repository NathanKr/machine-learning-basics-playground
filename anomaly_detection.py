from os.path import join 
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import normal_dist , F1score


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

def p_x1_vec(_x1):
    return np.vectorize(p_x1)(_x1)

def p_x2_vec(_x2):
    return np.vectorize(p_x1)(_x2)    

def p_x_vec(_x1,_x2):
    return p_x1_vec(_x1) * p_x2_vec(_x2)


def plot_probabilities():
    fig, axs = plt.subplots(2)
    axs[0].plot(x1,p_x1_vec(x1),'.')
    axs[0].set_title('p(x1) -> check that its normal distributed')
    axs[1].plot(x2,p_x2_vec(x2),'.')
    axs[1].set_title('p(x2) -> check that its normal distributed')
    plt.show()

def plot_contours():
    #X1, X2 = np.meshgrid(x1,x2) -> too big
    graph_start = 10
    graph_stop = 20
    X1, X2 = np.meshgrid(np.linspace(graph_start,graph_stop,100) ,np.linspace(graph_start,graph_stop,100) )
    P = p_x_vec(X1,X2)
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
    plt.title("server computers ycv : anomaly - x, normal - . ")
    plt.show()



def plots():
    plot_dataset()    
    plot_histogram() # check that it is normally distributed
    plot_probabilities()
    plot_contours()
    plot_cv()

def plot_results(_best_eps):
    fig, axs = plt.subplots(2)
    # cross validation
    ar_index_anomaly = np.where(ycv == 1)[0]
    ar_index_normal = np.where(ycv == 0)[0]
    axs[0].plot(x1cv[ar_index_anomaly],x2cv[ar_index_anomaly],'rx')
    axs[0].plot(x1cv[ar_index_normal],x2cv[ar_index_normal],'g.')
    axs[0].set_title("server computers ycv : anomaly - x , normal - .")

    # h = p(x) < eps
    p = p_x_vec(x1cv,x2cv)
    h = np.where(p < _best_eps ,1 ,0) # 1 is anomally
    ar_index_anomaly = np.where(h == 1)[0]
    ar_index_normal = np.where(h == 0)[0]
    axs[1].plot(x1cv[ar_index_anomaly],x2cv[ar_index_anomaly],'rx')
    axs[1].plot(x1cv[ar_index_normal],x2cv[ar_index_normal],'g.')
    axs[1].set_title("server computers h = p(x) < best_eps : anomaly - x , normal - . ")

    plt.show()




def compute_epsilon():
    # loop epsilon and compute score
    p = p_x_vec(x1cv,x2cv)
    num_eps = 50
    i = 0
    eps=1
    vec_F1score = []
    vec_eps = []
    while i < num_eps:
        actual_y = ycv
        h = np.where(p < eps ,1 ,0) # 1 is anomally
        vec_F1score.append(F1score(actual_y,h))
        vec_eps.append(eps)
        eps = eps / 2
        i += 1

    max_F1score_index = np.argmax(vec_F1score)
    eps_max_F1score = vec_eps[max_F1score_index]
    plt.plot(np.log10(vec_eps),vec_F1score,'x')
    plt.title("look for eps which maximaize F1Score (0 - 1) -> eps is {}".format(eps_max_F1score))
    plt.xlabel('log10(eps)')
    plt.ylabel('F1score')
    plt.grid()
    plt.show()
    return eps_max_F1score
# main
plots()
best_eps = compute_epsilon()
plot_results(best_eps)




