import numpy as np


#Teta ia vector with 
def cost_function_logistic_regression_J(Teta,X,Y):
    m = Y.size
    H_linear_regression =  np.dot(X,Teta)
    H = sigmond(H_linear_regression)
    j_vec = Y * np.log(H) + (1-Y)*np.log(1-H)
    J = (-1/m)*np.sum(j_vec)

    return J

def sigmond(val):
    return 1/(1+np.exp(-val))