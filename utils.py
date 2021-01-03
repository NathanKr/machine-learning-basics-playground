import numpy as np


# cost_function_logistic_regression_J

# Teta : 
#   - is features vector : teta_0,teta_1,....,teta_n . 
#   - Teta is column vector (n+1)x1 (n+1 is the number of features)
# X :
#   - a matrix [X0,X1,X2,....,Xn]. 
#   - each X0,X1,X2,....,Xn is a column vector mX1 
#   - m is number of data set points. 
#   - X is mxn. 
#   - Notice that X0 is always 1
# X*Teta :
#   - is : teta0*X0 + teta1*X1 + ….. +teta_n*Xn 
#   - X*Teta is a column vector mx1 

def cost_function_logistic_regression_J(Teta,X,Y):
    m = Y.size
    H_linear_regression =  np.dot(X,Teta)
    H = sigmond(H_linear_regression)
    j_vec = Y * np.log(H) + (1-Y)*np.log(1-H)
    J = (-1/m)*np.sum(j_vec)

    return J

def sigmond(val):
    return 1/(1+np.exp(-val))