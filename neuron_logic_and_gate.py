import numpy as np
from utils import cost_function_logistic_regression_J  , sigmoid , sigmoid_binari
from scipy import optimize

def train():
    X1 = np.array([1 , 0 , 0 , 1])
    X2 = np.array([1 , 0 , 1 , 0])
    m = X1.size # should be same as the X2,y
    x0= np.ones(m)
    Y =  np.array([1 , 0 , 0 , 0]) # logic and gate

    X= np.vstack((x0,X1,X2)).T
    res = optimize.minimize(cost_function_logistic_regression_J, x0=[0,0,0], args=(X,Y))
    if(res.status != 0):
        raise Exception("Expecting status 0 but got : {}".format(res.status))
    Teta = res.x
    return Teta

def logic_and(x1,x2):
    Teta = train()
    X = np.array([1 , x1 , x2])
    return sigmoid_binari(np.dot(X,Teta))


print("0 && 0 : ",logic_and(0,0))
print("0 && 1 : ",logic_and(0,1))
print("1 && 0 : ",logic_and(1,0))
print("1 && 1 : ",logic_and(1,1))