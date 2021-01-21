import numpy as np
from utils import cost_function_logistic_regression_J  , sigmoid
from scipy import optimize

x1 = np.array([1 , 0 , 0 , 1])
x2 = np.array([1 , 0 , 1 , 0])
m = x1.size # should be same as the x2,y
x0= np.ones(m)
Y =  np.array([1 , 0 , 0 , 0]) # logic and gate

X= np.vstack((x0,x1,x2)).T
res = optimize.minimize(cost_function_logistic_regression_J, x0=[0,0,0], args=(X,Y))
print(res)
Teta = res.x

H = sigmoid(np.dot(X,Teta))
logic_and_gate = H >= 0.5

print("logic_and_gate == Y : " , logic_and_gate == Y)