import numpy as np
from utils import softplus , random_normal_distribution , linear_line
from scipy import optimize
import numpy as np

# same data set as in neural_network_learn_gradient_descent.py

x = np.array([0 , 0.5 , 1]) # input dosage : between 0 and 1
y = np.array([0 , 1 , 0]) # output efficacy : 0 or 1


# b1 : Teta[0]
# b2 : Teta[1]
# b3 : Teta[2]
# w1 : Teta[3]
# w2 : Teta[4]
# w3 : Teta[5]
# w4 : Teta[6]

def forward_propagation(Teta):
    # layer 1 -> layer 2
    z1 = linear_line(x,Teta[3],Teta[0])
    z2 = linear_line(x,Teta[4],Teta[1])
    a1 = softplus(z1)
    a2 = softplus(z2)
    # layer 2 -> layer 3
    z3 = linear_line(a1,Teta[5],0)
    z4 = linear_line(a2,Teta[6],0)
    h  = z3+z4+Teta[2]
    return h


def sum_square_residuals(Teta):
    """This is actually the cost function , J by Andrew Ng
    """
    h = forward_propagation(Teta)
    residual = y - h
    # this sum (y[i]-h[i])^2 over all items or using vector notation (y-h)^2
    ssr = np.dot(residual,residual) 
    return ssr



#res = optimize.minimize(sum_square_residuals, x0=[0,0,0,1,2,3,4]) --> this is not going to find the global minima
# the following will give almost zero ssr for most runs but not all
res = optimize.minimize(sum_square_residuals, x0=[0,0,0,random_normal_distribution(),random_normal_distribution(),random_normal_distribution(),random_normal_distribution()])
print("res : ",res)    
print("ssr : ",sum_square_residuals(res.x))
print("h : ",forward_propagation(res.x))