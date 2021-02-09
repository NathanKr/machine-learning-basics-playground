import numpy as np
from utils import sum_square_residuals , softplus , random_normal_distribution , linear_line
from scipy import optimize
import numpy as np


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


def sum_square_residuals_with_Teta(Teta):
    """This is actually the cost function , J by Andrew Ng
    """
    h = forward_propagation(Teta)
    return sum_square_residuals(y,h)

def cost_function(Teta):
    return sum_square_residuals_with_Teta(Teta)

#res = optimize.minimize(sum_square_residuals, x0=[0,0,0,1,2,3,4]) --> this is not going to find the global minima
# the following will give almost zero ssr for most runs but not all 
# the letter x represent the features for optimize.minimize
res = optimize.minimize(cost_function, x0=[0,0,0,random_normal_distribution(),random_normal_distribution(),random_normal_distribution(),random_normal_distribution()])
print("res : ",res)    
print("ssr : ",cost_function(res.x))
print("h : ",forward_propagation(res.x))