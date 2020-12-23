import numpy as np
import matplotlib.pyplot as plt
m=10
d_teta1=0.01
teta_1_min=-1
teta_1_max=1

# here we actually have a graph with h_teta=teta1*x 
X = np.arange(m)
np.random.seed(42) #use to get the same results every run
rnd = np.random.rand(m)
Y = np.arange(m)*0.5 + rnd # actually we have h = teta1*x where teta1 is actually 0.5 

#estimated y -> H
Teta1 = np.arange(teta_1_min,teta_1_max,d_teta1)
# J is not required for comuting here gradient descent 
# J is used here just to get min(J) as a comparison for me
J=[] 

# can i do it without a loop ???
for teta1 in Teta1:
    H = X * teta1
    E = H - Y 
    cost = np.dot(E,E)/(2*m)
    J.append(cost)


# compute teta1 for min(J)
min_j_index = np.argmin(J)
teta1 = Teta1[min_j_index]

h_vec = X * Teta1[min_j_index] # computed linear estimation

# gradient descent
teta1_gds = 100 # arbitrary initial condition
err_gds = 1 # initial condition to enter the while loop
gds_eps = 0.001
iterations = 0
alfa = 0.001 # learning rate

while err_gds > gds_eps:
    H = X * teta1_gds
    E = H - Y 
    dj_to_dteta1 = (1/m)*np.dot(E, X)
    err_gds = abs(dj_to_dteta1)
    teta1_gds = teta1_gds - alfa * dj_to_dteta1
    iterations += 1
    j = np.dot(E,E)/(2*m)
    print(iterations,j)


print("min(J) : {:.3f} , tata1 @ min(J) : {:.3f}".format(J[min_j_index] , teta1))
print("gradient descent tata1  : {:.3f} , iterations : {:.3f}".format(teta1_gds,iterations))

