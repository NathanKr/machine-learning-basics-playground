import numpy as np
import matplotlib.pyplot as plt
m=10
d_teta1=0.01
teta_1_min=-1
teta_1_max=1

# here we actually have a graph with h_teta=teta1*x 
vec_x = np.arange(m)
np.random.seed(42) #use to get the same results every run
rnd = np.random.rand(m)
vec_y = np.arange(m)*0.5 + rnd # actually we have h = teta1*x where teta1 is actually 0.5 

#estimated y -> h_teta1_vec
teta1_vec = np.arange(teta_1_min,teta_1_max,d_teta1)
j_list=[]

# can i do it without a loop ???
for teta1 in teta1_vec:
    h_teta1_vec = vec_x * teta1
    err_vec = h_teta1_vec - vec_y 
    square_err_vec = np.power(err_vec, 2)
    sum_square = np.sum(square_err_vec)
    j_list.append(sum_square)


# compute teta1 for min(J)
min_j_index = np.argmin(j_list)
teta1 = teta1_vec[min_j_index]
print("min(J) : {} , tata1 @ min(J) : {}".format(j_list[min_j_index] , teta1))

h_vec = vec_x * teta1_vec[min_j_index] # computed linear estimation

# gradient descent
teta1_gds = 100 # arbitrary initial condition
err_gds = 1 # initial condition to enter the while loop
gds_eps = 0.001
iterations = 0
alfa = 0.01 # learning rate

while abs(err_gds) > gds_eps:
    h_teta1_vec = vec_x * teta1_gds
    err_vec = h_teta1_vec - vec_y 
    dj_to_dteta1 = (1/m)*np.dot(err_vec, vec_x)
    err_gds = teta1_gds
    teta1_gds = teta1_gds - alfa * dj_to_dteta1
    err_gds = err_gds - teta1_gds
    iterations += 1


print("gradient descent tata1  : {} , iterations : {}".format(teta1_gds,iterations))

