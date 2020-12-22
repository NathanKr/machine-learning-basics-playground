import numpy as np
import matplotlib.pyplot as plt
len=10
d_teta1=0.01
teta_1_min=-1
teta_1_max=1

# here we actually have a graph with h_teta=teta1*x 
vec_x = np.arange(len)
np.random.seed(42) #use to get the same results every run
rnd = np.random.rand(len)
vec_y = np.arange(len)*0.5 + rnd # actually we have h = teta1*x where teta1 is actually 0.5 

#estimated y -> h_teta1_vec
teta1_vec = np.arange(teta_1_min,teta_1_max,d_teta1)
j_list=[]

# can i do it without a loop ???
for teta1 in teta1_vec:
    h_teta1_vec = vec_x * teta1
    err_vec = vec_y - h_teta1_vec
    square_err_vec = np.power(err_vec, 2)
    sum_square = np.sum(square_err_vec)
    j_list.append(sum_square)


# compute teta1 for min(J)
min_j_index = np.argmin(j_list)
teta1 = teta1_vec[min_j_index]
print("min(J) : {} , tata1 @ min(J) : {}".format(j_list[min_j_index] , teta1))

h_vec = vec_x * teta1_vec[min_j_index] # computed linear estimation

# plots
plt.plot(vec_x, vec_y,'o')
plt.title('data set')
plt.xlabel("x")
plt.ylabel("y")
plt.show()


plt.plot(teta1_vec,j_list)
plt.title('loss function J : sum of (y-h)^2')
plt.xlabel("teta1")
plt.ylabel("J")
plt.show()


plt.plot(vec_x, h_vec,vec_x, vec_y,'o')
plt.title('linear estimation h = teta1 * x.    teta1 is chosen as min(J)')
plt.xlabel("x")
plt.ylabel("h , y")
plt.show()