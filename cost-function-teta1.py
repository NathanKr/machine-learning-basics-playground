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
J=[]

# mathematically J is (1/2*m)* eT * e
# can i do it without a loop ???
for teta1 in Teta1:
    H = X * teta1
    E = Y - H
    cost = np.dot(E,E)/(2*m)
    J.append(cost)


# compute teta1 for min(J)
min_j_index = np.argmin(J)
teta1 = Teta1[min_j_index]
print("min(J) : {} , tata1 @ min(J) : {}".format(J[min_j_index] , teta1))

h_vec = X * Teta1[min_j_index] # computed linear estimation

# plots
plt.plot(X, Y,'o')
plt.title('data set')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


plt.plot(Teta1,J)
plt.title('loss function J : sum of (Y-H)^2')
plt.xlabel("teta1")
plt.ylabel("J")
plt.show()


plt.plot(X, h_vec,X, Y,'o')
plt.title('linear estimation H = Teta1 * X.    teta1 is chosen as min(J)')
plt.xlabel("X")
plt.ylabel("H , Y")
plt.show()