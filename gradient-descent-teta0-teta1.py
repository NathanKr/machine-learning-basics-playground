import numpy as np
import matplotlib.pyplot as plt

X=np.array([])
Y=np.array([])
file = open("data\\ex1data1.txt", "r")
for row in file:
  ar = row.split(",")
  X = np.append(X,float(ar[0]))
  Y = np.append(Y,float(ar[1]))

m = X.size # should be same to Y.size

# gradient descent
teta1 = 100 # arbitrary initial condition
teta0 = 100 # arbitrary initial condition
err_gds = 1 # initial condition to enter the while loop
gds_eps = 0.001
iterations = 0
alfa = 0.001 # learning rate

while err_gds > gds_eps:
    H = np.ones(m)*teta0 + X * teta1
    E = H - Y 
    oldteta0 = teta0
    oldteta1 = teta1
    dj_to_dteta0 = (1/m)*np.sum(E)
    dj_to_dteta1 = (1/m)*np.dot(E, X)
    # then assign
    teta0 = teta0 - alfa * dj_to_dteta0
    teta1 = teta1 - alfa * dj_to_dteta1
    err_gds = max(abs(dj_to_dteta0) , abs(dj_to_dteta1))
    iterations += 1
    j = np.dot(E,E)/(2*m)
    if(iterations%100 == 0):
      print(iterations,j)


print("J : {:0.2f}, teta0 : {:0.2f} , teta1 : {:0.2f}".format(j,teta0, teta1))

H = teta0 + teta1 * X
# plots
plt.plot(X, Y,'x',X,H)
plt.xlabel("Population in city in 10,000")
plt.ylabel("Profit in 10,000$")
plt.title('garient descent fit for linear regression')
plt.show()
