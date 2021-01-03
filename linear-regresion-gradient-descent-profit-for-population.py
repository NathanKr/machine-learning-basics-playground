import numpy as np
import matplotlib.pyplot as plt
from utils import gradient_descent_linear_regresion 

X1=np.array([])
Y=np.array([])
file = open("data\\ex1data1.txt", "r")
for row in file:
  ar = row.split(",")
  X1 = np.append(X1,float(ar[0]))
  Y = np.append(Y,float(ar[1]))

m = X1.size # should be same to Y.size

iterations = 0
alfa = 0.001 # learning rate
max_iterations = 40000

Teta = np.array([100,100])
X0 = np.ones(m)
X= np.vstack((X0,X1)).T

while iterations <  max_iterations:
  res = gradient_descent_linear_regresion(Teta,X,Y,alfa)
  Teta = res[0]
  j = res[1]
  iterations += 1
  if(iterations%100 == 0):
      print(iterations,j)

print("J : {:0.2f}, teta0 : {:0.2f} , teta1 : {:0.2f}".format(j,Teta[0], Teta[1]))

H = Teta[0] + Teta[1]*X1 

# plots
plt.plot(X1, Y,'x',X1,H)
plt.xlabel("Population in city in 10,000")
plt.ylabel("Profit in 10,000$")
plt.title('garient descent fit for linear regression')
plt.show()
