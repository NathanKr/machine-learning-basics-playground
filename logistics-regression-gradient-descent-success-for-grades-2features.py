import numpy as np
import matplotlib.pyplot as plt
from utils import sigmoid , gradient_descent_logistic_regresion

X1=np.array([]) # grade 1
X2=np.array([]) # grade 2
Y=np.array([]) # admitted to university
file = open("data\\ex2data1.txt", "r")
for row in file:
  ar = row.split(",")
  X1 = np.append(X1,float(ar[0])) 
  X2 = np.append(X2,float(ar[1])) 
  Y = np.append(Y,float(ar[2]))   

m = Y.size # should be same as the size of X1,X2
X0 = np.ones(m) 

scale_factor = 100

# using scale_factor of 1 will cause an error "Desired error not necessarily achieved due to precision loss." 
# the problem is that X0 is 1 while X1,X2 are between 0 - 100 so 100 scale them to be around 1
X1 = X1/scale_factor
X2 = X2/scale_factor
X= np.vstack((X0,X1,X2)).T
Teta = [100,100,100] # arbitrary initial condition

iterations = 0
alfa = 0.01 # learning rate
max_iterations = 1000
    
while iterations <  max_iterations:
    res = gradient_descent_logistic_regresion(Teta,X,Y,alfa)
    Teta = res["Teta"]
    J = res["J"]
    iterations += 1

print("J : {:0.2f} , teta0 : {:0.2f} , teta1 : {:0.2f} ,  teta2 : {:0.2f}".format(J,Teta[0], Teta[1],Teta[2]))

def prob(x1,x2):
    return sigmoid(Teta[0]+Teta[1]*x1/scale_factor+Teta[2]*x2/scale_factor)

print("h @ x1=90 , x2=90",prob(90,90))  


ar_index_pass = np.where(Y == 1)
ar_index_fail = np.where(Y == 0)

x2_0_5_line = -(Teta[0]+Teta[1]*X1)/Teta[2] # prob 0.5

plt.title('+ : pass , o : fail \ncomputed gradient descent 0.5 logistic regression probability line')
plt.xlabel("grade1")
plt.ylabel("grade2")
plt.plot(scale_factor*X1[ar_index_pass],scale_factor*X2[ar_index_pass],'+')
plt.plot(scale_factor*X1[ar_index_fail],scale_factor*X2[ar_index_fail],'o')
plt.plot(scale_factor*X1,scale_factor*x2_0_5_line) 
plt.show()


