import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from utils import sigmond , cost_function_logistic_regression_J


X1=np.array([]) # grade 1
X2=np.array([]) # grade 2
Y=np.array([]) # admitted to university
file = open("data\\ex2data1.txt", "r")
for row in file:
  ar = row.split(",")
  X1 = np.append(X1,float(ar[0])) 
  X2 = np.append(X2,float(ar[1])) 
  Y = np.append(Y,float(ar[2]))   

scale_factor = 100

# using scale_factor of 1 will cause an error "Desired error not necessarily achieved due to precision loss." 
# the problem is that X0 is 1 while X1,X2 are between 0 - 100 so 100 scale them to be around 1

X1 = X1/scale_factor
X2 = X2/scale_factor

m = Y.size # should be same as the size of X1,X2
X0= np.ones(m)


X= np.vstack((X0,X1,X2)).T
res = optimize.minimize(cost_function_logistic_regression_J, x0=[0,0,0], args=(X,Y))
print(res)

[teta0,teta1,teta2] = res.x

def pass_prob(x1,x2):
  teta_x = teta0 + teta1*x1/scale_factor + teta2*x2/scale_factor
  return sigmond(teta_x)

print("pass_prob(95,95) : ",pass_prob(95,95))

ar_index_pass = np.where(Y == 1)
ar_index_fail = np.where(Y == 0)

x2_0_5_line = -(teta0+teta1*X1)/teta2 # prob 0.5


plt.title('+ : pass , o : fail \ncomputed optimize.minimize 0.5 logistic regression probability line')
plt.xlabel("grade1")
plt.ylabel("grade2")
plt.plot(scale_factor*X1[ar_index_pass],scale_factor*X2[ar_index_pass],'+')
plt.plot(scale_factor*X1[ar_index_fail],scale_factor*X2[ar_index_fail],'o')
plt.plot(scale_factor*X1,scale_factor*x2_0_5_line) 
plt.show()


