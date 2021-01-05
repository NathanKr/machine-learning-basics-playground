import scipy.io as sio
from os.path import join 
import os
import matplotlib.pyplot as plt
from scipy import optimize
from utils import cost_function_linear_regression_J , compute_X_with_normalization_for_polynom
import numpy as np


current_dir = os.path.abspath(".")
data_dir = join(current_dir, 'data')
file_name = join(data_dir,"ex5data1.mat")
mat_dict = sio.loadmat(file_name)
X1 = mat_dict["X"].reshape(-1)
y = mat_dict["y"].reshape(-1)
X1test = mat_dict["Xtest"].reshape(-1)
ytest = mat_dict["ytest"].reshape(-1)
Xval = mat_dict["Xval"].reshape(-1)
yval = mat_dict["yval"].reshape(-1)


m = X1.size
X0 = np.ones(m)

# ------ use 8 order polynomial
order = 8
X = compute_X_with_normalization_for_polynom(X1,order) #X is mx(n+1)
res = optimize.minimize(cost_function_linear_regression_J, x0=np.zeros(order+1) , args=(X,y))
print(res)
Teta = res.x #Teta is (n+1)*1
h_linear_regression_high_order = np.dot(X,Teta)
plt.plot(X1,y,'x',X1,h_linear_regression_high_order,'o')
plt.title('data set and {} order linear regression for X,Y using optimize.minimize\noverfitting is evidance !!!'.format(order))
plt.xlabel('Change in water level (x)')
plt.grid()
plt.ylabel('Water flowing out of the dam (y)')
plt.show()


# ------ use 1 order polynomial i.e. linear regression
X= np.vstack((X0,X1)).T #X is mx(n+1)
res = optimize.minimize(cost_function_linear_regression_J, x0=[0,0] , args=(X,y))
print(res)
Teta = res.x #Teta is (n+1)*1
h_linear_regression = np.dot(X,Teta)
# plt.plot(X1,y,'x',X1,h_linear_regression,X1test,ytest,'+',Xval,yval,'.')
# plt.title('data set and linear regression for X,Y using optimize.minimize\nX,Y as x , Xtest,ytest as + , Xval,yval as .')
plt.plot(X1,y,'x',X1,h_linear_regression)
plt.title('data set and linear regression for X,Y using optimize.minimize\nunderfitting is evidance !!!')
plt.xlabel('Change in water level (x)')
plt.grid()
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

