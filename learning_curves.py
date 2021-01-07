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
X1 = mat_dict["X"].reshape(-1) # train set , not clear why train size is 12 while test\validation is 21
y = mat_dict["y"].reshape(-1)  # train set
X1test = mat_dict["Xtest"].reshape(-1)# test  set
ytest = mat_dict["ytest"].reshape(-1) # test  set
X1cv = mat_dict["Xval"].reshape(-1) # cross validation set
ycv = mat_dict["yval"].reshape(-1) # cross validation set

m = X1.size
X0 = np.ones(m)


# plots 
# plt.plot(X1,y,'x',X1test,ytest,'+',Xval,yval,'.')
# plt.title('data set X,Y as x , Xtest,ytest as + , Xval,yval as .')
# plt.xlabel('Change in water level (x)')
# plt.grid()
# plt.ylabel('Water flowing out of the dam (y)')
# plt.show()

j_train_lc = []
j_cv_lc = []
i_lc = []
i=2
while i < m:
    X0_lc = X0[:i]
    i_lc.append(i)

    # --- train
    X1_train_lc = X1[:i]
    y_train_lc  = y[:i] 
    # ------ use 1 order polynomial i.e. linear regression
    X_train_lc= np.vstack((X0_lc,X1_train_lc)).T 
    res = optimize.minimize(cost_function_linear_regression_J, x0=[0,0] , args=(X_train_lc,y_train_lc))
    j_train_lc.append(res.fun)
    Teta = res.x 

    # cross validation
    X1_cv_lc = X1cv[:i]
    y_cv_lc  = ycv[:i] 
    X_cv_lc= np.vstack((X0_lc,X1_cv_lc)).T 
    j_cv_lc.append(cost_function_linear_regression_J(Teta,X_cv_lc,y_cv_lc))

    i += 1


plt.plot(i_lc,j_train_lc,'o',i_lc,j_cv_lc,'x')
plt.title('ploynomial order 1 - high bias, Jtrain : o , Jcv : x')
plt.xlabel('number of training samples')
plt.ylabel('Error')
plt.grid()
plt.show()





# ------ use 8 order polynomial
# order = 8
# X = compute_X_with_normalization_for_polynom(X1,order) #X is mx(n+1)
# res = optimize.minimize(cost_function_linear_regression_J, x0=np.zeros(order+1) , args=(X,y))
# print(res.fun)


