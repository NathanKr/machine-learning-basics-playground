import scipy.io as sio
from os.path import join 
import os
import matplotlib.pyplot as plt
from scipy import optimize
from utils import   cost_function_linear_regression_J , compute_X_with_normalization_for_polynom ,normalize
import numpy as np


# i am using a class  to easyly share variables between functions
class Regularization: 
    def __init__(self):
        self.X1 = None
        self.y = None
        self.X1cv = None
        self.m = None


    def load_dataset(self):
        current_dir = os.path.abspath(".")
        data_dir = join(current_dir, 'data')
        file_name = join(data_dir,"ex5data1.mat")
        mat_dict = sio.loadmat(file_name)
        # not clear why train size is 12 while test\validation is 21 
        # according to Andrew Ng it should be 60% , 20%\20% respectiely
        self.X1 = mat_dict["X"].reshape(-1) # train set 
        self.y = mat_dict["y"].reshape(-1)  # train set
        self.X1test = mat_dict["Xtest"].reshape(-1)# test  set -> not used
        self.ytest = mat_dict["ytest"].reshape(-1) # test  set -> not used
        self.X1cv = mat_dict["Xval"].reshape(-1) # cross validation set
        self.ycv = mat_dict["yval"].reshape(-1) # cross validation set
        self.m = self.X1.size


    def plot_dataset_engine(self,title_text):
        plt.plot(self.X1,self.y,'x',self.X1cv,self.ycv,'.',self.X1test,self.ytest,'s')
        plt.title(title_text)
        plt.xlabel('Change in water level (x)')
        plt.grid()
        plt.ylabel('Water flowing out of the dam (y)')


    def regularization_engine(self,order):
        j_train_regu = []
        j_cv_regu = []
        lamdas = []
        i=1
        num_lamdas = 12
        lamda = 0.001

        while i <= num_lamdas:
            lamdas.append(lamda)
            
            # ------ use polynomial 
            X_train = compute_X_with_normalization_for_polynom(self.X1,order)
            # ------ use linear regression
            res = optimize.minimize(cost_function_linear_regression_J, x0=np.zeros(order+1) , args=(X_train,self.y,lamda))
            j_train_regu.append(res.fun)
            Teta = res.x 

            # cross validation
            X_cv= compute_X_with_normalization_for_polynom(self.X1cv,order)
            j_cv_regu.append(cost_function_linear_regression_J(Teta,X_cv,self.ycv))
            
            # increment index
            i += 1
            lamda *= 2

        # get best lambda
        min_j_index = np.argmin(j_cv_regu)
        best_lambda = lamdas[min_j_index]

        # plot the learning curve
        plt.plot(np.log2(lamdas),j_cv_regu,'x')
        plt.title('lambda vs cross validation error. ploynomial order {}, Jcv : x\n best lambda is : {}'.format(order,best_lambda))
        plt.xlabel('log2(lambda)')
        plt.ylabel('Error')
        plt.grid()
        plt.show()

        # compute lambda for min cross validation cost
        return best_lambda

    def poly_fit(self,order,lamda):
        X = compute_X_with_normalization_for_polynom(self.X1,order)
        # ------ use linear regression
        res = optimize.minimize(cost_function_linear_regression_J, x0=np.zeros(order+1) , args=(X,self.y,lamda))
        Teta = res.x 
        H_linear_regression = np.dot(X,Teta)
        # sort vudu so i can plot line 
        list=zip(*sorted(zip(*(self.X1,H_linear_regression))))
        plt.plot(*list,color='red')
        self.plot_dataset_engine('X,Y as x ,Xcv,ycv as o, Xtest,ytest as square\nh ploynom order {} as red , lamda : {}'.format(order,lamda))
        plt.show()

# main
obj = Regularization()
obj.load_dataset()
order = 8 # hypothesis polynomial order
obj.poly_fit(order,0)
best_lambda_for_cv = obj.regularization_engine(order) 
obj.poly_fit(order,best_lambda_for_cv)






