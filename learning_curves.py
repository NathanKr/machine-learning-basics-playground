import scipy.io as sio
from os.path import join 
import os
import matplotlib.pyplot as plt
from scipy import optimize
from utils import   cost_function_linear_regression_J , compute_X_with_normalization_for_polynom ,normalize
import numpy as np

# i am using a class  to easyly share variables between functions
class LearningCurves: 
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
        self.X1 = mat_dict["X"].reshape(-1) # train set , not clear why train size is 12 while test\validation is 21
        self.y = mat_dict["y"].reshape(-1)  # train set
        # X1test = mat_dict["Xtest"].reshape(-1)# test  set
        # ytest = mat_dict["ytest"].reshape(-1) # test  set
        self.X1cv = mat_dict["Xval"].reshape(-1) # cross validation set
        self.ycv = mat_dict["yval"].reshape(-1) # cross validation set
        self.m = self.X1.size


    def plot_dataset_engine(self,title_text):
        plt.plot(self.X1,self.y,'x',self.X1cv,self.ycv,'.')
        plt.title(title_text)
        plt.xlabel('Change in water level (x)')
        plt.grid()
        plt.ylabel('Water flowing out of the dam (y)')

    # plots 
    def plot_dataset(self,title_text):
        self.plot_dataset_engine(title_text)
        plt.show()


    def learning_curves(self,order , error_type):
        j_train_lc = []
        j_cv_lc = []
        i_lc = []
        i=1
        while i < self.m:
            i_lc.append(i)

            # --- train
            X1_train_lc = self.X1[:i]
            y_train_lc  = self.y[:i] 
            
            # ------ use polynomial 
            X_train_lc = compute_X_with_normalization_for_polynom(X1_train_lc,order)
            # ------ use linear regression
            res = optimize.minimize(cost_function_linear_regression_J, x0=np.zeros(order+1) , args=(X_train_lc,y_train_lc))
            j_train_lc.append(res.fun)
            Teta = res.x 

            # cross validation
            X1_cv_lc = self.X1cv[:i]
            y_cv_lc  = self.ycv[:i] 
            X_cv_lc= compute_X_with_normalization_for_polynom(X1_cv_lc,order)
            j_cv_lc.append(cost_function_linear_regression_J(Teta,X_cv_lc,y_cv_lc))
            
            # increment index
            i += 1

        # plot the learning curve
        plt.plot(i_lc,j_train_lc,'.',i_lc,j_cv_lc,'x')
        plt.title('Learning curve : ploynomial order {} - {}, Jtrain : o , Jcv : x'.format(order,error_type))
        plt.xlabel('number of training samples')
        plt.ylabel('Error')
        plt.grid()
        plt.show()

    def poly_fit(self,order):
        X = compute_X_with_normalization_for_polynom(self.X1,order)
        # ------ use linear regression
        res = optimize.minimize(cost_function_linear_regression_J, x0=np.zeros(order+1) , args=(X,self.y))
        Teta = res.x 
        H_linear_regression = np.dot(X,Teta)
        # sort vudu so i can plot line 
        list=zip(*sorted(zip(*(self.X1,H_linear_regression))))
        plt.plot(*list,color='red')
        self.plot_dataset_engine('data set X,Y as x ,  Xcv,ycv as o, ploynom order {} as red'.format(order))
        plt.show()

# main
obj = LearningCurves()
obj.load_dataset()
obj.plot_dataset('data set X,Y as x ,  Xcv,ycv as o')
obj.poly_fit(1)
obj.poly_fit(8)
obj.learning_curves(1,"high bias") # 1 order polynomial
obj.learning_curves(8,"high variance") # 8 order polynomial






