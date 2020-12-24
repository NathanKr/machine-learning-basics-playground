import numpy as np

X = np.array([
    [1 , 1],
    [1 , 2],
    [1 , 3],
    [1 , 4],
    [1 , 5]
    ])#5 x 2
Y = np.array([
    [2],  
    [4] , 
    [5] , 
    [4]  ,
     [5]]
     ) #5 x 1

XtX = np.matmul(X.T,X)
XtX_inverse = np.linalg.inv(XtX)
XtX_inverse_Xt = np.matmul(XtX_inverse,X.T)
Teta = np.matmul(XtX_inverse_Xt ,Y)

print("Teta \n" , Teta)