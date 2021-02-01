import numpy as np
import math

# gradient_descent_logistic_regresion
# Teta : 
#   - is features vector : teta_0,teta_1,....,teta_n . 
#   - Teta is column vector (n+1)x1 (n+1 is the number of features)
# X :
#   - a matrix [X0,X1,X2,....,Xn]. 
#   - each X0,X1,X2,....,Xn is a column vector mx1 
#   - m is number of data set points. 
#   - X is mx(n+1). 
#   - Notice that X0 is always 1
# X*Teta :
#   - is : teta0*X0 + teta1*X1 + ….. +teta_n*Xn 
#   - X*Teta is a column vector mx1 
def gradient_descent_logistic_regresion (Teta,X,Y,alfa):
  m = Y.size
  H_linear_regression =  np.dot(X,Teta)
  H = sigmoid(H_linear_regression)
  E = H - Y 
  dJ_to_dTeta = np.dot(E, X)
  Teta = Teta - alfa * dJ_to_dTeta
  j_vec = Y * np.log(H) + (1-Y)*np.log(1-H)
  J = (-1/m)*np.sum(j_vec)
  return {"Teta" : Teta , "J" : J}



# gradient_descent_linear_regresion
# Teta : 
#   - is features vector : teta_0,teta_1,....,teta_n . 
#   - Teta is column vector (n+1)x1 (n+1 is the number of features)
# X :
#   - a matrix [X0,X1,X2,....,Xn]. 
#   - each X0,X1,X2,....,Xn is a column vector mx1 
#   - m is number of data set points. 
#   - X is mx(n+1). 
#   - Notice that X0 is always 1
# X*Teta :
#   - is : teta0*X0 + teta1*X1 + ….. +teta_n*Xn 
#   - X*Teta is a column vector mx1 
def gradient_descent_linear_regresion (Teta,X,Y,alfa):
  m = Y.size
  H_linear_regression =  np.dot(X,Teta)
  E = H_linear_regression - Y
  dJ_to_dTeta = (1/m)*np.dot(E, X)
  Teta = Teta - alfa * dJ_to_dTeta
  J = np.dot(E,E)/(2*m)
  return {"Teta" :  Teta , "J" : J}

# cost_function_logistic_regression_J
# Teta : 
#   - is features vector : teta_0,teta_1,....,teta_n . 
#   - Teta is column vector (n+1)x1 (n+1 is the number of features)
# X :
#   - a matrix [X0,X1,X2,....,Xn]. 
#   - each X0,X1,X2,....,Xn is a column vector mx1 
#   - m is number of data set points. 
#   - X is mx(n+1). 
#   - Notice that X0 is always 1
# X*Teta :
#   - is : teta0*X0 + teta1*X1 + ….. +teta_n*Xn 
#   - X*Teta is a column vector mx1 
def cost_function_logistic_regression_J(Teta,X,Y):
    m = Y.size
    H_linear_regression =  np.dot(X,Teta)
    H = sigmoid(H_linear_regression)
    j_vec = Y * np.log(H) + (1-Y)*np.log(1-H)
    J = (-1/m)*np.sum(j_vec)
    return J


# cost_function_linear_regression_J
# Teta : 
#   - is features vector : teta_0,teta_1,....,teta_n . 
#   - Teta is column vector (n+1)x1 (n+1 is the number of features)
# X :
#   - a matrix [X0,X1,X2,....,Xn]. 
#   - each X0,X1,X2,....,Xn is a column vector mx1 
#   - m is number of data set points. 
#   - X is mx(n+1). 
#   - Notice that X0 is always 1
# X*Teta :
#   - is : teta0*X0 + teta1*X1 + ….. +teta_n*Xn 
#   - X*Teta is a column vector mx1 
# lamda :
#   - is a scalar which is regularization parameter used for over fitting
#   - by default there is no regularization thus lambda is zero
def cost_function_linear_regression_J(Teta,X,Y,lamda=0):
    m = Y.size
    H_linear_regression =  np.dot(X,Teta)
    E = Y - H_linear_regression
    J = np.dot(E,E)/(2*m)
    if (lamda > 0):
      # regulrization does not affect Teta[0]
      J += lamda*np.dot(Teta[1:],Teta[1:])/(2*m)
    
    return J    


def softplus(val):
    return np.log(1+np.exp(val))

# it turns out that the derivative to softplus is exactly sigmoid 
# check here(https://medium.com/@abhinavr8/activation-functions-neural-networks-66220238e1ff#:~:text=Softplus%20function%3A%20f(x),also%20called%20the%20logistic%20function.)
def dsoftplus_to_dval(val):
    return sigmoid(val)
    

def sigmoid(val):
    return 1/(1+np.exp(-val))

# return 0 or 1
def sigmoid_binari(val):
    sig = 1/(1+np.exp(-val))
    return 1 if sig >= 0.5 else 0    

# return 0 or 1
def neuron(Teta,X):
  val = np.dot(Teta,X)
  return sigmoid_binari(val)

# a, b : 0,1
def logical_and(a,b):
  Teta = np.array([-30 , 20 , 20 ])
  X = np.array([1 , a , b ])
  return neuron(Teta,X)



# compute_X_with_normalization_for_polynom
# e.g. for order 2 X before normalization is [X0,X1,np.power(X1,2)]
# normalization is done for all Xi (i != 0)
#
# X1    :
#   - column vector mx1
# order : 
#   - order of the polynom 
#   - order must be > 1
# X     :
#   - a matrix [X0,X1,X2,....,Xn]
#   - each X0,X1,X2,....,Xn is a column vector mx1 
#   - m is number of data set points. 
#   - X is mx(n+1)
#   - Notice that X0 is always 1
def compute_X_with_normalization_for_polynom(X1,order):
  if(order < 1):
    raise Exception("order must be > 1")

  m = X1.size 
  X0 = np.ones(m)
  X= np.vstack((X0,normalize(X1)))

  temp_pow = 2
  while (temp_pow <= order):
    Xi = np.power(X1,temp_pow)
    Xi = normalize(Xi)
    X= np.vstack((X,Xi))
    temp_pow += 1

  return X.T    

# normalize a 1d array in general to -0.5 : 0.5
def normalize(v1d):
  v1d_range = np.amax(v1d) - np.amin(v1d)
  v1d_mean = np.mean(v1d)
  return (v1d-v1d_mean)/v1d_range if v1d_range > 0 else v1d

def normal_dist(sample_x,x):
    mean_x = np.mean(x)
    variance_x = np.var(x) 
    return normal_dist_short(sample_x,mean_x,variance_x)


def normal_dist_short(sample_x,mean_x,variance_x):
    delta = sample_x - mean_x
    p = (1/math.sqrt(2*math.pi*variance_x))*math.exp(-math.pow(delta,2)/(2*variance_x))
    return p

# actual and estimated are 1
def true_positive(actual,estimated):
  mul = actual * estimated
  return np.sum(mul) # sum 1

# actual is 0 but estimated is 1
def false_positive(actual,estimated): 
  ar_index_estimated_1 = np.where(estimated == 1)[0]
  actual_slice = actual[ar_index_estimated_1]
  actual_slice_0 = np.where(actual_slice == 0)[0]
  num_actual_slice_0 = len(actual_slice_0)
  return num_actual_slice_0

# actual is 1 but estimated is 0
def false_negative(actual,estimated): 
  ar_index_actual_1 = np.where(actual == 1)[0]
  estimated_slice = estimated[ar_index_actual_1]
  estimated_slice_0 = np.where(estimated_slice == 0)[0]
  num_estimated_slice_0 = len(estimated_slice_0)
  return num_estimated_slice_0
  

def precision(actual,estimated):
    tp = true_positive(actual,estimated)
    fp = false_positive(actual,estimated)
    return tp/(tp+fp)

def recall(actual,estimated):
    tp = true_positive(actual,estimated)
    fn = false_negative(actual,estimated)
    return tp /(tp + fn)



def F1score(actual,estimated):    
    prec = precision(actual,estimated)
    rec = recall(actual,estimated)
    return 2*prec*rec/(prec+rec)

def symetric_random(val_max):
  """[summary]
  get a ranome number [-val_max , val_max]

  Args:
      val_max (number): [description]

  Returns:
      [type]: [description]
  """
  return np.random.rand()*2*val_max - val_max 