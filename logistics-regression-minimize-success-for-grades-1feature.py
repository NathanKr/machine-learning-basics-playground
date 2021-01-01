import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from utils import sigmond

# pass
Y = np.array([0	,0	,0	,0	,0	,0	,1	,0	,1	,0	,1	,0	,1	,0	,1	,1	,1	,1	,1	,1])
# hours
X = np.array([0.50	,0.75	,1.00	,1.25	,1.50	,1.75	,1.75	,2.00	,2.25	,2.50	,2.75	,3.00	,3.25	,3.50	,4.00	,4.25	,4.50	,4.75	,5.00	,5.50])
m = X.size

def cost_function_J(Teta):
    H_linear_regression =  Teta[0] + X * Teta[1]
    H = sigmond(H_linear_regression)
    j_vec = Y * np.log(H) + (1-Y)*np.log(1-H)
    J = (-1/m)*np.sum(j_vec)

    return J
    


res = optimize.minimize(cost_function_J, x0=[0,0])
print(res)
teta0 = res.x[0]
teta1 = res.x[1]

teta_x = teta0 + teta1 *X
h = sigmond(teta_x)


plt.scatter(X, Y)
plt.plot(X, h)
plt.title('data set and logistic regression')
plt.ylabel('pass')
plt.xlabel('hours')
plt.grid()
plt.show()

