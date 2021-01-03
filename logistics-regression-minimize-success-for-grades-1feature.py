import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from utils import sigmond , cost_function_logistic_regression_J

# pass
Y = np.array([0	,0	,0	,0	,0	,0	,1	,0	,1	,0	,1	,0	,1	,0	,1	,1	,1	,1	,1	,1])
# hours
X1 = np.array([0.50	,0.75	,1.00	,1.25	,1.50	,1.75	,1.75	,2.00	,2.25	,2.50	,2.75	,3.00	,3.25	,3.50	,4.00	,4.25	,4.50	,4.75	,5.00	,5.50])
m = X1.size
X0 = np.ones(m)
X= np.vstack((X0,X1)).T


res = optimize.minimize(cost_function_logistic_regression_J, x0=[0,0] , args=(X,Y))
print(res)
teta0 = res.x[0]
teta1 = res.x[1]

teta_x = teta0 + teta1 *X1
h = sigmond(teta_x)


plt.scatter(X1, Y)
plt.plot(X1, h)
plt.title('data set and logistic regression using optimize.minimize')
plt.ylabel('pass')
plt.xlabel('hours')
plt.grid()
plt.show()

