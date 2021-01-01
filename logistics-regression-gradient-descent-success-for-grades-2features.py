import numpy as np
import matplotlib.pyplot as plt
from utils import sigmond


# ****************************
#       This dataset is problematic 
#       The dataset is from Andrew Ng course but he did not used here gradient descent instead he used 
#       i am getting nan on J for some iterations
#       plotting the j one can see that it is very noisy
#       it is very sensitive to max_iteration and alfa
#       --------> not clear why
# ****************************



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

X1_admit = []
X2_admit = []
X1_not_admit = []
X2_not_admit = []

i = 0
while i < m:
    if(Y[i] == 1):
        X1_admit.append(int(X1[i]))
        X2_admit.append(int(X2[i]))
    else:
        X1_not_admit.append(int(X1[i]))
        X2_not_admit.append(int(X2[i]))
    i += 1        


scale_factor = 100

# using scale_factor of 1 will cause an error "Desired error not necessarily achieved due to precision loss." 
# the problem is that X0 is 1 while X1,X2 are between 0 - 100 so 100 scale them to be around 1

X1 = X1/scale_factor
X2 = X2/scale_factor

teta0 = 100 # arbitrary initial condition
teta1 = 100 # arbitrary initial condition
teta2 = 100 # arbitrary initial condition
iterations = 0
alfa = 0.01 # learning rate
max_iterations = 500


# todo nath using matrix might be much easier !!!!!!

j_list=[]
j_min=float('inf')

while iterations <  max_iterations:
    # this is a linear hypotesis 
    H_linear_regression = X0 * teta0 + X1 * teta1 + X2 * teta2
    H = sigmond(H_linear_regression)
    E = H - Y 
    dj_to_dteta0 = np.sum(E)
    dj_to_dteta1 = np.dot(E, X1)
    dj_to_dteta2 = np.dot(E, X2)

    # then assign
    teta0 = teta0 - alfa * dj_to_dteta0
    teta1 = teta1 - alfa * dj_to_dteta1
    teta2 = teta2 - alfa * dj_to_dteta2

    j_vec = Y * np.log(H) + (1-Y)*np.log(1-H)
    J = (-1/m)*np.sum(j_vec)
    iterations += 1
    if(iterations%10 == 0):
        if(not np.isnan(J)):
            print("iterations : {} , J : {:0.2f}".format(iterations,J))
            j_list.append(J)
            if(J < j_min):
                teta_min=[teta0 , teta1, teta2]
                j_min=J

print("teta0 : {:0.2f} , teta1 : {:0.2f} ,  teta2 : {:0.2f}".format(teta0, teta1,teta2))

def prob(x1,x2):
    return sigmond(teta0+teta1*x1/scale_factor+teta2*x2/scale_factor)

print("h @ x1=90 , x2=90",prob(90,90))  
# plt.plot(j_list)
# plt.title('J not nan')
# plt.show()

# plt.plot(X1_admit,X2_admit,'+',X1_not_admit,X2_not_admit,'o')
# plt.title('admit : + , not admit : o')
# plt.xlabel("X1 - grade1")
# plt.ylabel("X2 - grade2")
# plt.show()

ar_index_pass = np.where(Y == 1)
ar_index_fail = np.where(Y == 0)

x2_0_5_line = -(teta0+teta1*X1)/teta2 # prob 0.5


plt.title('+ : pass , o : fail \ncomputed gradient descent 0.5 logistic regression probability line')
plt.xlabel("grade1")
plt.ylabel("grade2")
plt.plot(scale_factor*X1[ar_index_pass],scale_factor*X2[ar_index_pass],'+')
plt.plot(scale_factor*X1[ar_index_fail],scale_factor*X2[ar_index_fail],'o')
plt.plot(scale_factor*X1,scale_factor*x2_0_5_line) 
plt.show()


