import numpy as np
import matplotlib.pyplot as plt

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

# gradient descent
Teta=[100 , 100 , 100] # teta0 , teta1, teta2 -> arbitrary initial condition
X = [X0,X1,X2]

teta0 = 100 # arbitrary initial condition
teta1 = 100 # arbitrary initial condition
teta2 = 100 # arbitrary initial condition
iterations = 0
alfa = 0.001 # learning rate
max_iterations = 100000

def sigmond(val):
    return 1/(1+np.exp(-val))

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

# teta0=teta_min[0]
# teta1=teta_min[1]
# teta2=teta_min[2]
print("teta0 : {:0.2f} , teta1 : {:0.2f} ,  teta2 : {:0.2f}".format(teta0, teta1,teta2))

def prob(x1,x2):
    return sigmond(teta0+teta1*x1+teta2*x2)

print("h @ x1=100 , x2=100",prob(100,100))    
print("h @ x1=90 , x2=90",prob(90,90))  
print("h @ x1=80 , x2=80",prob(80,80))   
print("h @ x1=50 , x2=100",prob(50,100))   
print("h @ x1=55 , x2=90",prob(55,90))   
print("h @ x1=40 , x2=100",prob(40,100))   
print("h @ x1=75 , x2=75",prob(75,75))    
print("h @ x1=70 , x2=70",prob(70,70))    
print("h @ x1=70 , x2=50",prob(70,50))    
print("h @ x1=65 , x2=65",prob(65,65))    
print("h @ x1=60 , x2=80",prob(60,80))    
print("h @ x1=60 , x2=70",prob(60,70))    
print("h @ x1=60 , x2=60",prob(60,60))    
print("h @ x1=50 , x2=50",prob(50,50))    
print("h @ x1=40 , x2=40",prob(40,40))    



plt.plot(j_list)
plt.title('J not nan')
plt.show()

plt.plot(X1_admit,X2_admit,'+',X1_not_admit,X2_not_admit,'o')
plt.title('admit : + , not admit : o')
plt.xlabel("X1 - grade1")
plt.ylabel("X2 - grade2")
plt.show()
