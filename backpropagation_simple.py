import numpy as np
import matplotlib.pyplot as plt
from utils import softplus , dsoftplus_to_dval

# this is based on this https://www.youtube.com/watch?v=IN2XmBhILt4

# this is only estimation
class BackSimple:

    def plug_final_values(self):
        self.b1 = -1.43
        self.b2 = 0.57
        self.b3 = 2.61
        self.w1 = 3.34
        self.w2 = -3.53
        self.w3 = -1.22
        self.w4 = -2.3

    def __init__(self):
        self.x = np.array([0 , 0.5 , 1]) # input dosage : between 0 and 1
        self.y = np.array([0 , 1 , 0]) # output efficacy : 0 or 1

        self.min_step_size = 0.001
        self.learning_rate = 0.01 
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0
        self.z1 = None
        self.z2 = None
        self.a1 = None
        self.a2 = None
        self.z3 = None
        self.z4 = None

    def linear_line(self,x,w,b):
        return w * x + b


    def plot_dataset(self):
        plt.plot(self.x,self.y,'o')
        plt.title('Data set')
        plt.xlabel("Dosage")
        plt.ylabel("Efficacy ")
        plt.grid()
        plt.show()

    def forward_propagation(self):
        self.z1 = self.linear_line(self.x,self.w1,self.b1)
        self.z2 = self.linear_line(self.x,self.w2,self.b2)
        self.a1 = softplus(self.z1)
        self.a2 = softplus(self.z2)
        self.z3 = self.linear_line(self.a1,self.w3,0)
        self.z4 = self.linear_line(self.a2,self.w4,0)
        self.h = self.compute_h()
    
    def compute_h(self):
        return self.z3+self.z4+self.b3

    def dssr_to_dh(self):
        #ssr is sum over (y[i]-h[i])^2 
        #dssr/db3 = (dssr/dh)*(dh/db3)
        residual = self.y-self.compute_h()
        dssr_dh = 2*np.dot(residual,residual)*(-1) # this is sum over (y[i]-h[i])^2 
        return dssr_dh

    # ************ level 2 --> 3 ************ 
    def dh_to_db3(self):    
        # h is z3+z4+b3
        # dh / db3 is 1 because given z3,z4 only b3 is relevant so we have
        return 1

    def dh_to_dz3(self):    
        # h is z3+z4+b3
        # dh / dz3 = 1 
        return 1

    def dh_to_dz4(self):    
            # h is z3+z4+b3
            # dh / dz4 = 1 
            return 1        

    def dz3_to_dw3(self):
        # z3 is a1*w3+0
        # dz3 / dw3 = w3
        return self.w3

    def dz4_to_dw4(self):
            # z3 is a1*w3+0
            # dz4 / dw4 = w4
            return self.w4        

    # ************ level 1 --> 2 ************
    def dz4_to_da2(self): 
            # z4 = a2*w4+0
            # dz4 / da2 = w4
            return self.w4

    def da2_to_dz2(self):
        # a2 = softplus(z2)
        # da2 / dz2 = d_softplus(z2) i.e. derivative of softplus
        return dsoftplus_to_dval(self.z2)
    
    def dz2_to_db2(self):
        # z2 = x * w2 + b2
        # dz2 / db2 is 1
        return 1

    def dz2_to_dw2(self):
        # z2 = x * w2 + b2
        # dz2 / dw2 is x
        return self.x

    def dz3_to_da1(self):
        # z3 = a1*w3+0
        # dz3 / da1 = w3
        return self.w3

    def da1_to_dz1(self):
        # a1 = softplus(z1)
        # da1 / dz1 = d_softplus(z1) i.e. derivative of softplus
        return dsoftplus_to_dval(self.z1)

    def dz1_to_db1(self):
        # z1 = x * w1 + b1
        # dz1 / db1 is 1
        return 1

    def dz1_to_dw1(self):
        # z1 = x * w1 + b1
        # dz1 / db1 is x
        return self.x        

    # ************ derivative of cost - ssr with respect to the features    
    def dssr_to_db3(self):
        return obj.dssr_to_dh()*obj.dh_to_db3()

    def dssr_to_dw3(self):
        return obj.dssr_to_dh()*obj.dh_to_dz3()*obj.dz3_to_dw3()

    def dssr_to_dw4(self):
        return obj.dssr_to_dh()*obj.dh_to_dz4()*obj.dz4_to_dw4()

    def dssr_to_db1(self):            
        return obj.dssr_to_dh()*obj.dh_to_dz3()*obj.dz3_to_da1()*obj.da1_to_dz1()*obj.dz1_to_db1()

    def dssr_to_dw1(self):            
        return obj.dssr_to_dh()*obj.dh_to_dz3()*obj.dz3_to_da1()*obj.da1_to_dz1()*obj.dz1_to_dw1()


    def dssr_to_db2(self):            
        return obj.dssr_to_dh()*obj.dh_to_dz4()*obj.dz4_to_da2()*obj.da2_to_dz2()*obj.dz2_to_db2()

    def dssr_to_dw2(self):            
        return obj.dssr_to_dh()*obj.dh_to_dz4()*obj.dz4_to_da2()*obj.da2_to_dz2()*obj.dz2_to_dw2()

    def gradient_descent_b3(self):
        step_size = 1 # just a value to enter the loop
        self.b3 = 0
        while abs(step_size) > self.min_step_size:
            # ------------ this is the gradient descent engine
            step_size = self.dssr_to_db3() * self.learning_rate
            self.b3 = self.b3 - step_size # i did not see a proof for this


    def sum_square_residuals(self):
        """This is actually the cost function , J by Andrew Ng
        """
        residual = self.y - self.h
        # this sum (y[i]-h[i])^2 over all items or using vector notation (y-h)^2
        ssr = np.dot(residual,residual) 
        return ssr

    def plot_signals(self):
        fig, axs = plt.subplots(4, 2)

        axs[0,0].plot(self.x,self.z1,'o')
        axs[0,0].set_title("z1 vs x")
        axs[0,1].plot(self.x,self.z2,'o')
        axs[0,1].set_title("z2 vs x")
        axs[1,0].plot(self.x,self.a1,'o')
        axs[1,0].set_title("a1 vs x")
        axs[1,1].plot(self.x,self.a2,'o')
        axs[1,1].set_title("a2 vs x")
        axs[2,0].plot(self.x,self.z3,'o')
        axs[2,0].set_title('z3 vs x')
        axs[2,1].plot(self.x,self.z4,'o')
        axs[2,1].set_title('z4 vs x')
        axs[3,0].plot(self.x,self.h,'o')
        axs[3,0].set_title('h vs x')
        axs[3,1].plot(self.x,self.y,'o')
        axs[3,1].set_title('y vs x')
        fig.suptitle('signals')
        plt.tight_layout()
        plt.show()


# main
obj = BackSimple()
# plot_dataset()
obj.plug_final_values()
obj.forward_propagation()
# obj.plot_signals()
obj.gradient_descent_b3()
print("b3 : ",obj.b3)

