import numpy as np
import matplotlib.pyplot as plt
from utils import softplus

# this is based on this https://www.youtube.com/watch?v=IN2XmBhILt4

# this is only estimation
class BackSimple:
    def __init__(self):
        self.x = np.array([0 , 0.5 , 1]) # input dosage : between 0 and 1
        self.xy = np.array([0 , 1 , 0]) # output efficacy : 0 or 1

        self.b1 = -1.43
        self.b2 = 0.57
        self.b3 = 2.61
        self.w1 = 3.34
        self.w2 = -3.53
        self.w3 = -1.22
        self.w4 = -2.3
        self.h = None
        self.z1 = None
        self.z2 = None
        self.a1 = None
        self.a2 = None
        self.z3 = None
        self.z4 = None
        self.y = None

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
        self.h = self.z3+self.z4+self.b3

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
        fig.suptitle('signals')
        plt.tight_layout()
        plt.show()


# main
obj = BackSimple()
# plot_dataset()
obj.forward_propagation()
obj.plot_signals()

