import numpy as np
import matplotlib.pyplot as plt
from utils import softplus , dsoftplus_to_dval , symetric_random
import sys

# this is based on this https://www.youtube.com/watch?v=IN2XmBhILt4

class BackSimple:

    def print_features(self):
        print("b1 : {}\nb2 : {}\nb3 : {}\nw1 : {}\nw2 : {}\nw3 : {}\nw4 : {}\n".format(self.b1 , self.b2, self.b3 , self.w1 , self.w2 , self.w3 , self.w4))

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
        self.m = self.y.size # number of data set points

        self.MIN_STEP_SIZE = 0.0001
        self.LEARNING_RATE = 0.001 # alfa
        self.MAX_ITERATIONS = 500
        self.NUMERICAL_DERIVATIVE_EPS = 0.001

        self.feature_names = ["b1" , "b2" , "b3" , "w1" , "w2" , "w3" , "w4"]
        self.n = len(self.feature_names) # number of features
        self.feature_derivatives = [self.dssr_to_db1 , self.dssr_to_db2 , self.dssr_to_db3 , self.dssr_to_dw1 , self.dssr_to_dw2 ,self.dssr_to_dw3 , self.dssr_to_dw4]


        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.w1 = None
        self.w2 = None
        self.w3 = None
        self.w4 = None
        self.z1 = None
        self.z2 = None
        self.a1 = None
        self.a2 = None
        self.z3 = None
        self.z4 = None
        self.steps = None
        self.current_num_iterations = None

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
        # layer 1 -> layer 2
        self.z1 = self.linear_line(self.x,self.w1,self.b1)
        self.z2 = self.linear_line(self.x,self.w2,self.b2)
        self.a1 = softplus(self.z1)
        self.a2 = softplus(self.z2)
        # layer 2 -> layer 3
        self.z3 = self.linear_line(self.a1,self.w3,0)
        self.z4 = self.linear_line(self.a2,self.w4,0)
        self.h  = self.z3+self.z4+self.b3
    

    def dssr_to_dh(self):
        #ssr is sum over (y[i]-h[i])^2 
        #dssr/db3 = (dssr/dh)*(dh/db3)
        residual = self.y-self.h
        # dssr_dh = 2*np.sum(residual)*(-1) # -> no sum because it is done allready on the cost to feature derivatives
        dssr_dh = 2*residual*(-1)
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

    def dz3_to_da1(self):
        # z3 = a1*w3+0
        # dz3 / da1 = w3
        return self.w3

    def dz3_to_dw3(self):
        # z3 is a1*w3+0
        # dz3 / dw3 = w3
        return self.a1

    def dz4_to_da2(self): 
        # z4 = a2*w4+0
        # dz4 / da2 = w4
        return self.w4

    def dz4_to_dw4(self):
        # z4 is a2*w4+0
        # dz4 / dw4 = w4
        return self.a2        


    # ************ level 1 --> 2 ************

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
        # dz1 / dw1 is x
        return self.x        

    # ************ derivative of cost - ssr with respect to the features    
    def dssr_to_db3(self):
        # sum over all data set
        return np.sum(obj.dssr_to_dh()*obj.dh_to_db3())

    def dssr_to_dw3(self):
        # sum over all data set
        return np.sum(obj.dssr_to_dh()*obj.dh_to_dz3()*obj.dz3_to_dw3())

    def dssr_to_dw4(self):
        # sum over all data set
        return np.sum(obj.dssr_to_dh()*obj.dh_to_dz4()*obj.dz4_to_dw4())

    def dssr_to_db1(self):            
        # sum over all data set
        return np.sum(obj.dssr_to_dh()*obj.dh_to_dz3()*obj.dz3_to_da1()*obj.da1_to_dz1()*obj.dz1_to_db1())

    def dssr_to_dw1(self):            
        # sum over all data set
        return np.sum(obj.dssr_to_dh()*obj.dh_to_dz3()*obj.dz3_to_da1()*obj.da1_to_dz1()*obj.dz1_to_dw1())


    def dssr_to_db2(self):            
        # sum over all data set
        return np.sum(obj.dssr_to_dh()*obj.dh_to_dz4()*obj.dz4_to_da2()*obj.da2_to_dz2()*obj.dz2_to_db2())

    def dssr_to_dw2(self):            
        # sum over all data set
        return np.sum(obj.dssr_to_dh()*obj.dh_to_dz4()*obj.dz4_to_da2()*obj.da2_to_dz2()*obj.dz2_to_dw2())


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

    def compute_step(self,derivative):
        step = self.LEARNING_RATE * derivative
        print("step : ",step)
        self.steps.append(step)
        return step

    def random_normal_distribution(self):
        return np.random.normal()

    def compute_initial_value_for_all_the_features(self):
        # random weight and zero bias
        # following is suggested by StatsQuest
        # self.b1 = self.b2 = self.b3 = self.w1 = 0
        # self.w2 = self.random_normal_distribution()
        # self.w3 = self.random_normal_distribution()
        # self.w4 = self.random_normal_distribution()
        val_max = 0.5
        # following is suggested by Andrew Ng
        self.b1 = symetric_random(val_max)
        self.b2 = symetric_random(val_max)
        self.b3 = symetric_random(val_max)
        self.w1 = symetric_random(val_max)
        self.w2 = symetric_random(val_max)
        self.w3 = symetric_random(val_max)
        self.w4 = symetric_random(val_max)

    def current_max_step(self):
        if(len(self.steps) == 0):
             return None
        else:
            return max(np.abs(self.steps))

    def gradient_descent_algorithm_is_finish(self):
        
        if(len(self.steps) == 0):
            step_condition = False
        else:
            step_condition = self.current_max_step() < self.MIN_STEP_SIZE
        
        if(step_condition == True):
            print("step condition is True , current_max_step : {} , MIN_STEP_SIZE : {}".format(self.current_max_step(),self.MIN_STEP_SIZE))
            print("current_num_iterations : {} , MAX_ITERATIONS : {}".format(self.current_num_iterations,self.MAX_ITERATIONS))

        iterations_condition = self.current_num_iterations > self.MAX_ITERATIONS
        if(iterations_condition == True):
            print("num iteration condition is True , current_num_iterations : {} , MAX_ITERATIONS : {}".format(self.current_num_iterations,self.MAX_ITERATIONS))
            print("current_max_step : {} , MIN_STEP_SIZE : {}".format(self.current_max_step(),self.MIN_STEP_SIZE))

        
        return step_condition or iterations_condition


    def compute_new_features_value_given_step_per_feature(self):
        # compute derivative of cost with respect to every feature using the chain rule
        # compute step per feature given derivative 
        # compute new features value given step per feature (order is not important)
        self.steps = []

        b1 = self.b1 - self.compute_step(self.dssr_to_db1())
        b2 = self.b2 - self.compute_step(self.dssr_to_db2())
        b3 = self.b3 - self.compute_step(self.dssr_to_db3())
        w1 = self.w1 - self.compute_step(self.dssr_to_dw1())
        w2 = self.w2 - self.compute_step(self.dssr_to_dw2())
        w3 = self.w3 - self.compute_step(self.dssr_to_dw3())
        w4 = self.w4 - self.compute_step(self.dssr_to_dw4())

        # now we have new features , so update them all
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        
    def update_new_signals_using_forward_propagation(self):
        self.forward_propagation()


    def learn_using_gradient_descent(self):
        self.steps = []
        ssr_vec = []
        self.current_num_iterations = 0
        self.compute_initial_value_for_all_the_features()
        # self.plug_final_values()
        self.update_new_signals_using_forward_propagation()
        
        while(self.gradient_descent_algorithm_is_finish() == False):
            self.compute_new_features_value_given_step_per_feature()
            self.update_new_signals_using_forward_propagation()
            ssr = self.sum_square_residuals()
            print("ssr : {} , iteration : {} , current_max_step : {}".format(ssr , self.current_num_iterations, self.current_max_step()))
            ssr_vec.append(ssr)
            self.print_features()
            self.current_num_iterations += 1

        plt.plot(ssr_vec)
        plt.title('cost - sum square residual vs iterations')
        plt.grid()
        plt.show()

    def compute_numerical_derivative(self,func_value_plus_eps,func_value_minus_eps):
        return (func_value_plus_eps - func_value_minus_eps) / (2*self.NUMERICAL_DERIVATIVE_EPS)


    def debug_check_analytical_derivative(self):
        """ use this to verify derivative are ok
            compare analytical derivative computed with the chain rule to numerical derivatives
            no need to invoke this after it is ok
            complete match -> score is 100
        """
        self.compute_initial_value_for_all_the_features()
        self.update_new_signals_using_forward_propagation()

        i = 0
        while i < self.n: # loop all n features 
            analytic_derivative = self.feature_derivatives[i]
            feature_name = self.feature_names[i]
            self.__dict__[feature_name] += self.NUMERICAL_DERIVATIVE_EPS
            self.update_new_signals_using_forward_propagation()
            func_value_plus_eps = self.sum_square_residuals()

            self.__dict__[feature_name] -= 2*self.NUMERICAL_DERIVATIVE_EPS
            self.update_new_signals_using_forward_propagation()
            func_value_minus_eps = self.sum_square_residuals()

            numerical_derivative = self.compute_numerical_derivative(func_value_plus_eps,func_value_minus_eps)
            print("feature : {} , score : {}  , 100 marks excact analytic derivative as numeric".format(feature_name,100*abs(analytic_derivative()/numerical_derivative)))
            i += 1

obj = BackSimple()
# obj.debug_check_analytical_derivative()
# obj.plot_dataset()
obj.learn_using_gradient_descent()
obj.plot_signals()
