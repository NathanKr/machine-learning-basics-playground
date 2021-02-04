import numpy as np
from utils import sigmoid  , cost_function_logistic_regression_J , compute_numerical_derivative


class BackPropagation:
    def __init__(self):
        self.x = np.array([0 , 0.5 , 1]) # input dosage : between 0 and 1 ,  3 x 1
        self.y = np.array([0 , 1 , 0]) # output efficacy : 0 or 1 , 3 x 1
        self.m = self.x.size # number points in data set (same as self.y.size) , 3
        self.n = 7 # number of features . 4 in Teta1 , 3 in Teta2

        self.NUMERICAL_DERIVATIVE_EPS = 0.001
        self.EMPTY_FLOAT = np.empty(1)[0]

        self.x_sample = np.array([1 , self.EMPTY_FLOAT]) # 2 x 1 including bias as first
        self.y_sample = None # scalar
       

        self.numeric_dcost_to_dteta_sample  = np.empty(7) # 7 x 1

        # level 1
        self.a1_sample = None  # 2 x 1 including bias

        # level 1 -> level 2
        self.Teta1_sample = None # 2 x 2
        self.dcost_to_dteta1_sample = np.empty((2,2)) # 2 x 2
        self.dcost_to_dteta1 = np.empty((2,2)) # 2 x 2
        
        # level 2
        self.z2_sample = None # 2 x 1
        self.a2_sample = np.array([1 , self.EMPTY_FLOAT , self.EMPTY_FLOAT]) # 3 x 1 including bias as first
        self.delta2_sample = None # 2 x 1

        # level 2 -> level 3
        self.Teta2_sample = None # 3 x 1
        self.dcost_to_dteta2_sample = None # 3 x 1
        self.dcost_to_dteta2 = None # 3 x 1

        # level 3
        self.a3_sample = None # scalar
        self.z3_sample = None # scalar
        self.delta3_sample = None # scalar
        self.h_sample = None # scalar


    def set_sample(self,i_sample,Teta1_sample,Teta2_sample):
        # self.x_sample[0] is all ready 1 - bias
        self.x_sample[1] = self.x[i_sample]
        self.y_sample = self.y[i_sample]
        self.Teta1_sample = Teta1_sample
        self.Teta2_sample = Teta2_sample


    def forward_propagation_sample(self):
        # layer 1
        self.a1_sample = self.x_sample

        # layer 1 -> layer 2
        self.z2_sample = np.matmul(self.Teta1_sample , self.a1_sample)

        # self.a2_sample[0] is 1 anyway - bias
        sig_z2 = sigmoid(self.z2_sample)
        # todo nath -> use compact
        self.a2_sample[1] = sig_z2[0]
        self.a2_sample[2] = sig_z2[1]
        
        # layer 2 -> layer 3
        self.z3_sample = np.matmul(self.Teta2_sample , self.a2_sample) 
        self.a3_sample = sigmoid(self.z3_sample)        
        self.h_sample = self.a3_sample


    def backward_propagation_sample(self):
        # layer 3
        self.delta3_sample = self.a3_sample - self.y_sample

        # layer 2
        # operator * is used here for element by element multiplication
        # dg_to_dz2 = 
        # for sigmoid : dcost_to_dval(self.z2) is equal to self.a2 * (1 - self.a2)
        # following is correct for sigmoid thus the bias is not included
        # dg_to_dz2_sample = self.a2_sample[1:] * (1 - self.a2_sample[1:])
        dg_to_dz2_sample = self.a2_sample * (1 - self.a2_sample)

        self.delta2_sample = np.matmul(self.Teta2_sample.T , self.delta3_sample) * dg_to_dz2_sample
        # i see no reason to take the bias in delta which is a measure of error , for bias it has no meaning
        self.delta2_sample = self.delta2_sample[1:]

        # layer 3 -> self.delta1 has no meaning for layer 1 because we only have input

    def compute_dcost_to_dteta1_sample(self):
        # this is correct only for the cost function of logisitc regression --> verify this
        # todo nath use this self.dcost_to_dteta1_sample = np.matmul(self.delta2_sample , self.a1_sample.T)
        i=0
        while i < 2:
            j=0
            while j < 2:
                self.dcost_to_dteta1_sample[i,j] = self.a1_sample[i] * self.delta2_sample[j]  
                j += 1
            i += 1
        


    def compute_dcost_to_dteta2_sample(self):
        # this is correct only for the cost function of logisitc regression --> verify this
        # self.dcost_to_dteta2_sample = np.matmul( self.delta3_sample , self.a2_sample.T)
        self.dcost_to_dteta2_sample =  self.delta3_sample * self.a2_sample

    def compute_dcost_to_dteta_sample(self):
        self.forward_propagation_sample()
        self.backward_propagation_sample()
        self.compute_dcost_to_dteta2_sample()
        self.compute_dcost_to_dteta1_sample()



    def compute_cost_sample(self,Teta,X,Y):
        # Teta = self.get_Teta_sample()
        # Y = self.y_sample
        # ## todo nath , replace with symetric_random 
        # X = np.random.rand(self.n) 
        return cost_function_logistic_regression_J(Teta,X,Y)

    def compute_cost_numerical_derivative_sample_per_feature(self,i_feature):
        """[summary]

        Args:
            i_feature (integer): index of the feature in Teta_sample 0-6

        Returns:
            [type]: [description]
        """

        Y = self.y_sample
        # ## todo nath , replace with symetric_random 
        X = np.random.rand(self.n) 

        # save teta state
        saved_Teta1_sample = self.Teta1_sample.copy()
        saved_Teta2_sample = self.Teta2_sample.copy()

        eps = self.NUMERICAL_DERIVATIVE_EPS
        Teta_features_sample = np.copy(self.get_Teta_sample())

        # add eps
        Teta_features_sample[i_feature] += eps
        self.forward_propagation_sample()
        cost_plus_feature_eps = self.compute_cost_sample(Teta_features_sample,X,Y)

        # subtract eps
        Teta_features_sample[i_feature] -= 2*eps
        self.forward_propagation_sample()
        cost_minus_feature_eps = self.compute_cost_sample(Teta_features_sample,X,Y)

        # restore teta state
        self.Teta1_sample = saved_Teta1_sample
        self.Teta2_sample = saved_Teta2_sample
        self.forward_propagation_sample()

        return  compute_numerical_derivative(cost_plus_feature_eps,cost_minus_feature_eps,eps)

    def get_Teta_sample(self):
        """[summary]

        Returns:
            ([number]): 7 x 1 which is 4 from Teta1_sample and 3 from Teta2_sample
        """
        teta1_sample_flat = self.Teta1_sample.reshape(-1)
        teta2_sample_flat = self.Teta2_sample.reshape(-1)

        return np.concatenate((teta1_sample_flat , teta2_sample_flat))

    def compute_numeric_dcost_to_dteta_sample(self):
        i_feature = 0
        while i_feature < self.n: # loop over features
            dcost_to_dfeature = self.compute_cost_numerical_derivative_sample_per_feature(i_feature)
            self.numeric_dcost_to_dteta_sample[i_feature] = dcost_to_dfeature
            i_feature += 1

    def compute_cost_derivative_and_check(self):
        i_sample = 0
        obj.dcost_to_dteta1 = ...........
        while i_sample < obj.m: # loop over data set samples
            # use rand because i want simply to check derivative
            Teta1_sample = np.random.rand(2,2)
            Teta2_sample = np.random.rand(1,3)
            obj.set_sample(i_sample,Teta1_sample,Teta2_sample)
            obj.compute_dcost_to_dteta_sample() 
            obj.compute_numeric_dcost_to_dteta_sample()
            obj.compare_derivatives_sample()
            i_sample += 1

    def compare_derivatives_sample(self):
        dcost_to_dteta1_sample_flat = self.dcost_to_dteta1_sample.reshape(-1)
        dcost_to_dteta_sample_flat =  np.concatenate((dcost_to_dteta1_sample_flat , self.dcost_to_dteta2_sample))

        result_percent = 100 * self.numeric_dcost_to_dteta_sample / dcost_to_dteta_sample_flat
        print(f"delta rule vs numeric dcost / dteta  (100 is perfect match): {result_percent}")

# main
obj = BackPropagation()
obj.compute_cost_derivative_and_check()
