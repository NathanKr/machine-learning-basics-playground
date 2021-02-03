import numpy as np
from utils import sigmoid  , cost_function_logistic_regression_J , compute_numerical_derivative


class BackPropagation:
    def __init__(self):
        self.x = np.array([0 , 0.5 , 1]) # input dosage : between 0 and 1 ,  3 x 1
        self.y = np.array([0 , 1 , 0]) # output efficacy : 0 or 1 , 3 x 1
        self.m = self.x.size # number points in data set (same as self.y.size) , 3
        self.n = 7 # number of features . 4 in Teta1 , 3 in Teta2

        self.x_sample = np.array([1 , None]) # 2 x 1 including bias as first
        self.y_sample = None # scalar
       
        self.NUMERICAL_DERIVATIVE_EPS = 0.001

        self.numeric_dcost_to_dteta_sample  = np.empty(7) # 7 x 1

        # level 1
        self.a1_sample = None  # 2 x 1 including bias

        # level 1 -> level 2
        self.Teta1_sample = None # 2 x 2
        self.dcost_to_dteta1_sample = None # 2 x 2
        
        # level 2
        self.z2_sample = None # 2 x 1
        self.a2_sample = None  # 3 x 1 including bias
        self.delta2_sample = None # 2 x 1

        # level 2 -> level 3
        self.Teta2_sample = None # 1 x 3
        self.dcost_to_dteta2_sample = None # 1 x 3

        # level 3
        self.a3_sample = None # scalar
        self.z3_sample = None # scalar
        self.delta3_sample = None # scalar
        self.h_sample = None # scalar


    def set_sample(self,i_sample,Teta1_sample,Teta2_sample):
        # self.x_sample[0] is 1 - bias anyway
        self.x_sample[1] = self.x[i_sample]
        self.y_sample = self.y[i_sample]
        self.Teta1_sample = Teta1_sample
        self.Teta2_sample = Teta2_sample


    def forward_propagation(self):
        # layer 1
        self.a1_sample = self.x_sample

        # layer 1 -> layer 2
        self.z2_sample = np.matmul(self.Teta1_sample , self.a1_sample)
        self.a2_sample = sigmoid(self.z2_sample)
        
        # layer 2 -> layer 3
        self.z3_sample = np.matmul(self.Teta2_sample , self.a2_sample) 
        self.a3_sample = sigmoid(self.z3_sample)        
        self.h_sample = self.a3_sample


    def backward_propagation(self):
        # layer 3
        self.delta3_sample = self.a3_sample - self.y_sample

        # layer 2
        # operator * is used here for element by element multiplication
        # dg_to_dz2 = 
        # for sigmoid : dcost_to_dval(self.z2) is equal to self.a2 * (1 - self.a2)
        dg_to_dz2_sample = self.a2_sample * (1 - self.a2_sample)
        self.delta2_sample = np.matmul(self.Teta2_sample.T , self.delta3_sample) * dg_to_dz2_sample

        # layer 3 -> self.delta1 has no meaning for layer 1 because we only have input

    def compute_dcost_to_dteta1_sample(self):
        # this is correct only for the cost function of logisitc regression --> verify this
        self.dcost_to_dteta1_sample = self.a1_sample * self.delta2_sample

    def compute_dcost_to_dteta2_sample(self):
        # this is correct only for the cost function of logisitc regression --> verify this
        self.dcost_to_dteta2_sample = self.a2_sample * self.delta3_sample

    def compute_dcost_to_dteta_sample(self):
        self.forward_propagation()
        self.backward_propagation()
        self.compute_dcost_to_dteta1_sample()
        self.compute_dcost_to_dteta2_sample()

    def compute_cost_sample(self):
        return cost_function_logistic_regression_J(self.get_Teta_sample(),self.x_sample,self.y_sample)

    def compute_cost_numerical_derivative_sample_per_feature(self,i_feature):
        """[summary]

        Args:
            i_feature (integer): index of the feature in Teta_sample 0-6

        Returns:
            [type]: [description]
        """
        eps = self.NUMERICAL_DERIVATIVE_EPS
        Teta_sample_feature = np.copy(self.get_Teta_sample())

        Teta_sample_feature[i_feature] += eps
        cost_plus_feature_eps = self.compute_cost_sample()

        Teta_sample_feature[i_feature] -= 2*eps
        cost_minus_feature_eps = self.compute_cost_sample()

        return  compute_numerical_derivative(cost_plus_feature_eps,cost_minus_feature_eps,eps)

    def get_Teta_sample(self):
        """[summary]

        Returns:
            ([number]): 7 x 1 which is 4 from Teta1_sample and 3 from Teta2_sample
        """
        teta_1__sample_flat = self.Teta1_sample.reshape(-1)
        return np.concatenate(teta_1__sample_flat , self.Teta2_sample)

    def compute_numeric_cost_derivative_sample(self):
        i_feature = 0
        while i_feature < self.n: # loop over features
            dcost_to_dfeature = self.compute_cost_numerical_derivative_sample_per_feature(i_feature)
            self.numeric_dcost_to_dteta_sample[i_feature] = dcost_to_dfeature
            i_feature += 1

    def compute_cost_derivative_and_check(self):
        i_sample = 0
        while i_sample < obj.m:
            # use rand simply to check derivative
            Teta1_sample = np.rand(2,2)
            Teta2_sample = np.rand(1,3)
            obj.set_sample(i_sample,Teta1_sample,Teta2_sample)
            obj.compute_dcost_to_dteta_sample() 
            obj.compute_numeric_cost_derivative_sample()
            i_sample += 1


# main
obj = BackPropagation()
obj.compute_cost_derivative_and_check()
