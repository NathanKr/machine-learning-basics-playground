from utils import true_positive , false_positive , precision , false_negative , recall , F1score
import numpy as np

actual =   np.array([ 0 , 1 , 1 , 0 , 1 , 0 , 0 , 0])
estimated = np.array([0 , 1 , 0 , 1 , 1 , 1 , 0 , 1])

print(true_positive(actual,estimated) == 2)    

print(false_positive(actual,estimated) == 3)    

print(false_negative(actual,estimated) == 1)    

expected_prec =  2/(2+3)
print(precision(actual,estimated) == expected_prec)

expected_recall = 2/(2+1)
print(recall(actual,estimated) == expected_recall)

print(F1score(actual,estimated) == 2*expected_recall*expected_prec/(expected_recall+expected_prec))





