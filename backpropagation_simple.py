import numpy as np
import matplotlib.pyplot as plt


# this is only estimation
x = np.array([0 , 0.5 , 1])
y = np.array([0 , 1 , 0])

def plot_dataset():
    plt.plot(x,y,'o')
    plt.title('Data set')
    plt.xlabel("Dosage")
    plt.ylabel("Efficacy ")
    plt.grid()
    plt.show()

# main
# plot_dataset()