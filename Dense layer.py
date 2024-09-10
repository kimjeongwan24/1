import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = 'brg')
plt.show()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Forward pass
        self.weights = np.random.uniform(0,1,(n_inputs,n_neurons))
        self.biases = np.random.uniform(0,1,(1,n_neurons))

    def forward(self, inputs):
        return np.dot(inputs,self.weights) + self.biases

X,y = spiral_data(samples=100,classes=3)
Dense_Layer = Layer_Dense(2,5)






