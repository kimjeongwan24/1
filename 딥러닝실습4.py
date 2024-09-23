import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt
nnfs.init()

# Define Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Define ReLU Activation Function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  # ReLU activation

# Create layers
dense1 = Layer_Dense(1, 8)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(8, 8)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(8, 1)

# Generate data
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

# Forward pass through layers with activations
dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

dense3.forward(activation2.output)

# Plot the results
plt.plot(X, y, label="True Sine Wave", color='blue')
plt.plot(X, dense3.output, label="Predicted by Network", color='red')
plt.legend()
plt.title('Sine Wave Approximation with Neural Network')
plt.show()