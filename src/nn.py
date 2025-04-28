import numpy as np
from src.utils import print_array
import matplotlib.pyplot as plt

n_input_l = 2
n_hidden_l = 4
n_output_l = 1

def sigmoid(z):
    return 1 / (1 + (np.e ** -z))

import numpy as np

def initialize_parameters():
    np.random.seed(0)  # Fixar a aleatoriedade para reprodutibilidade

    b1 = np.zeros((n_hidden_l, 1))

    W1 = np.random.randn(n_hidden_l, n_input_l) * 0.01

    b2 = np.zeros((n_output_l, 1))

    W2 = np.random.randn(n_output_l, n_hidden_l) * 0.01

    # b1:
    # +---+
    # | 0 |
    # +---+
    # | 0 |
    # +---+
    # | 0 |
    # +---+
    # | 0 |
    # +---+

    # Each row (i) represents a neuron in the hidden layer.
    # b1[i] corresponds to the bias added to the i-th hidden neuron before activation.
    # Bias terms allow neurons to shift their activation independently of the input,
    # enabling the network to model functions that do not necessarily pass through the origin.

    # W1:
    # +--------------+------------+
    # |  0.01764052  |  0.00400157 |
    # +--------------+------------+
    # |  0.00978738  |  0.02240893 |
    # +--------------+------------+
    # |  0.01867558  | -0.00977278 |
    # +--------------+------------+
    # |  0.00950088  | -0.00151357 |
    # +--------------+------------+

    # W1 is the weight matrix connecting the input layer to the hidden layer.
    # Each row (i) corresponds to a neuron in the hidden layer,
    # and each column (j) corresponds to an input feature (x1, x2, etc.).
    # Thus, W1[i][j] represents the weight applied to the j-th input feature for the i-th hidden neuron.
    # These weights determine the influence of each input on each hidden neuron.

    # b2:
    # +---+
    # | 0 |
    # +---+

    # b2 is the bias for the output neuron.
    # Since there is only one output neuron, there is a single bias term.
    # This bias is added to the weighted sum of hidden layer activations before applying the output activation function.

    # W2:
    # +------------+-------------+--------------+------------+
    # | 0.00103219 | 0.00410599   | 0.00144044   | 0.01454274 |
    # +------------+-------------+--------------+------------+

    # W2 is the weight matrix connecting the hidden layer to the output layer.
    # Each row corresponds to an output neuron (only one row in this case),
    # and each column corresponds to a hidden neuron.
    # Thus, W2[0][k] represents the weight applied to the activation output of the k-th hidden neuron
    # when computing the final output.
    # These weights allow the model to combine the learned features from the hidden layer to predict the final result.

    return {
        "b1": b1, 
        "W1": W1, 
        "b2": b2, 
        "W2": W2
    }

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(parameters, x):
    b1 = parameters["b1"]
    W1 = parameters["W1"]
    b2 = parameters["b2"]
    W2 = parameters["W2"]

    # First layer (input -> hidden)
    Z1 = np.dot(W1, x) + b1

    # Activation from hidden layer
    A1 = sigmoid(Z1)

    # Second layer (hidden -> output)
    Z2 = np.dot(W2, A1) + b2

    # Activation from output layer
    A2 = sigmoid(Z2)

    # Z1:
    # +--------------+
    # |  value       |
    # +--------------+
    # Represents the pre-activation linear combination at the hidden layer
    # before applying the non-linear activation function (sigmoid).

    # A1:
    # +--------------+
    # | sigmoid(Z1)  |
    # +--------------+
    # Represents the output after applying the activation function
    # for each hidden neuron.

    # Z2:
    # +--------------+
    # |  value       |
    # +--------------+
    # Represents the pre-activation linear combination at the output layer
    # before applying the final activation function (sigmoid).

    # A2:
    # +--------------+
    # | sigmoid(Z2)  |
    # +--------------+
    # Represents the final prediction output (between 0 and 1).

    print("\nZ1 (hidden pre-activation):")
    print_array(Z1)
    
    print("\nA1 (hidden activation):")
    print_array(A1)
    
    print("\nZ2 (output pre-activation):")
    print_array(Z2)
    
    print("\nA2 (output activation):")
    print_array(A2)

    return A2

def gen_dataset_xor():
    np.random.seed(0)
    
    X = np.random.rand(200, 2) * 2 - 1  # [-1, 1]
    Y = (X[:, 0] * X[:, 1] < 0).astype(int) # XOR

    # Ploting

    # class_0 = X[Y == 0]
    # class_1 = X[Y == 1]
    # plt.figure(figsize=(8, 6))
    # plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Classe 0')
    # plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Classe 1')
    # plt.title('Distribuição dos dados XOR com ruído')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return X, Y

print("=== Inicialização dos Parâmetros ===\n")

parameters = initialize_parameters()

print("b1:")
print_array(parameters["b1"])
print("\nW1:")
print_array(parameters["W1"])
print("\nb2:")
print_array(parameters["b2"])
print("\nW2:")
print_array(parameters["W2"])

print("\nforward_propagation:")

x = np.array([[10], [-100]])

forward_propagation(parameters, x)

X, Y = gen_dataset_xor()

print(X.shape)
print(Y.shape)
