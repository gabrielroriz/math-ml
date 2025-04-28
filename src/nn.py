import numpy as np
from src.utils import print_array

n_input_l = 2
n_hidden_l = 4
n_output_l = 1

def sigmoid(z):
    return 1 / (1 + (np.e ** -z))

def initialize_parameters():
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
    # |  0.000639755 | 0.00251784 |
    # +--------------+------------+
    # | -0.00170115  | 0.0141058  |
    # +--------------+------------+
    # |  0.00996568  | 0.00237355 |
    # +--------------+------------+
    # |  0.00850378  | 0.00579479 |
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
    # | 0.00224718 | -0.00892212 | -3.02401e-05 | 0.00264288 |
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

def forward_propagation(parameters, x1, x2):
    b1 = parameters["b1"]
    W1 = parameters["W1"]
    b2 = parameters["b2"]
    W2 = parameters["W2"]

    a1 = x1 * W1 + b1
    a2 = x2 * W2 + b2
    # a1:
    # +--------------+------------+
    # |  0.00639755  | 0.0251784  |
    # +--------------+------------+
    # | -0.0170115   | 0.141058   |
    # +--------------+------------+
    # |  0.0996568   | 0.0237355  |
    # +--------------+------------+
    # |  0.0850378   | 0.0579479  |
    # +--------------+------------+

    # Matrix a1 represents the activations computed at the hidden layer
    # based on the weighted sum of the input feature x1 plus the bias b1.
    # Each row (i) corresponds to a hidden neuron.
    # Each column (j) corresponds to the contribution from a specific input feature (x1).
    # 
    # Formula: 
    #   a1[i][j] = W1[i][j] * x1 + b1[i]
    #
    # These intermediate activations are not yet passed through a non-linear activation function.
    # They are still in the linear pre-activation stage.

    # a2:
    # +-----------+------------+--------------+-----------+
    # | 0.224718  | -0.892212  | -0.00302401  | 0.264288  |
    # +-----------+------------+--------------+-----------+

    # Matrix a2 represents the activations computed at the output layer
    # based on the weighted sum of the second input feature x2 plus the bias b2.
    # Each value in a2 corresponds to the raw contribution from a hidden neuron
    # to the final output neuron.
    #
    # Formula:
    #   a2[0][k] = W2[0][k] * x2 + b2[0]
    #
    # Like a1, these values are pre-activation outputs — 
    # they should be passed through an activation function (e.g., sigmoid, softmax) 
    # depending on the task (binary classification, multi-class classification, etc.).

    print("\na1:")
    print_array(a1)
    print("\na2:")
    print_array(a2)



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
forward_propagation(parameters, 10, 100)
