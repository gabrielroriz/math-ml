import numpy as np
from src.utils import print_array, print_vars
#import matplotlib.pyplot as plt

n_input_l = 2
n_hidden_l = 4
n_output_l = 1

learning_rate = 0.1

def sigmoid(z):
    return 1 / (1 + (np.e ** -z))

import numpy as np

def initialize_parameters(show_print=False):
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

    if show_print == True:
        # print("=== Inicialização dos Parâmetros ===\n")
        # print_array(b1, 
        #             ["Bias 1"], 
        #             [f"[l1] Neuron {i + 1}" for i in range(n_hidden_l)], 
        #             title_table="B1 | Bias - Layer 1",
        #             title_columns="From (bias):",
        #             title_rows="To:",
        #             )

        print_array(W1, 
                    [f"(X) Feature {i + 1}" for i in range(n_input_l)], 
                    [f"[l1] Neuron {i + 1}" for i in range(n_hidden_l)], 
                    title_table="W1 | Weights - Layer 1", 
                    title_columns="From (inputs):",
                    title_rows="To (neurons):", 
                    )

        # print_array(b2,
        #             ["Bias 2"], 
        #             [f"[l2] Neuron {i + 1}" for i in range(n_output_l)],
        #             title_table="B2 | Bias - Layer 2 (hidden)",
        #             title_columns="From:",
        #             title_rows="To:", 
        #             )
        
        print_array(W2, 
                    [f"[l2] Neuron {i + 1}" for i in range(n_hidden_l)], 
                    [f"[l2] Output {i + 1}" for i in range(n_output_l)],
                    title_table="W2 | Weights - Layer 2 (hidden)", 
                    title_columns="From (neurons):",
                    title_rows="To (output):", 
                    )

    return {
        "b1": b1, 
        "W1": W1, 
        "b2": b2, 
        "W2": W2
    }

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(parameters, x, show_print=False):
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

    if show_print == True:
        print("\n=== Forward Propagation ===\n")

        print_array(
        Z1, 
        title_table="Z1 (hidden pre-activation)", 
        row_headers=[f"[l2] Neuron {i + 1}" for i in range(n_hidden_l)],
        headers=[f"Z"])
    
        print("\n")

        print_array(
            A1, 
            title_table="A1 (hidden activation)", 
            row_headers=[f"[l2] Neuron {i + 1}" for i in range(n_hidden_l)], 
            headers=[f"sigmoid(Z)"])
        print("\n")
        
        print_array(
            Z2, 
            title_table="Z2 (output pre-activation)")
        print("\n")

        print_array(
            A2, 
            title_table="A2 (output activation)")
        print("\n")

    return {
        "Z2": Z2,
        "A1": A1,
        "Z1": Z1,
        "A2": A2,
    }

def gen_dataset_xor():
    np.random.seed(0)
    n = 500
    
    X = np.random.rand(n, 2) * 2 - 1  # [-1, 1]
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

def log_loss(y, y_hat):
    # L(y,ŷ) = (y * log(ŷ)) + ((1-y) * log(1-ŷ))
    return -1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def back_propagation(log_loss_arr, X, Y, a1_arr, y_hat_arr, parameters):

    def d_loss_d_yhat(y, y_hat):
        # -(y - Ŷ) / (Ŷ * (1- Ŷ))
        return (-1 * (y - y_hat)) / (y_hat * (1 - y_hat))

    def d_yhat_d_zL2(y_hat):
        # Sigmoid: Ŷ * (1-Ŷ)
        return y_hat * (1 - y_hat)

    def d_zL2_d_aL1(W2):
        # W2
        return W2

    def d_aL1_d_zL1(aN):
        # Sigmoid: An * (1 - An)
        return aN * (1 - aN)

    def d_zL1_d_wN(xN):
        # xN
        return xN

    def layer_1_gradient_formula(x, aN, w2, y, y_hat):
        gradient_formula_full = d_zL1_d_wN(x) * d_aL1_d_zL1(aN) * d_zL2_d_aL1(w2) * d_yhat_d_zL2(y_hat) * d_loss_d_yhat(y, y_hat)        
        return gradient_formula_full

    # Layer 1 -> Output

    # Gradient descent [1]w1,1
    # Gradient descent [1]w1,2

    # Gradient descent [1]w2,1
    # Gradient descent [1]w2,2

    # Gradient descent [1]w3,1
    # Gradient descent [1]w3,2

    # Gradient descent [1]w4,1
    # Gradient descent [1]w4,2

    # Gradient descent [1]b1

    def gradient_descent(log_loss, y_hat, x, y, parameters):
        # First layer
        for i in range(n_input_l):
            for j in range(n_hidden_l):
                w1 = parameters["W1"][j, i]
                w2 = parameters["W2"][0][j]
                a1 = a1_arr[0][j]
                gradient = layer_1_gradient_formula(x, a1, w2, y, y_hat)
                new_weight = w1 - (learning_rate * gradient)
                print_vars(x, w1, a1, w2, y, y_hat, learning_rate, gradient, new_weight)
                parameters["W1"][j, i] = new_weight
                # print(new_weight)
                # print(f"[l1]W{i},{j} = {W1[j, i]}")
        
        # Hidden layer
        for i in range(n_hidden_l):
            for j in range(n_output_l):
                pass
                # print(f"[l2]W{i},{j}")

    gradient_descent(log_loss_arr[0], y_hat_arr[0], X[0][0], Y[0], parameters)
    

    # Layer 2 -> Output

    # Gradient Descent [2]w1,1
    # Gradient Descent [2]w2,1
    # Gradient Descent [2]w3,1
    # Gradient Descent [2]w4,1
    pass



def run():
    parameters = initialize_parameters(True)

    X, Y = gen_dataset_xor()

    results = [forward_propagation(parameters, np.array([[x[0]], [x[1]]]), show_print=False) for x in X]

    y_hat_arr = np.array([r["A2"][0][0] for r in results])

    a1_arr = np.array([r["A1"].flatten() for r in results])

    log_loss_array = [
        log_loss(y_item, y_hat_arr[i])
        for i, y_item in enumerate(Y)
    ]

    back_propagation(log_loss_array, X, Y, a1_arr, y_hat_arr, parameters)




    # print(f"X = {x_item}")
    # print(f"Y = {y_item}")
    # print(f"Ŷ = {y_hat[0][0]}")
    # print(f"Log-loss = {log_loss(y_item, y_hat[0][0])}")

    # print(X.shape)
    # print(Y.shape)
    # print(Y[0])
    # print(X[0])


run()
