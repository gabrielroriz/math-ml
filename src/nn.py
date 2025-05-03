import numpy as np
from src.utils import print_array, print_vars
from sklearn.metrics import log_loss as sk_log_loss

from decimal import Decimal, getcontext


getcontext().prec = 50  # Define a precisão global


n_input_l = 2
n_hidden_l = 4
n_output_l = 1

learning_rate = 0.1

def sigmoid(z):
    return 1 / (1 + (np.e ** -z))


def print_parameters(parameters):
    B1 = parameters["B1"]
    W1 = parameters["W1"]
    B2 = parameters["B2"]
    W2 = parameters["W2"]

    print("=== Inicialização dos Parâmetros ===\n")

    # B1: shape (1, n_hidden_l) → 1 linha, várias colunas (cada coluna é um neurônio da camada 1)
    print_array(B1,
                [f"Neuron [1]{i + 1}" for i in range(n_hidden_l)],
                ["Bias 1"],
                title_table="B1 | Bias - Layer 1",
                title_columns="To (neurons [1])",
                title_rows="From (bias)",
                )

    # W1: shape (n_input_l, n_hidden_l) → cada linha é uma feature de entrada, cada coluna é um neurônio da camada 1
    print_array(W1,
                [f"Neuron [1]{i + 1}" for i in range(n_hidden_l)],  # colunas = destino
                [f"Feature X{i + 1}" for i in range(n_input_l)],  # linhas = origem
                title_table="W1 | Weights - Layer 1",
                title_columns="To (neurons [1])",
                title_rows="From (features)",
                )

    # B2: shape (1, n_output_l) → 1 linha, várias colunas (cada coluna é uma saída)
    print_array(B2,
                [f"Neuron [2]{i + 1}" for i in range(n_output_l)],
                ["Bias 2"],
                title_table="B2 | Bias - Layer 2",
                title_columns="To (neurons [l2])",
                title_rows="From (bias)",
                )

    # W2: shape (n_hidden_l, n_output_l) → cada linha = neurônio da camada oculta, cada coluna = saída
    print_array(W2,
                [f"Neuron [2]{i + 1}" for i in range(n_output_l)],  # colunas = destino
                [f"[l1] Neuron {i + 1}" for i in range(n_hidden_l)],  # linhas = origem
                title_table="W2 | Weights - Layer 2",
                title_columns="To (neurons[2])",
                title_rows="From (neurons[1])",
                )

def initialize_parameters(show_print=False):
    np.random.seed(0)  # Fixar a aleatoriedade para reprodutibilidade

    B1 = np.zeros((1, n_hidden_l))

    W1 = np.random.randn(n_input_l, n_hidden_l) * 0.01

    B2 = np.zeros((1, n_output_l))

    W2 = np.random.randn(n_hidden_l, n_output_l) * 0.01

    parameters = {
        "B1": B1, 
        "W1": W1, 
        "B2": B2, 
        "W2": W2
    }

    if show_print == True: print_parameters(parameters)
        
    return parameters

def initialize_parameters_fixed(show_print=False):
    B1 = np.array([[-3.9686972729406733, 
                    3.616801466722483, 
                    -3.963258397115321, 
                    -4.188478200079775]])

    W1 = np.array([[6.845696558372827,   6.163896748126158,  -6.762078570871174,  7.5943138515861435],
                   [6.752729189010516,  -6.6118234805183125, -7.258583045498932, -7.022057369035268]])

    B2 = np.array([[19.289751171239327]])

    W2 = np.array([[-19.66074984961242],
                   [-19.64813378916913],
                   [-18.496934206613314],
                   [19.142335549306]])

    parameters = {
        "B1": B1,
        "W1": W1,
        "B2": B2,
        "W2": W2
    }

    if show_print:
        print_parameters(parameters)
        
    return parameters

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(parameters, x, show_print=False):
    B1 = parameters["B1"]
    W1 = parameters["W1"]
    B2 = parameters["B2"]
    W2 = parameters["W2"]

    Z1 = np.dot(W1.T, x) + B1.T

    A1 = sigmoid(Z1)

    Z2 = np.dot(W2.T, A1) + B2.T

    A2 = sigmoid(Z2)

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


import matplotlib.pyplot as plt

def gen_dataset_xor(n, ploting=False):
    np.random.seed(0)

    X = np.random.rand(n, 2) * 2 - 1  # [-1, 1]
    Y = (X[:, 0] * X[:, 1] < 0).astype(int) # XOR

    # Ploting
    if ploting:
        class_0 = X[Y == 0]
        class_1 = X[Y == 1]
        plt.figure(figsize=(8, 6))
        plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Classe 0')
        plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Classe 1')
        plt.title('Distribuição dos dados XOR com ruído')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    return X, Y

def log_loss(y, y_hat):
    # L(y,ŷ) = (y * log(ŷ)) + ((1-y) * log(1-ŷ))
    return -1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def back_propagation(parameters, X, Y, forward_cache, show_print=False):

    def d_loss_d_yhat(y: Decimal, y_hat: Decimal) -> Decimal:
        return -((y / y_hat) - ((1 - y) / (1 - y_hat)))

    def d_yhat_d_z2(y_hat: Decimal) -> Decimal:
        # Sigmoid derivative: ŷ * (1 - ŷ)
        return y_hat * (1 - y_hat)

    def d_z2_d_a1(w2: Decimal) -> Decimal:
        return w2
    
    def d_z2_d_w2(a1: Decimal) -> Decimal:
        return a1

    def d_a1_d_z1(a1: Decimal) -> Decimal:
        # Sigmoid derivative: a1 * (1 - a1)
        return a1 * (1 - a1)

    def d_z1_d_w1(x: Decimal) -> Decimal:
        return x
    
    def d_z1_d_b1() -> Decimal:
        return 1

    def d_z2_d_b2() -> Decimal:
        return 1

    def w1_gradient_formula(x, a1, w2, y, y_hat):    
        v__d_z1_d_w1 = d_z1_d_w1(x)
        v__d_a1_d_z1 = d_a1_d_z1(a1)
        v__d_z2_d_a1 = d_z2_d_a1(w2)
        v__d_yhat_d_z2 = d_yhat_d_z2(y_hat)
        v__d_loss_d_yhat = d_loss_d_yhat(y, y_hat)

        gradient_formula_full = v__d_z1_d_w1 * v__d_a1_d_z1 * v__d_z2_d_a1 * v__d_yhat_d_z2 * v__d_loss_d_yhat

        simple_formula = (-x) * (w2) * (a1) * (1 - a1) * (y - y_hat)
        
        # print_var(gradient_formula_full, simplified_formula)
        assert np.isclose(gradient_formula_full, simple_formula), "Mismatch between forms!"

        return gradient_formula_full
    
    def b1_gradient_formula(a1, w2, y, y_hat):    
        v__d_z1_d_b1 = d_z1_d_b1()
        v__d_a1_d_z1 = d_a1_d_z1(a1)
        v__d_z2_d_a1 = d_z2_d_a1(w2)
        v__d_yhat_d_z2 = d_yhat_d_z2(y_hat)
        v__d_loss_d_yhat = d_loss_d_yhat(y, y_hat)

        gradient_formula_full = v__d_z1_d_b1 * v__d_a1_d_z1 * v__d_z2_d_a1 * v__d_yhat_d_z2 * v__d_loss_d_yhat

        simple_formula = (-w2) * (a1) * (1 - a1) * (y - y_hat)
        
        # print_var(gradient_formula_full, simplified_formula)
        assert np.isclose(gradient_formula_full, simple_formula), "Mismatch between forms!"

        return gradient_formula_full
        
    
    def w2_gradient_formula(a1, y, y_hat):
        dLoss_dYhat = d_loss_d_yhat(y, y_hat)
        dYhat_dZ2 = d_yhat_d_z2(y_hat)
        dZ2_dW2 = d_z2_d_w2(a1)

        simple_formula = a1 * (-(y - y_hat))
        gradient_formula_full = dZ2_dW2 * dYhat_dZ2 * dLoss_dYhat

        # print_vars(simple_formula, gradient_formula_full)
        assert np.isclose(gradient_formula_full, simple_formula), "Mismatch between forms!"
        return gradient_formula_full
    
    def b2_gradient_formula(y, y_hat):
        dLoss_dYhat = d_loss_d_yhat(y, y_hat)           # derivada da log-loss
        dYhat_dZ2 = d_yhat_d_z2(y_hat)                  # derivada da sigmoid
        dZ2_dB2 = d_z2_d_b2()                            # constante: 1

        gradient = dLoss_dYhat * dYhat_dZ2 * dZ2_dB2     # produto da cadeia

        return gradient
                
    
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]

    A1 = forward_cache["A1"]
    Z1 = forward_cache["Z1"]
    A2 = forward_cache["A2"]
    Z2 = forward_cache["Z2"]

    # W1
    for i_input in range(n_input_l):
        for j_neuron in range(n_hidden_l):
            if show_print: print(f"W[1]{i_input},{j_neuron}")
            x = X[i_input]
            w1 = W1[i_input][j_neuron]
            a1 = A1[j_neuron][0]
            w2 = W2[j_neuron][0]
            a2 = A2[0][0]

            gradient = w1_gradient_formula(x, a1, w2, Y, a2)
            new_w1 = w1 - (learning_rate * gradient)
            W1[i_input][j_neuron] = new_w1
            #print_vars(x, w1, a1, w2, a2, gradient, new_weight)

    # B1
    for i_neuron in range(n_hidden_l):
        if show_print: print(f"B[1]{i_neuron}")
        b1 = B1[0][i_neuron]
        a1 = A1[i_neuron][0]
        w2 = W2[i_neuron][0]
        a2 = A2[0][0]

        gradient = b1_gradient_formula(a1, w2, Y, a2)
        new_b1 = b1 - (learning_rate * gradient)
        B1[0][i_neuron] = new_b1
        #print_vars(x, w1, a1, w2, a2, gradient, new_weight)
    
    # W2
    for i_neuron in range(n_hidden_l):
        for j_output in range(n_output_l):
            if show_print: print(f"W[2]{i_neuron},{j_output}")
            a1 = A1[i_neuron][0]
            a2 = A2[0][0]
            w2 = W2[i_neuron][0]
            gradient = w2_gradient_formula(a1, Y, a2)
            new_w2 = w2 - (learning_rate * gradient)
            W2[i_neuron][j_output] = new_w2
    
    # B2
    for i_output in range(n_output_l):
        if show_print: print(f"B[2]{i_output}")
        b2 = B2[0][i_output]
        a2 = A2[0][0]
        gradient = b2_gradient_formula(Y, a2)
        new_b2 = b2 - (learning_rate * gradient)
        B2[0][i_output] = new_b2

def predict(parameters, x_input, threshold=0.5, show_print=False):
    # Converte para coluna se estiver em linha
    if x_input.ndim == 1:
        x_input = x_input.reshape(-1, 1)

    forward_cache = forward_propagation(parameters, x_input, show_print=show_print)
    y_hat = forward_cache["A2"][0][0]

    predicted_class = int(y_hat >= threshold)

    if show_print:
        print_vars(x_input, y_hat, predicted_class)

    return y_hat, predicted_class

def run():

    parameters = initialize_parameters(show_print=True)

    epochs = 100

    X, Y = gen_dataset_xor(2000)

    for _ in range(epochs):
        for i in range(len(Y)):
            # Features
            x_selected = X[i]

            # True output
            y_selected = Y[i]

            # Step 1 - Forward Propagation
            forward_cache = forward_propagation(parameters, np.array([[x_selected[0]], [x_selected[1]]]), show_print=False)

            y_hat = forward_cache["A2"][0][0]
            
            # Step 2 - Calculate cost (Log-Loss)
            ll = log_loss(y_selected, y_hat)

            scilog = sk_log_loss(np.array([y_selected]), np.array([y_hat]), labels=[0, 1])

            # print_vars(ll, scilog)
            assert np.isclose(ll, scilog), "Mismatch log-loss values!"
            
            # 3 - Back propagation + update parameters
            back_propagation(parameters, x_selected, y_selected, forward_cache, show_print=True)

    print_parameters(parameters)

    print("\n=== Predição ===\n")

    # Vamos testar a entrada [1, -1], que deve dar classe 1 no XOR
    x_test = np.array([[1.0], [-1.0]])
    y_hat, prediction = predict(parameters, x_test, show_print=True)

    print(f"Input: [1, -1] -> y_hat: {y_hat:.5f}, Predição: {prediction}")


def run_predict():
    parameters = initialize_parameters_fixed(show_print=True)

    print("\n=== Predição ===\n")

    test_inputs = [
        [1.0, -1.0],   # XOR → 1
        [0.0, 0.0],    # → 0
        [1.0, 1.0],    # → 0
        [-1.0, -1.0],  # → 0
        [0.5, -0.5],   # → 1
        [-0.5, 0.5],   # → 1
        [0.9, -0.9],   # → 1
        [-0.9, 0.9],   # → 1
        [1.0, 0.0],    # → 1
        [0.0, 1.0],    # → 1
        [-1.0, 0.0],   # → 1
        [0.0, -1.0],   # → 1
        [0.2, 0.2],    # → 0
        [-0.2, -0.2],  # → 0
        [0.7, -0.7],   # → 1
        [-0.7, 0.7],   # → 1
        [0.8, 0.8],    # → 0
        [-0.8, -0.8],  # → 0
        [1.0, -0.8],   # → 1
        [-1.0, 0.8],   # → 1
        [0.3, -0.3],   # → 1
        [0.1, 1],   # → 0
    ]

    for i, (x1, x2) in enumerate(test_inputs):
        x_test = np.array([[x1], [x2]])
        y_hat, prediction = predict(parameters, x_test, show_print=False)
        print(f"#{i+1:02d} | Input: [{x1:>5}, {x2:>5}] -> y_hat: {y_hat:.5f} | Predição: {prediction}")


#run()
run_predict()
