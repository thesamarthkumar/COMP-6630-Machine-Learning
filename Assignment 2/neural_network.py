# ---------- Neural Network from Scratch ---------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset using pandas
X_train = pd.read_csv("X_train.csv", header=None, delimiter=r"\s+").astype(float).values
Y_train = pd.read_csv("Y_train.csv", header=None, delimiter=r"\s+").astype(float).values
X_test = pd.read_csv("X_test.csv", header=None, delimiter=r"\s+").astype(float).values
Y_test = pd.read_csv("Y_test.csv", header=None, delimiter=r"\s+").astype(float).values

'''
Define the activation functions and their derivatives.
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative)
}

'''
Define all the functions for building and training the neural network.
'''

# Setting the parameters.
def initialize_params(X_train, hidden_size):
  W1 = np.random.randn(X_train.shape[1], hidden_size) * 0.01
  b1 = np.zeros((1, hidden_size))
  W2 = np.random.randn(hidden_size, 1) * 0.01
  b2 = np.zeros((1, 1))

  return W1, b1, W2, b2

# Updating the parameters.
def update_params(W1, b1, W2, b2, dL_dW1, dL_db1, dL_dW2, dL_db2):
  W1 -= learning_rate * dL_dW1
  b1 -= learning_rate * dL_db1
  W2 -= learning_rate * dL_dW2
  b2 -= learning_rate * dL_db2

  return W1, b1, W2, b2

# Forward pass.
def forward_pass(X, W1, b1, W2, b2, activation_func):
    Z1 = np.dot(X, W1) + b1
    A1 = activation_func(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z1, A1, Z2

# Backward propagation.
def backpropagation(X, Y, Z1, A1, Z2, W2, activation_deriv, learning_rate):
    dL_dY_pred = 2 * (Z2 - Y) / Y.shape[0]
    dL_dW2 = np.dot(A1.T, dL_dY_pred)
    dL_db2 = np.sum(dL_dY_pred, axis=0, keepdims=True)
    dL_dA1 = np.dot(dL_dY_pred, W2.T)
    dL_dZ1 = dL_dA1 * activation_deriv(A1)
    dL_dW1 = np.dot(X.T, dL_dZ1)
    dL_db1 = np.sum(dL_dZ1, axis=0, keepdims=True)
    
    # Gradient clipping for higher learning rates to prevent instability.
    if learning_rate > 0.1:
        dL_dW1 = np.clip(dL_dW1, -3, 3)
        dL_db1 = np.clip(dL_db1, -3, 3)
        dL_dW2 = np.clip(dL_dW2, -3, 3)
        dL_db2 = np.clip(dL_db2, -3, 3)

    return dL_dW1, dL_db1, dL_dW2, dL_db2


# Training function
def train_network(X_train, Y_train, X_test, Y_test, hidden_size, learning_rate, num_epochs, activation):
    
    activation_func, activation_deriv = activation_functions[activation]
    print(f"Training with {hidden_size} hidden neurons using {activation} activation")

    # Initialize parameters.
    W1, b1, W2, b2 = initialize_params(X_train, hidden_size)

    losses_train = []
    for epoch in range(num_epochs):
        Z1, A1, Z2 = forward_pass(X_train, W1, b1, W2, b2, activation_func)
        
        loss_train = np.mean((Y_train - Z2) ** 2)
        losses_train.append(loss_train)

        # Backpropagation.
        dL_dW1, dL_db1, dL_dW2, dL_db2 = backpropagation(
            X_train, Y_train, Z1, A1, Z2, W2,
            activation_deriv, learning_rate
        )

        # Update the parameters.
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, 
            dL_dW1, dL_db1, dL_dW2, dL_db2
        )

        # Print the loss every 100 epochs.
        if epoch % 100 == 0 or epoch == num_epochs-1:
            print(f"    Epoch {epoch}, Train Loss: {loss_train:.4f}")

    # Generate Predictions
    _, _, Y_pred = forward_pass(X_test, W1, b1, W2, b2, activation_func)

    # Return the parameters and predictions for further analysis.
    return {
        "hidden_size": hidden_size,
        "activation_function": activation,
        "final_loss": loss_train,
        "losses_train": losses_train,
        "weights": (W1, b1, W2, b2),
        "predictions": Y_pred
    }

'''
Define functions for creating the plots.
'''
# Plot training loss
def plot_train_loss(results):
    plt.figure(figsize=(8, 6))
    plt.plot(results["losses_train"], label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Loss for {results['activation_function']} with {results['hidden_size']} Neurons, Learning rate = {learning_rate}")
    plt.legend()
    plt.show()

# Plot predictions vs test data
def plot_predictions(results, Y_test):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_test, results["predictions"], alpha=0.6, label="Predicted Values")
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r', lw=2, label='Test Values')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs. Predicted for {results['activation_function']} with {results['hidden_size']} Neurons, Learning rate = {learning_rate}")
    plt.legend()
    plt.show()

'''
Training the Neural Network by manually choosing:
- hidden_size: number of neurons in the hidden layer
- learning_rate: between 0.001 and 1.0
- num_epochs: 1000 by default, kept constant for all experiments
- activation: Sigmoid, ReLU and TanH activation functions
'''

# Define training parameters
hidden_size = 10  # Choose the number of hidden neurons
learning_rate = 0.01
num_epochs = 1000  
activation = "relu"  # Choose from "sigmoid", "relu", "tanh"

# Train the model
results = train_network(X_train, Y_train, X_test, Y_test, hidden_size, learning_rate, num_epochs, activation)

# Plot training loss
plot_train_loss(results)

# Plot predictions vs test data
plot_predictions(results, Y_test)
