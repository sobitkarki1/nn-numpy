from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")
data.head()

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

_,m_train = X_train.shape

batch_size = 64

def init_param() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0)

def softmax(Z: np.ndarray) -> np.ndarray:
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z: np.ndarray) -> np.ndarray:
    return Z > 0

def back_prop(Z1: np.ndarray, A1: np.ndarray, Z2: np.ndarray, A2: np.ndarray, W1: np.ndarray, W2: np.ndarray, X: np.ndarray, Y: np.ndarray, m_train: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = W1 - dW1 * alpha
    b1 = b1 - db1 * alpha
    W2 = W2 - dW2 * alpha
    b2 = b2 - db2 * alpha
    return W1, b1, W2, b2

# Gradient Descent

def get_predictions(A2: np.ndarray) -> np.ndarray:
    return np.argmax(A2, 0)

def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X: np.ndarray, Y: np.ndarray, iterations: int, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1, b1, W2, b2 = init_param()
    m_train = X.shape[1]
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y, m_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 20 == 0:
            loss = -np.sum(one_hot(Y) * np.log(A2)) / m_train
            predictions = get_predictions(A2)
            print(f"Iteration: {i}, Loss: {loss}, Accuracy: {get_accuracy(predictions, Y)}")
    return W1, b1, W2, b2 

# Define constants
LEARNING_RATE = 0.05
ITERATIONS = 1000

def train_model(X_train: np.ndarray, Y_train: np.ndarray, iterations: int, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return gradient_descent(X_train, Y_train, iterations, alpha)

def validate_model(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, X_dev: np.ndarray, Y_dev: np.ndarray) -> float:
    Z1_dev, A1_dev, Z2_dev, A2_dev = forward_prop(W1, b1, W2, b2, X_dev)
    predictions_dev = get_predictions(A2_dev)
    accuracy = get_accuracy(predictions_dev, Y_dev)
    print("Validation Accuracy: ", accuracy)
    return accuracy

# Use constants in gradient_descent call
if __name__ == "__main__":
    W1, b1, W2, b2 = train_model(X_train, Y_train, ITERATIONS, LEARNING_RATE)
    validate_model(W1, b1, W2, b2, X_dev, Y_dev)
