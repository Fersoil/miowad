import numpy as np
from scipy.special import expit

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return expit(x)

def linear(x):
    return x

def tanh(x):
    return np.tanh(x)

def prelu(x, alpha):
    return np.maximum(0, x) + alpha * np.minimum(0, x)


# prime functions - derivatives
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def linear_prime(x):
    return 1

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def prelu_prime(x, alpha):
    return np.where(x > 0, 1, alpha)