import numpy as np
from scipy.special import expit as sigmoid_scipy
from scipy.special import softmax as softmax_scipy

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


def softmax(x):
    return softmax_scipy(x, axis=0)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_prime(x):
    raise NotImplementedError("The derivative of the softmax function is difficult to implement and is not used in backpropagation.")
    sh = x.shape
    sm = softmax_scipy(x, axis=1)
    DM = sm.reshape(sh[0],-1,1) * np.diag(np.ones(sh[1])) # Diagonal matrices
    OP = np.matmul(sm.reshape(sh[0],-1,1), sm.reshape(sh[0],1,-1)) # Outer products
    Jsm = DM - OP
    return Jsm