import numpy as np
from .assets import one_hot

class Loss:
    def __init__(self):
        pass

    def calculate_loss(self, y_true, y_pred):
        raise NotImplementedError
    
    def calculate_loss_prime(self, y_true, y_pred):
        raise NotImplementedError
    
    def __str__(self):
        return "base class Loss"
    

class MSE(Loss):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def calculate_loss_prime(self, y_true, y_pred):
        y_true = np.array(y_true)
        batch_size = y_true.shape[0]

        return 2 * (y_pred - y_true) / batch_size
    
    def calculate_output_layer_prime(self, y_true, y_pred):
        y_true = np.array(y_true)
        batch_size = y_true.shape[1]

        return 2 * (y_pred - y_true) / batch_size
    
    def __str__(self):
        return "Mean Squared Error Loss"
    

class CrossEntropy(Loss):
    def __init__(self):
        super().__init__()
    
    def calculate_loss(self, y_true, y_pred):
        # assumes y_true and y_pred are one-hot encoded, y_pred is softmax output
        # return -np.log(y_pred[np.where(y_true)])

        assert y_true.shape == y_pred.shape, "Error while calculating cross entropy loss, y_true and y_pred shapes do not match."

        eps = 1e-10

        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = np.array(y_true)
        batch_size = y_true.shape[1]

        return np.sum(-y_true * np.log(y_pred)) / batch_size
    
    def calculate_loss_prime(self, y_true, y_pred):
        y_true = np.array(y_true)

        eps = 1e-10
        y_pred = np.clip(y_pred, eps, 1 - eps)
        batch_size = y_true.shape[1]

        return (-np.divide(y_true, y_pred) + np.divide(1 - y_true, 1 - y_pred) )/ batch_size
    
    def calculate_output_layer_prime(self, y_true, y_pred):

        y_true = np.array(y_true)
        batch_size = y_true.shape[1]

        return (y_pred - y_true)/batch_size
    
    def __str__(self):
        return "Cross Entropy Loss"