from abc import ABC, abstractmethod
import numpy as np

class Regularization(ABC):   
    
    @abstractmethod
    def apply(self, data):
        pass
    
    def loss(self, weights):
        raise NotImplementedError("The loss function has not been implemented")
    
    def grad_loss(self, weights):
        raise NotImplementedError("The gradient of the loss function has not been implemented")
    
    def __str__(self):
        return self.__class__.__name__
    
    
class NoneReg(Regularization):
    def apply(self, data):
        return data
    
    def loss(self, weights):
        return 0
    
    def grad_loss(self, weights):
        return [np.zeros_like(w) for w in weights]
    

class L1(Regularization):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def loss(self, weights):
        return self.alpha * np.sum(np.abs(weights))
    
    def grad_loss(self, weights):
        return self.alpha * np.sign(weights)
    

class L2(Regularization):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def loss(self, weights):
        return self.alpha * np.sum(weights**2)
    
    def grad_loss(self, weights):
        return 2 * self.alpha * weights