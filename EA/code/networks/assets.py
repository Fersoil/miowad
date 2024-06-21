from numpy import expand_dims
import numpy as np


class Normalizator:
    def __init__(self, data):
        self.data = data
        self.mean = data.mean()
        self.std = data.std()
        self.indexes = data.index

    def __call__(self, data):
        return (data - self.mean) / self.std
    
    def denorm(self, data, index=None):
        if index is None:
            index = self.indexes
        if not isinstance(index, list):
            index = [index]

        
        return data * expand_dims(self.std[index], axis=0) + expand_dims(self.mean[index], axis=0)
    



    def __repr__(self):
        return f"Normalizator(mean={self.mean}, std={self.std})"
    

def one_hot(y, num_classes):

        y = y.astype(int)

        #return pd.get_dummies(y.flatten()).values.T

        one_hot = np.zeros((num_classes, y.shape[1]))
        for i in range(y.shape[1]):
            one_hot[y[0, i], i] = 1
        return one_hot