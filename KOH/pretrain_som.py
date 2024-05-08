import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from som import KohonenNetwork, GaussianNeighboringFunc, MinusOneGaussianNeighboringFunc
import pickle

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

np.random.seed(44)

def cross_val(X, y, cv=5, kmeans = True, mapping = True, epochs=10, M=10, N=10, grid = "rectangular", verbose=False, save_path=None):
    # train test split
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    idx = idx % cv
    
    for i in range(cv):
        # train test split
        X_train = X[idx != i]
        y_train = y[idx != i]
        
        X_test = X[idx == i]
        y_test = y[idx == i]        
        
        koh = KohonenNetwork(M, N, vec_dim=X.shape[1], neighboring_func=GaussianNeighboringFunc(), grid=grid)
        koh.fit(X_train, epochs=epochs, verbose=verbose)
        
                
        if save_path:
            with open(f'{save_path}_{i}.pkl', 'wb') as f:
                pickle.dump(koh, f)
    return idx



# load the MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)




