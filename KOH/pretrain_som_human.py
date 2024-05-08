import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from som import KohonenNetwork, GaussianNeighboringFunc, MinusOneGaussianNeighboringFunc
import pickle

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

np.random.seed(44)

def cross_val(X, y, cv=5, epochs=10, M=10, N=10, grid = "rectangular", neighbouring_func = GaussianNeighboringFunc(initial_neighbouring_radius=0.3), lambda_param = 10, verbose=False, save_path=None):
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
        
        koh = KohonenNetwork(M, N, vec_dim=X.shape[1], neighboring_func=neighbouring_func, lambda_param = lambda_param, grid=grid)
        koh.fit(X_train, epochs=epochs, verbose=verbose)
        
                
        if save_path:
            with open(f'{save_path}_{i}.pkl', 'wb') as f:
                pickle.dump(koh, f)
    return idx


train = pd.read_table('data/human_train.txt', header=None, delim_whitespace=True)
test = pd.read_table('data/human_test.txt', header=None, delim_whitespace=True)

y_train = pd.read_table('data/human_train_target.txt', header=None, delim_whitespace=True)
y_test = pd.read_table('data/human_test_target.txt', header=None, delim_whitespace=True)


X = np.concatenate([train.values, test.values])

y = np.concatenate([y_train.values, y_test.values]).flatten()


idx = cross_val(X, y, cv=5, epochs=15, M=10, N=10, grid = "rectangular", verbose=False, save_path='som_snapshots/net10x10/human_som')
# save the indices
np.save('som_snapshots/net10x10/human_som_rect_idx.npy', idx)


idx = cross_val(X, y, cv=5, epochs=15, M=25, N=25, grid = "rectangular", verbose=False, save_path='som_snapshots/net25x25/human_som')
np.save('som_snapshots/net25x25/human_som_rect_idx.npy', idx)



idx = cross_val(X, y, cv=5, epochs=15, M=10, N=10, grid = "rectangular", neighbouring_func=MinusOneGaussianNeighboringFunc(initial_neighbouring_radius=0.3), lambda_param = 10, verbose=False, save_path='som_snapshots/net10x10/human_som_minusone')
np.save('som_snapshots/net10x10/human_som_minusone_rect_idx.npy', idx)

