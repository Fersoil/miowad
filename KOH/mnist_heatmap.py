import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from som import KohonenNetwork, GaussianNeighboringFunc, MinusOneGaussianNeighboringFunc
import pickle

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

np.random.seed(44)

# load the MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)


X = mnist.data.values
y = mnist.target.values.to_numpy().astype(int)


with open('som_snapshots/net25x25/hex/mnist_som_0.pkl', 'rb') as f:
    koh = pickle.load(f)
    
koh.plot_heatmap(X, y, "tab20b")

plt.title("MNIST mapping with Kohonen network")

plt.show()