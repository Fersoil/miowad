import numpy as np 
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from time import time

import pickle

random.seed(0)
np.random.seed(0)

from networks import Population

import numpy as np
import pandas as pd
from pathlib import Path
import os

import sys
sys.path.append("..")

import matplotlib.pyplot as plt

import networks

data_dir = Path("../data")

multimodal_train = pd.read_csv(data_dir / "multimodal-large-training.csv", index_col=False)
multimodal_test = pd.read_csv(data_dir / "multimodal-large-test.csv", index_col=False)

from sklearn import datasets

iris = datasets.load_iris()

X = iris["data"]
y = iris["target"]

# example definition of a neural network
layers = [
    {"output_dim": 20, "activation": "relu", "init": "he"},
    {"output_dim": 20, "activation": "relu", "init": "he"},
    {"activation": "linear", "init": "normal"}
]



X_train = multimodal_train[["x"]].to_numpy()
X_test = multimodal_test[["x"]].to_numpy()

y_train =  multimodal_train[["y"]].to_numpy()
y_test = multimodal_test[["y"]].to_numpy()

population = Population(layers, X_train, y_train, X_test, y_test)

population.generate_individuals(30)
print(population)

best_scores, avg_scores = population.run(100, verbose = True, how_many_individuals_mutate=0.7, temp=None, magnitude=0.1)



plt.scatter(X_train, y_train, label="training data")
plt.scatter(X_train, population.predict(population.individuals[0]))
plt.show()
best_scores, avg_scores = population.run(100, verbose = True, how_many_individuals_mutate=0.7, temp=None, magnitude=0.05)


plt.scatter(X_train, y_train, label="training data")
plt.scatter(X_train, population.predict(population.individuals[0]))
plt.show()
best_scores, avg_scores = population.run(100, verbose = True, how_many_individuals_mutate=0.7, temp=None, magnitude=0.01)

plt.plot(best_scores, label = "best")
plt.plot(avg_scores, label = "avg")

plt.legend()
plt.show()

