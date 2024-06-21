print("Starting evaluation of cube dataset")


import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from som import KohonenNetwork, GaussianNeighboringFunc
import pickle

from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics import completeness_score, homogeneity_score, v_measure_score

def save_scores(scores, filename):
    with open(Path("scores") / ("cube_" + filename + ".pkl"), "wb") as f:
        pickle.dump(scores, f)

def load_scores(filename):
    with open(Path("scores") / ("cube_" + filename + ".pkl"), "rb") as f:
        return pickle.load(f)
    

data_dir = Path("data")

cube = pd.read_csv(data_dir / "cube.csv")
hexagon = pd.read_csv(data_dir / "hexagon.csv")

n_trials = 10

print("Analysing number of epochs")



import math

def closestDivisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: 
        a -= 1
    return a,n//a



cluster_sizes = [4, 6, 8, 10, 14, 20, 30, 40]
cluster_dims = [closestDivisors(size) for size in cluster_sizes]
print("Evaluating cluster number")

print(cluster_dims)


num_trails = 10
scores = list(range(num_trails))
labels = cube["c"].values

for trail in range(num_trails):
    print(f"Trail {trail}")
    scores[trail] = {"homogeneity": [], "completeness": [], "v_measure": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for dim in cluster_dims:
        size = dim[0] * dim[1]
        epochs = 10 * size
        koh = KohonenNetwork(dim[0], dim[1], GaussianNeighboringFunc(initial_neighbouring_radius=0.25), vec_dim=3)    
        koh.fit(cube[["x", "y", "z"]].values, epochs, history=False, verbose=False)
        som_labels = koh.predict(cube[["x", "y", "z"]].values, return_labels=True)
        scores[trail]["homogeneity"].append(homogeneity_score(cube["c"].values, som_labels))
        scores[trail]["completeness"].append(completeness_score(cube["c"].values, som_labels))
        scores[trail]["v_measure"].append(v_measure_score(cube["c"].values, som_labels))

        scores[trail]["silhouette"].append(silhouette_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["davies_bouldin"].append(davies_bouldin_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["calinski_harabasz"].append(calinski_harabasz_score(cube[["x", "y", "z"]].values, som_labels))



save_scores(filename = "cube_cluster_sizes_025_lambda_1", scores = scores)


for trail in range(num_trails):
    print(f"Trail {trail}")
    scores[trail] = {"homogeneity": [], "completeness": [], "v_measure": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for dim in cluster_dims:
        size = dim[0] * dim[1]
        epochs = 10 * size
        koh = KohonenNetwork(dim[0], dim[1], GaussianNeighboringFunc(initial_neighbouring_radius=0.5), vec_dim=3)    
        koh.fit(cube[["x", "y", "z"]].values, epochs, history=False, verbose=False)
        som_labels = koh.predict(cube[["x", "y", "z"]].values, return_labels=True)
        scores[trail]["homogeneity"].append(homogeneity_score(cube["c"].values, som_labels))
        scores[trail]["completeness"].append(completeness_score(cube["c"].values, som_labels))
        scores[trail]["v_measure"].append(v_measure_score(cube["c"].values, som_labels))

        scores[trail]["silhouette"].append(silhouette_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["davies_bouldin"].append(davies_bouldin_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["calinski_harabasz"].append(calinski_harabasz_score(cube[["x", "y", "z"]].values, som_labels))



save_scores(filename = "cube_cluster_sizes_05_lambda_1",scores =  scores)



for trail in range(num_trails):
    print(f"Trail {trail}")
    scores[trail] = {"homogeneity": [], "completeness": [], "v_measure": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for dim in cluster_dims:
        size = dim[0] * dim[1]
        epochs = 10 * size
        koh = KohonenNetwork(dim[0], dim[1], GaussianNeighboringFunc(initial_neighbouring_radius=0.25), lambda_param=5, vec_dim=3)    
        koh.fit(cube[["x", "y", "z"]].values, epochs, history=False, verbose=False)
        som_labels = koh.predict(cube[["x", "y", "z"]].values, return_labels=True)
        scores[trail]["homogeneity"].append(homogeneity_score(cube["c"].values, som_labels))
        scores[trail]["completeness"].append(completeness_score(cube["c"].values, som_labels))
        scores[trail]["v_measure"].append(v_measure_score(cube["c"].values, som_labels))

        scores[trail]["silhouette"].append(silhouette_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["davies_bouldin"].append(davies_bouldin_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["calinski_harabasz"].append(calinski_harabasz_score(cube[["x", "y", "z"]].values, som_labels))



save_scores(filename = "cube_cluster_sizes_025_lambda_5",scores = scores)



for trail in range(num_trails):
    print(f"Trail {trail}")
    scores[trail] = {"homogeneity": [], "completeness": [], "v_measure": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for dim in cluster_dims:
        size = dim[0] * dim[1]
        epochs = 10 * size
        koh = KohonenNetwork(dim[0], dim[1], GaussianNeighboringFunc(initial_neighbouring_radius=0.5), lambda_param=5, vec_dim=3)    
        koh.fit(cube[["x", "y", "z"]].values, epochs, history=False, verbose=False)
        som_labels = koh.predict(cube[["x", "y", "z"]].values, return_labels=True)
        scores[trail]["homogeneity"].append(homogeneity_score(cube["c"].values, som_labels))
        scores[trail]["completeness"].append(completeness_score(cube["c"].values, som_labels))
        scores[trail]["v_measure"].append(v_measure_score(cube["c"].values, som_labels))

        scores[trail]["silhouette"].append(silhouette_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["davies_bouldin"].append(davies_bouldin_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["calinski_harabasz"].append(calinski_harabasz_score(cube[["x", "y", "z"]].values, som_labels))



save_scores(filename = "cube_cluster_sizes_05_lambda_5", scores = scores)




print("Calculating widths")

widths = np.linspace(0.1, 1, 10)

num_trails = 10

scores = list(range(num_trails))
epochs = 80

labels = cube["c"].values

for trail in range(num_trails):
    print(f"Trail {trail}")
    scores[trail] = {"homogeneity": [], "completeness": [], "v_measure": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for width in widths:
        koh = KohonenNetwork(2, 3, GaussianNeighboringFunc(initial_neighbouring_radius=width), vec_dim=3)    
        koh.fit(cube[["x", "y", "z"]].values, epochs, history=False, verbose=False)
        som_labels = koh.predict(cube[["x", "y", "z"]].values, return_labels=True)
        scores[trail]["homogeneity"].append(homogeneity_score(cube["c"].values, som_labels))
        scores[trail]["completeness"].append(completeness_score(cube["c"].values, som_labels))
        scores[trail]["v_measure"].append(v_measure_score(cube["c"].values, som_labels))

        scores[trail]["silhouette"].append(silhouette_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["davies_bouldin"].append(davies_bouldin_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["calinski_harabasz"].append(calinski_harabasz_score(cube[["x", "y", "z"]].values, som_labels))
        
save_scores(scores = scores, filename="neighbouring_widths")


for trail in range(num_trails):
    print(f"Trail {trail}")
    scores[trail] = {"homogeneity": [], "completeness": [], "v_measure": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for width in widths:
        koh = KohonenNetwork(2, 3, GaussianNeighboringFunc(initial_neighbouring_radius=width), lambda_param=5, vec_dim=3)    
        koh.fit(cube[["x", "y", "z"]].values, epochs, history=False, verbose=False)
        som_labels = koh.predict(cube[["x", "y", "z"]].values, return_labels=True)
        scores[trail]["homogeneity"].append(homogeneity_score(cube["c"].values, som_labels))
        scores[trail]["completeness"].append(completeness_score(cube["c"].values, som_labels))
        scores[trail]["v_measure"].append(v_measure_score(cube["c"].values, som_labels))

        scores[trail]["silhouette"].append(silhouette_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["davies_bouldin"].append(davies_bouldin_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["calinski_harabasz"].append(calinski_harabasz_score(cube[["x", "y", "z"]].values, som_labels))
        
save_scores(scores = scores, filename = "neighbouring_widths_lambda_5")




print("Calculating different neighbouring functions")

from som import MexicanSombreroNeighboringFunc, MinusOneGaussianNeighboringFunc, DistNeighboringFunc
funcs = [GaussianNeighboringFunc(initial_neighbouring_radius=0.3), MinusOneGaussianNeighboringFunc(initial_neighbouring_radius=0.3), DistNeighboringFunc(initial_neighbouring_radius=0.3), MexicanSombreroNeighboringFunc(initial_neighbouring_radius=0.3)]

num_trails = 10

scores = list(range(num_trails))
epochs = 80

labels = cube["c"].values

for trail in range(num_trails):
    print(f"Trail {trail}")
    scores[trail] = {"homogeneity": [], "completeness": [], "v_measure": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": []}
    for neighbouring_func in funcs:
        koh = KohonenNetwork(2, 3, neighboring_func=neighbouring_func, vec_dim=3)    
        koh.fit(cube[["x", "y", "z"]].values, epochs, history=False, verbose=False)
        som_labels = koh.predict(cube[["x", "y", "z"]].values, return_labels=True)
        scores[trail]["homogeneity"].append(homogeneity_score(cube["c"].values, som_labels))
        scores[trail]["completeness"].append(completeness_score(cube["c"].values, som_labels))
        scores[trail]["v_measure"].append(v_measure_score(cube["c"].values, som_labels))

        scores[trail]["silhouette"].append(silhouette_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["davies_bouldin"].append(davies_bouldin_score(cube[["x", "y", "z"]].values, som_labels))
        scores[trail]["calinski_harabasz"].append(calinski_harabasz_score(cube[["x", "y", "z"]].values, som_labels))
        
save_scores(scores = scores, filename="neighbouring_functions")

