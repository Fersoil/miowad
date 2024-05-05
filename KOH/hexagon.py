import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from som import KohonenNetwork, GaussianNeighboringFunc, MinusOneGaussianNeighboringFunc


data_dir = Path("data")


cube = pd.read_csv(data_dir / "cube.csv")
hexagon = pd.read_csv(data_dir / "hexagon.csv")



koh = KohonenNetwork(3, 2, MinusOneGaussianNeighboringFunc(initial_neighbouring_radius=0.1), init_method="dataset", dataset=hexagon[["x", "y"]].values)    

plt.scatter(hexagon["x"], hexagon["y"], c=hexagon["c"])
plt.title("Initial weights of Kohonen Network")
koh.plot_graph()


koh.fit(hexagon[["x", "y"]].values, 100)

print("Finished training")

labels = koh.predict(hexagon[["x", "y"]].values)


plt.scatter(hexagon["x"], hexagon["y"], c=labels)
plt.title("Clustering with Kohonen Network")
koh.plot_graph()
plt.show()


print("Finished")
