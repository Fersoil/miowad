import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from som import KohonenNetwork


data_dir = Path("data")


cube = pd.read_csv(data_dir / "cube.csv")
hexagon = pd.read_csv(data_dir / "hexagon.csv")

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(cube["x"], cube["y"], cube["z"])
plt.show()

koh = KohonenNetwork(4, 2, vec_dim=3, lambda_param=1)


koh.fit(cube[["x", "y", "z"]].values, 100)

print("Finished training")

koh.plot_graph()

print("Finished")
