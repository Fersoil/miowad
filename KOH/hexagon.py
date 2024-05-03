import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from som import KohonenNetwork


data_dir = Path("data")


cube = pd.read_csv(data_dir / "cube.csv")
hexagon = pd.read_csv(data_dir / "hexagon.csv")



koh = KohonenNetwork(3, 2)


koh.fit(hexagon[["x", "y"]].values, 50)

print("Finished training")

reshaped_cells = koh.cells.reshape(-1, 2)

plt.scatter(reshaped_cells[:, 0], reshaped_cells[:, 1])
plt.show()




print("Finished")
print("Cube")
