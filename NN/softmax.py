import networks
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


# load datasets

data_dir = Path("data")
easy_train = pd.read_csv(data_dir / "classification" / "easy-training.csv")
easy_test = pd.read_csv(data_dir / "classification" / "easy-test.csv")
rings_train = pd.read_csv(data_dir / "classification" / "rings3-regular-training.csv")
rings_test = pd.read_csv(data_dir / "classification" / "rings3-regular-test.csv")
xor_train = pd.read_csv(data_dir / "classification" / "xor3-training.csv")
xor_test = pd.read_csv(data_dir / "classification" / "xor3-test.csv")



layers = [
    {"output_dim": 10, "activation": "relu", "init": "he"},
    {"output_dim": 10, "activation": "relu", "init": "he"},
    {"output_dim": 3, "activation": "softmax", "init": "normal"}
]


norm = networks.assets.Normalizator(rings_train[["x", "y"]])

train = norm(rings_train[["x", "y"]])
y_train = rings_train.c.values.reshape(-1, 1)
test = norm(rings_test[['x', 'y']])
y_test = rings_test.c.values.reshape(-1, 1)


max_epochs = 500


mlp = networks.MLP(layers, input=train.T, output_type="classification", regularization="l2", regularization_alpha=0.001)
losses = mlp.train(train.T, y_train.T, max_epochs=max_epochs, learning_rate=0.001)
y_hat = mlp.predict(train.T)
plt.plot(losses)
plt.show()
print(y_hat.shape)


plt.scatter(train.x, train.y, c=y_hat)
plt.show()


plt.scatter(train.x, train.y, c=y_train)
plt.show()

print("accuracy: ", np.mean(y_train.flatten() == y_hat))
print

