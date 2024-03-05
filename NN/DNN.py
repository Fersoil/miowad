import networks
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)


X = np.linspace(-1,1,4).reshape(1,4)

y = 2*X**2


layers = [
    {"output_dim": 2, "activation": "relu", "init": "he"},
    {"output_dim": 2, "activation": "relu", "init": "he"},
    {"activation": "linear"}
]

mlp = networks.MLP(layers, input=X)

for layer in mlp.layers:
    print(layer.weights)

losses = mlp.train(X, y, max_epochs=1000, learning_rate=0.01, batch_size=32)

plt.plot(losses)

plt.show()

y_hat = mlp.full_forward_pass(X)

plt.scatter(X,y)
plt.scatter(X,y_hat, c="red")
plt.show()




# # load datasets

# data_dir = Path("data")
# square_simple_train = pd.read_csv(data_dir / "regression" / "square-simple-training.csv", index_col=0)
# square_simple_test = pd.read_csv(data_dir / "regression" / "square-simple-test.csv", index_col=0)
# steps_small_train = pd.read_csv(data_dir / "regression" / "steps-small-training.csv", index_col=0)
# steps_small_test = pd.read_csv(data_dir / "regression" / "steps-small-test.csv", index_col=0)
# multimodal_large_train = pd.read_csv(data_dir / "regression" / "multimodal-large-training.csv")
# multimodal_large_test = pd.read_csv(data_dir / "regression" / "multimodal-large-test.csv")


# layers = [
#     {"output_dim": 20, "activation": "relu", "init": "he"},
#     {"output_dim": 20, "activation": "relu", "init": "he"},
#     {"activation": "linear"}
# ]

# mlp = networks.MLP(layers, input=multimodal_large_train[["x"]])


# norm = networks.assets.Normalizator(multimodal_large_train)

# multimodal_large_train_norm = norm(multimodal_large_train)
# multimodal_large_test_norm = norm(multimodal_large_test)


# losses = mlp.train(multimodal_large_train_norm[["x"]].T.to_numpy(), multimodal_large_train_norm[["y"]].T.to_numpy(), max_epochs=100000, learning_rate=0.01, batch_size=256)