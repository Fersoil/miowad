import networks
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt



# load datasets

data_dir = Path("data")
square_simple_train = pd.read_csv(data_dir / "regression" / "square-simple-training.csv", index_col=0)
square_simple_test = pd.read_csv(data_dir / "regression" / "square-simple-test.csv", index_col=0)
steps_small_train = pd.read_csv(data_dir / "regression" / "steps-small-training.csv", index_col=0)
steps_small_test = pd.read_csv(data_dir / "regression" / "steps-small-test.csv", index_col=0)
multimodal_large_train = pd.read_csv(data_dir / "regression" / "multimodal-large-training.csv")
multimodal_large_test = pd.read_csv(data_dir / "regression" / "multimodal-large-test.csv")


layers = [
    {"output_dim": 20, "activation": "relu", "init": "he"},
    {"output_dim": 20, "activation": "relu", "init": "he"},
    {"activation": "linear"}
]

mlp = networks.MLP(layers, input=multimodal_large_train[["x"]])


norm = networks.assets.Normalizator(multimodal_large_train)

multimodal_large_train_norm = norm(multimodal_large_train)
multimodal_large_test_norm = norm(multimodal_large_test)


losses = mlp.train(multimodal_large_train_norm[["x"]].T.to_numpy(), multimodal_large_train_norm[["y"]].T.to_numpy(), max_epochs=100000, learning_rate=0.01, batch_size=256)