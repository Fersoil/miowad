import networks
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(123)


# load datasets

data_dir = Path("data")
multimodal_train = pd.read_csv(data_dir / "regression" / "multimodal-sparse-training.csv")
multimodal_test = pd.read_csv(data_dir / "regression" / "multimodal-sparse-test.csv")
rings3_train = pd.read_csv(data_dir / "classification" / "rings3-balance-training.csv")
rings5_train = pd.read_csv(data_dir / "classification" / "rings5-sparse-training.csv")
rings3_test = pd.read_csv(data_dir / "classification" / "rings3-balance-test.csv")
rings5_test = pd.read_csv(data_dir / "classification" / "rings5-sparse-test.csv")
xor3_train = pd.read_csv(data_dir / "classification" / "xor3-balance-training.csv")
xor3_test = pd.read_csv(data_dir / "classification" / "xor3-balance-test.csv")


multimodal_val = multimodal_test.sample(frac=0.2)
multimodal_test = multimodal_test.drop(multimodal_val.index)

rings3_val = rings3_test.sample(frac=0.2)
rings3_test = rings3_test.drop(rings3_val.index)

rings5_val = rings5_test.sample(frac=0.2)
rings5_test = rings5_test.drop(rings5_val.index)


xor3_val = xor3_test.sample(frac=0.2)
xor3_test = xor3_test.drop(xor3_val.index)




norm = networks.assets.Normalizator(rings3_train[['x', 'y']])
train = norm(xor3_train[["x", "y"]]).to_numpy().T
y_train = xor3_train[["c"]].to_numpy().T
test = norm(xor3_test[["x", "y"]])
y_test = xor3_test[["c"]].to_numpy().T
val = norm(xor3_val[["x", "y"]]).to_numpy().T
y_val = xor3_val[["c"]].to_numpy().T



layers = [
    {"output_dim": 10, "activation": "relu", "init": "he"},
    {"output_dim": 10, "activation": "relu", "init": "he"},
    {"output_dim": 10, "activation": "relu", "init": "he"},
    {"output_dim": 2, "activation": "softmax", "init": "uniform"}
]

max_epochs = 10000

regularizations = [None, "l1", "l2"]
early_stops = [False, True]

for reg in regularizations:
    for early_stop in early_stops:

        mlp = networks.MLP(layers, input=train, regularization=reg, output_type="classification", regularization_alpha=0.001)
        losses, test_losses = mlp.train(train, y_train, val, y_val,
                        max_epochs=max_epochs, learning_rate=0.01, early_stopping=early_stop, plot_losses=False, verbose=False, min_stopping_delta=1e-8, patience=5)

        plt.figure(figsize=(10, 6))
        plt.plot(losses, label="training set")
        plt.plot(test_losses, c="red", label="validation set")

        epochs = len(losses)

        plot_title = "Values of losses in each epoch "
        if early_stop:
            plot_title = plot_title + f"with early stop at epoch: {epochs} "
        else:
            plot_title = plot_title + "without early stopping, "
            if reg is None:
                plot_title = plot_title + "nor regularization"

        if reg is None and early_stop:
            plot_title = plot_title + "without regularization"
        elif reg is not None:
            plot_title = plot_title + f"with {reg} regularization"


        plt.title(plot_title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f'plots/xor3/{reg}_{early_stop}.png', bbox_inches='tight')


        # calculate MSE

        y_pred = mlp.full_forward_pass(test.to_numpy().T)
        y_pred_train = mlp.full_forward_pass(train)
        y_pred_val = mlp.full_forward_pass(val)
        y_hat = mlp.predict(test.to_numpy().T)
        y_test_onehot = networks.assets.one_hot(y_test, 2)
        y_train_onehot = networks.assets.one_hot(y_train, 2)
        y_val_onehot = networks.assets.one_hot(y_val, 2)
        print(plot_title)
        print("Fscore train: ", mlp.Fscore(y_pred_train, y_train_onehot))
        print("Fscore val: ", mlp.Fscore(y_pred_val, y_val_onehot))
        print("Fscore test: ", mlp.Fscore(y_pred, y_test_onehot))
        print("accuracy for test: ", np.sum(y_hat == y_test.flatten()) / len(y_hat))
