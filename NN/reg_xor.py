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
        sum_losses = {}
        test_sum_losses = {}
        
        fscore_train = []
        fscore_val = []
        fscore_test = []
        accuracy = []
        epochs = []
        
        for run in range(5):
            mlp = networks.MLP(layers, input=train, regularization=reg, output_type="classification", regularization_alpha=0.001)
            losses, test_losses = mlp.train(train, y_train, val, y_val,
                            max_epochs=max_epochs, learning_rate=0.01, early_stopping=early_stop, plot_losses=False, verbose=False, min_stopping_delta=1e-8, patience=5)
            # check if losses are too short
            epochs.append(len(losses))
            if len(losses) < max_epochs:
                losses = np.pad(losses, (0, max_epochs - len(losses)), 'constant', constant_values=(0, 0))
                test_losses = np.pad(test_losses, (0, max_epochs - len(test_losses)), 'constant', constant_values=(0, 0))
            sum_losses[run] = losses
            test_sum_losses[run] = test_losses
            
            
            
            # calculate MSE

            y_pred = mlp.full_forward_pass(test.to_numpy().T)
            y_pred_train = mlp.full_forward_pass(train)
            y_pred_val = mlp.full_forward_pass(val)
            y_hat = mlp.predict(test.to_numpy().T)
            y_test_onehot = networks.assets.one_hot(y_test, 2)
            y_train_onehot = networks.assets.one_hot(y_train, 2)
            y_val_onehot = networks.assets.one_hot(y_val, 2)
            fscore_train.append(mlp.Fscore(y_pred_train, y_train_onehot))
            fscore_val.append(mlp.Fscore(y_pred_val, y_val_onehot))
            fscore_test.append(mlp.Fscore(y_pred, y_test_onehot))
            accuracy.append(np.sum(y_hat == y_test.flatten()) / len(y_hat))
            
        fscore_train = np.array(fscore_train)
        fscore_val = np.array(fscore_val)
        fscore_test = np.array(fscore_test)
        accuracy = np.array(accuracy)
        
        print(f"Results for {reg} regularization and early stop {early_stop}")
        print(f"Fscore train:", "{:.2f}±{:.2f}".format(fscore_train.mean(), fscore_train.std()))
        print(f"Fscore val:", "{:.2f}±{:.2f}".format(fscore_val.mean(), fscore_val.std()))
        print(f"Fscore test:", "{:.2f}±{:.2f}".format(fscore_test.mean(), fscore_test.std()))
        print(f"accuracy for test:", "{:.2f}±{:.2f}".format(np.mean(accuracy), np.std(accuracy)))
        print(f"epochs:", "{:.2f}±{:.2f}".format(np.mean(epochs), np.std(epochs)))
            
            
        sum_losses = pd.DataFrame(sum_losses)
        test_sum_losses = pd.DataFrame(test_sum_losses)
        plt.figure(figsize=(10, 6))
        plt.plot(sum_losses.index, sum_losses.mean(axis = 1), label="training set")
        plt.fill_between(sum_losses.index, sum_losses.mean(axis = 1) - sum_losses.std(axis = 1), sum_losses.mean(axis = 1) + sum_losses.std(axis = 1), alpha=0.3)
        
        plt.plot(test_sum_losses.index, test_sum_losses.mean(axis = 1), c="red", label="validation set")
        plt.fill_between(test_sum_losses.index, test_sum_losses.mean(axis = 1) - test_sum_losses.std(axis = 1), test_sum_losses.mean(axis = 1) + test_sum_losses.std(axis = 1), alpha=0.3, color="red")

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


