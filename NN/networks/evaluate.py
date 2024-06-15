from .mlp import MLP

from .assets import Normalizator


def evaluate_mlp(X_train, y_train, X_test, y_test, norm, layers,  n_runs = 5, *args, **kwargs):

    
    test_loss = []
    train_loss = []
    for i in range(n_runs):
        mlp = MLP(layers, input=X_train.T)
        mlp.fit(X_train, y_train, *args, **kwargs)
        y_
        
        y_hat_denorm = norm.denorm(y_hat.T, index="y").flatten()

        test_loss.append(mlp.evaluate(X_test, y_test))
        train_loss.append(mlp.evaluate(X_train, y_train))
        
    return test_loss, train_loss