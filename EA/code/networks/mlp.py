import numpy as np
from .layers import FCLayer
from .losses import MSE, CrossEntropy
import matplotlib.pyplot as plt
from .assets import one_hot
from .regularizations import L1, L2, NoneReg


class MLP:
    __slots__ = ["layers", "depth", "loss", "output_type", "output_dim", "regularization"]

    def __init__(self, init_layers, input, output_type="regression", regularization = "l2", regularization_alpha = 0.01, weights=None, biases=None, verbose = False):
            """
            Initialize the Multi-Layer Perceptron (MLP) network.

            Args:
                init_layers (list): List of dictionaries representing the initialization parameters for each layer.
                    Each dictionary should contain the following keys:
                        - "input_dim" (int, optional): The input dimension of the layer.
                        - "output_dim" (int, optional): The output dimension of the layer.
                        - "activation" (str): The activation function to be used in the layer.
                input (array-like): The input data for the network.
                output_type (str, optional): The type of the output. Defaults to "regression". Possible values are "regression" or "classification".
                regularization (str, optional): The type of regularization to be used. Defaults to "l2". Possible values are "l1", "l2" or None.
                regularization_alpha (float, optional): The regularization parameter. Defaults to 0.01.
                weights (array-like, optional): The initial weights for the network. If not provided, the weights will be randomly initialized.
                biases (array-like, optional): The initial biases for the network. If not provided, the biases will be randomly initialized.
                verbose (boolean, optional): Whether to print additional information. Defaults to False.
            """

            if input.ndim == 1:
                x_size = 1
            else:
                x_size = input.shape[0]
    
            self.layers = []

            self.output_type = output_type

            if regularization == "l1":
                self.regularization = L1(regularization_alpha)
            elif regularization == "l2":
                self.regularization = L2(regularization_alpha)
            elif regularization is None:
                self.regularization = NoneReg()
            else:
                raise ValueError("Regularization type not recognized")

            if output_type == "regression":
                if init_layers[-1].get("activation") == "softmax":
                    raise ValueError("The softmax activation function is not suitable for regression problems.")
                if init_layers[-1].get("activation") is not None and init_layers[-1]["activation"] != "linear":
                    print("Consider using the linear activation function for the output layer in regression problems.")

                self.loss = MSE()

            elif output_type == "classification":
                if init_layers[-1].get("activation") is not None and init_layers[-1]["activation"] != "softmax" and init_layers[-1]["activation"] != "sigmoid":
                    print("Consider using the softmax or sigmoid activation function for the output layer in classification problems.")
            
                self.loss = CrossEntropy()

            for i, init_layer in enumerate(init_layers):
                input_dim = init_layer.get("input_dim")
                if i == 0:
                    input_dim = x_size
                else:
                    input_dim = init_layers[i-1]["output_dim"]
                
                if i == len(init_layers) - 1 and output_type == "regression":
                    if verbose:
                        print("Output dimension for the output layer is set to 1")
                    output_dim = 1 # only one dim output is allowed for the output layer
                else:
                    output_dim = init_layer.get("output_dim")
                    self.output_dim = output_dim
                if output_dim is None:
                    raise ValueError("Output dimension not specified for layer {}, softmax layers needs number outout_dim to be the number of classes".format(i))
                
                activation = init_layer.get("activation")
                if activation is None:
                    raise ValueError("Activation function not specified for layer {}".format(i))
                
                init_method = init_layer.get("init")
                if init_method is None:
                    init_method = "uniform"
                
                if weights is not None and biases is not None:
                    try:
                        if verbose:
                            print(f"initializing layer {i} using predefined weights and biases")
                        layer = FCLayer(input_dim, output_dim, activation, weights[i], biases[i], )
                    except IndexError:
                        raise ValueError("The number of weights and biases provided does not match the number of layers")
                else:
                    if verbose:
                        print(f"initializing layer {i} using {init_method} initialization")
                    layer = FCLayer(input_dim, output_dim, activation=activation, init=init_method)
                self.layers.append(layer)

                self.depth = len(self.layers)

    def __str__(self) -> str:
        description = "MLP with layers:\n"
        for layer in self.layers:
            description += str(layer) + "\n"
        return description
    
    def save_weights(self):
        """
        saves weights and biases of network, returns it as two lists
        """
        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.weights)
            biases.append(layer.bias)

        return weights, biases
    
    def load_weights(self, weights, biases):
        """
        loads the weights and biases, the input should be two lists with weights and bias of each layer
        """
        depth = len(weights)
        assert depth == len(biases) and depth == self.depth, "depth of the given weights is incorrect"

        for i in range(depth):
            assert self.layers[i].weights.shape == weights[i].shape, f"incorrect weights shape for layer number {i}"
            assert self.layers[i].bias.shape == biases[i].shape, f"incorrect bias shape for layer number {i}"

            self.layers[i].weights = weights[i]
            self.layers[i].bias = biases[i]



    def full_forward_pass(self, input: np.ndarray):
        """
        input: an input matrix X
        """
        
        output = input
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def calculate_loss(self, input, y, output = None):
        if self.output_type == "classification" and y.shape[0] == 1:
            y = one_hot(y, self.output_dim)

        if output is None:
            output = self.full_forward_pass(input)

        reg_loss = np.sum([self.regularization.loss(layer.weights) for layer in self.layers])

        return self.loss.calculate_loss(y, output) + reg_loss


    def Fscore(self, y_pred, y_true, apply_one_hot_encoding = False):
        """
        F-score is the harmonic mean of precision and recall. 
        The F-score is the balance between precision and recall.
        inputs should be one hot encoded
        """
        assert y_pred.shape == y_true.shape, "The shapes of the predicted and true values do not match"
        if apply_one_hot_encoding:
            y_pred = one_hot(y_pred, self.output_dim)
            y_true = one_hot(y_true, self.output_dim)

        assert y_pred.shape[0] == self.output_dim, "The number of classes in the one-hot encoded vectors does not match the number of classes in the output layer"
        true_positives = np.sum(y_true * y_pred)
        false_positives = np.sum((1 - y_true) * y_pred)
        false_negatives = np.sum(y_true * (1 - y_pred))

        return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)


    def predict(self, input):
        if self.output_type == "classification":
            return self.full_forward_pass(input).argmax(axis=0)
        else:
            return self.full_forward_pass(input)

    def full_backward_propagation(self, input: np.ndarray, y: np.ndarray, verbose = False):
        """function performs one full backward pass through the network and updates the params. Uses optionally the momentum method and RMSprop.

        Args:
            input (np.ndarray): matrix of input data
            y (np.ndarray): single column matrix of target values
            verbose (bool, optional): print the shapes of the gradients. Defaults to False.
            
        Returns:
            (tuple): tuple of gradients of weights and biases
        """

        y_hat = self.full_forward_pass(input) # forward pass automatically stores the output in the layers
        
        if verbose:
            print("y_hat: ", input.shape, y_hat.shape, y.shape)
        
        db_table = [None] * self.depth # derivatives of output in respect to weights
        dw_table = [None] * self.depth # derivatives of output in respect to biases
        

        num_of_iterations = self.depth
            
        g =  self.loss.calculate_output_layer_prime(y, y_hat)
        

        for i in reversed(range(num_of_iterations)):

            dw, db, g = self.layers[i].backward_propagation(g)

            dw = dw + self.regularization.grad_loss(self.layers[i].weights)
            
            db_table[i] = db
            dw_table[i] = dw


        return dw_table, db_table
    
    
    
    def update_params(self, dw_table, db_table):
        """function updates parameters of the network using the given already rescaled with learning rate and using optimization algorithm gradients"""
            
        for i in range(self.depth):
            self.layers[i].update_params(dw_table[i], db_table[i])
    

    def train(self, input, y, input_val = None, y_val = None, max_epochs = 100, batch_size = 32, learning_rate = 0.01, stochastic_descent = False, momentum = False, 
              momentum_decay = 0.9, rms_prop = False, squared_gradient_decay = 0.99, adam = False, epsilon=1e-8, early_stopping = True, min_stopping_delta = 1e-5, patience = 5, verbose=True, plot_losses = True):
        """
                Trains the MLP network using the given input and target values.

        Parameters:
        - input (ndarray): The input values for training the network. Should be in the shape (n, m) where n is the number of features and m is the number of samples.
        - y (ndarray): The target values for training the network. Should be in the shape (1, m) for regression problems or (n, m) for classification problems with n being the number of classes.
        - input_val (ndarray, optional): The input values from the validation dataset to measure performance of the network. Default is None - no validation is being done.
        - y_val (ndarray, optional): The target values from the validation dataset to measure performance of the the network. Default is None - no validation is being done.
        - max_epochs (int, optional): The maximum number of epochs to train the network. Default is 100.
        - batch_size (int, optional): The batch size for gradient descent. Default is 32.
        - learning_rate (float, optional): The learning rate for updating the network weights. Default is 0.01.
        - stochastic_descent (bool, optional): Whether to use stochastic gradient descent. Default is False.
        - momentum (bool, optional): Whether to use momentum optimization. Default is False.
        - momentum_decay (float, optional): The decay rate for momentum optimization. Default is 0.9.
        - rms_prop (bool, optional): Whether to use RMSprop optimization. Default is False.
        - squared_gradient_decay (float, optional): The decay rate for squared gradient in RMSprop optimization. Default is 0.99.
        - adam (bool, optional): Whether to use Adam optimization. Default is False.
        - epsilon (float, optional): A small value to avoid division by zero in Adam and RMSprop optimization. Default is 1e-8.
        - early_stopping (bool, optional): Whether to use early stopping based on loss from validation dataset. Default is True.
        - min_stopping_delta (float, optional): The threshold for early stopping based on loss from validation dataset. Default is 1e-5.
        - patience (int, optional): The number of epochs to wait before stopping the training if the loss from the validation dataset does not improve. Default is 5.
        - verbose (bool, optional): Whether to print training progress. Default is True.
        - plot_losses (bool, optional): Whether to plot the training and validation losses. Default is True.

        Returns:
        - losses (list): The training losses for each epoch.

        """

        if not isinstance(input, np.ndarray):
            input = np.array(input)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        assert input.shape[0] == self.layers[0].input_dim, f"The input data should have {self.layers[0].input_dim} features, but {input.shape[0]} were given."

        assert max_epochs > 0 and isinstance(max_epochs, int), "The maximum number of epochs should be a positive integer."
        assert batch_size > 0 and isinstance(batch_size, int), "The batch size should be a positive integer."
        

        assert self.output_type == "regression" or self.output_type == "classification", "The output type should be either 'regression' or 'classification'."

        if self.output_type == "classification":
            if y.shape[0] == 1:
                y = one_hot(y, self.output_dim)
            else:
                assert y.shape[0] == self.output_dim, f"The target values should have {self.output_dim} classes, but {y.shape[0]} were given."
            if input_val is not None and y_val is not None:
                y_val = one_hot(y_val, self.output_dim)

        elif self.output_type == "regression":
            assert y.shape[0] == self.layers[-1].output_dim, f"The target values should have {self.layers[-1].output_dim} classes, but {y.shape[0]} were given."



        validation = False
        if input_val is not None and y_val is not None:
            validation = True
            if not isinstance(input_val, np.ndarray):
                input_val = np.array(input_val)
            if not isinstance(y_val, np.ndarray):
                y_val = np.array(y_val)

        assert patience >= 0,  "The patience parameter should be non-negative."
        assert isinstance(patience, int), "The patience parameter should be an integer"
        
        losses = []
        test_losses = []
        counter = 0

        m = input.shape[1]

        if momentum or adam:
            momentum_gradients = [np.zeros_like(layer.weights) for layer in self.layers]
        if rms_prop or adam:
            squared_gradients = [np.ones_like(layer.weights) for layer in self.layers]

        for epoch in range(max_epochs):
            if stochastic_descent:
                permutation = np.random.permutation(input.shape[1])
                input = input[:, permutation]
                y = y[:, permutation]
            
            for batch_number in range(0, m, batch_size):
                input_batch = input[:, batch_number:batch_number+batch_size]
                y_batch = y[:, batch_number:batch_number+batch_size]
                
                dw, db = self.full_backward_propagation(input_batch, y_batch)
                

                if momentum:
                    momentum_gradients = [dw[i] * (1 - momentum_decay) + momentum_gradients[i] * momentum_decay for i in range(self.depth)]
                    dw = [momentum_gradients[i] for i in range(self.depth)]
                elif rms_prop:
                    squared_gradients = [squared_gradient_decay * squared_gradients[i] + (1 - squared_gradient_decay) * squared_gradients[i] ** 2 for i in range(self.depth)]
                    dw = [dw[i] / (np.sqrt(squared_gradients[i]) + epsilon) for i in range(self.depth)]
                elif adam:
                    momentum_gradients = [dw[i] * (1 - momentum_decay) + momentum_gradients[i] * momentum_decay for i in range(self.depth)]
                    squared_gradients = [squared_gradient_decay * squared_gradients[i] + (1 - squared_gradient_decay) * squared_gradients[i] ** 2 for i in range(self.depth)]

                    counter += 1

                    # bias correction
                    corrected_momentum_gradients = [momentum_gradients[i] / (1 - momentum_decay**counter) for i in range(self.depth)]
                    corrected_squared_gradients = [squared_gradients[i] / (1 - squared_gradient_decay**counter) for i in range(self.depth)]

                    dw = [corrected_momentum_gradients[i] / (np.sqrt(corrected_squared_gradients[i]) + epsilon) for i in range(self.depth)]
                
                
                dw = [dw[i] * learning_rate for i in range(self.depth)]                
                db = [db[i] * learning_rate for i in range(self.depth)]

                
                self.update_params(dw, db)

            loss = self.calculate_loss(input, y)
            if validation:
                test_loss = self.calculate_loss(input_val, y_val)
                if early_stopping:
                    if len(test_losses) > 0 and test_loss > test_losses[-1] - min_stopping_delta:
                        # save weights 
                        prev_weights, prev_biases = self.save_weights()
                        if verbose:
                            print("No observed enough improvement in the validation loss, waiting for ", patience - counter, " epochs. Validation loss difference: ", test_loss - test_losses[-1])

                        counter += 1
                        if counter == patience:
                            if verbose:
                                print(f"Early stopping, no observed enough improvement after {patience} epochs in the validation loss.")
                                print("loading the old weights")
                            self.load_weights(prev_weights, prev_biases)
                            return losses, test_losses
                    else:
                        counter = 0

                test_losses.append(test_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.calculate_loss(input, y)}")
                if validation:
                    print(f"Validation Loss: {self.calculate_loss(input_val, y_val)}")

            losses.append(loss)
            
            
        if plot_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, label="training set")
            if validation:
                plt.plot(test_losses, c="red", label="validation set")

            plt.title("Values of losses in each epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show() 

        if test_losses is not None:
            return losses, test_losses
        return losses
    
           
    
    def get_weights(self):
        return [(layer.get_weights(), layer.get_biases()) for layer in self.layers]
    


    
    
