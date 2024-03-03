import numpy as np
from .layers import FCLayer


class MLP:
    __slots__ = ["layers", "seed", "depth"]

    def __init__(self, init_layers, input, weights=None, biases=None, learning_rate=0.01, seed=42):
            """
            Initialize the Multi-Layer Perceptron (MLP) network.

            Args:
                init_layers (list): List of dictionaries representing the initialization parameters for each layer.
                    Each dictionary should contain the following keys:
                        - "input_dim" (int, optional): The input dimension of the layer.
                        - "output_dim" (int, optional): The output dimension of the layer.
                        - "activation" (str): The activation function to be used in the layer.
                input (array-like): The input data for the network.
                weights (array-like, optional): The initial weights for the network. If not provided, the weights will be randomly initialized.
                biases (array-like, optional): The initial biases for the network. If not provided, the biases will be randomly initialized.
                seed (int, optional): The seed value for numpy.random. Defaults to 42.
            """

            self.seed = seed
            np.random.seed(seed)  # Set the seed for numpy.random
            if input.ndim == 1:
                x_size = 1
            else:
                x_size = input.shape[1]
    
            self.layers = []
            

            for i, init_layer in enumerate(init_layers):
                input_dim = init_layer.get("input_dim")
                if i == 0:
                    input_dim = x_size
                else:
                    input_dim = init_layers[i-1]["output_dim"]
                
                if i == len(init_layers) - 1:
                    output_dim = 1 # only one dim output is allowed for the output layer
                else:
                    output_dim = init_layer.get("output_dim")
                if output_dim is None:
                    raise ValueError("Output dimension not specified for layer {}".format(i))
                
                activation = init_layer.get("activation")
                if activation is None:
                    raise ValueError("Activation function not specified for layer {}".format(i))
                
                init_method = init_layer.get("init")
                if init_method is None:
                    init_method = "uniform"
                
                if weights is not None and biases is not None:
                    try:
                        layer = FCLayer(input_dim, output_dim, activation, weights[i], biases[i], )
                    except IndexError:
                        raise ValueError("The number of weights and biases provided does not match the number of layers")
                else:
                    layer = FCLayer(input_dim, output_dim, activation, init_method)
                self.layers.append(layer)

                self.depth = len(self.layers)

    def __str__(self) -> str:
        description = "MLP with layers:\n"
        for layer in self.layers:
            description += str(layer) + "\n"
        return description


    def full_forward_pass(self, input: np.ndarray):
        """
        input: an input matrix X
        """
        
        output = input
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def loss(self, input, y, output = None):
        if output is None:
            output = self.full_forward_pass(input)

        return np.mean((output - y) ** 2)

    def full_backward_propagation(self, input: np.ndarray, y: np.ndarray):
        """function performs one full backward pass through the network and updates the params. Uses optionally the momentum method and RMSprop.

        Args:
            input (np.ndarray): matrix of input data
            y (np.ndarray): single column matrix of target values
            learning_rate (float, optional): learning rate of the gradient descent. Defaults to 0.01.
            momentum_rate (float, optional): momentum rate - describes how much the previous change of the weights should influence the new change. Defaults to 0.01.
            gamma (float, optional): gamma parameter of the RMSprop. Defaults to 1.
            epsilon (float, optional): small value used in RMSprop algorithm. Defaults to 1e-8.

        Returns:
            (tuple): tuple of gradients of weights and biases
        """

        y_hat = self.full_forward_pass(input) # forward pass automatically stores the output in the layers
        
        batch_size = input.shape[1]
        
        g = 2 * (y_hat - y) / float(batch_size) # gradient of the loss function
        db_table = [None] * self.depth # derivatives of output in respect to weights
        dw_table = [None] * self.depth # derivatives of output in respect to biases

        for i in reversed(range(self.depth)):
            dw, db, g = self.layers[i].backward_propagation(g)
            
            # print("shape of g: ", g.shape)
            # print("shape of h: ", h[i].shape)
            # print("shape of dw: ", dw[i].shape)
            # print("shape of db: ", db[i].shape)
            # print("shape of weights: ", self.layers[i].weights.shape)
            # print("shape of biases: ", self.layers[i].bias.shape)

            # update params
            # self.layers[i].update_params(dw, db, learning_rate = learning_rate, momentum_rate = momentum_rate, gamma = gamma, epsilon = epsilon)

            db_table[i] = db
            dw_table[i] = dw

        return dw_table, db_table
    
    def update_params(self, dw_table, db_table):
        
        # add regularization
        
        for i in range(self.depth):
            self.layers[i].update_params(dw_table[i], db_table[i])
    

    def train(self, input, y, epochs = 100, batch_size = 32, learning_rate = 0.01, stochastic_descent = False, momentum = False, 
              momentum_rate = 0.01, rms_prop = False, rms_rate = 0.9, *args, **kwargs):
        """
        trains a neural network, returns a list of losses for each epoch, a wrapper function
        """
        
        losses = []
        if momentum:
            dw_momentum = [np.zeros_like(layer.weights) for layer in self.layers]
        if rms_prop:
            dw_rms = [np.zeros_like(layer.weights) for layer in self.layers]

        for epoch in range(epochs):
            permutation = np.random.permutation(input.shape[1])
            input = input[:, permutation]
            y = y[:, permutation]
            
            if not stochastic_descent:
                batch_input_len = input.shape[1]
            else:
                batch_input_len = batch_size

            for batch_number in range(0, batch_input_len, batch_size):
                input_batch = input[:, batch_number:batch_number+batch_size]
                y_batch = y[:, batch_number:batch_number+batch_size]
                
                dw, db = self.full_backward_propagation(input_batch, y_batch)
                
                # modify the weights and biases
                # add momentum
                if momentum:
                    dw_momentum = [dw[i] * learning_rate + dw_momentum[i] * momentum_rate for i in range(self.depth)]
                    dw = dw_momentum
                    
                # add RMSprop

                if rms_prop:
                    dw_rms = [rms_rate * dw_rms[i] + (1 - rms_rate) * dw[i] ** 2 for i in range(self.depth)]
                    dw = [dw[i] / np.sqrt(dw_rms[i]) for i in range(self.depth)]
                
                db = [db[i] * learning_rate for i in range(self.depth)]
                
                self.update_params(dw, db)
                
            losses.append(self.loss(input, y))
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.loss(input, y)}")
        return losses
           
    
    def get_weights(self):
        return [(layer.get_weights(), layer.get_biases()) for layer in self.layers]
    


    
    
