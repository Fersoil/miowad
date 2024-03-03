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

    def full_backward_propagation(self, input: np.ndarray, y: np.ndarray, learning_rate = 0.01, return_grads = False):
        """
        input: an input matrix X
        y: the target matrix
        learning_rate: the learning rate
        """    

        y_hat = self.full_forward_pass(input) # forward pass automatically stores the output in the layers
        
        batch_size = input.shape[1]
        
        g = 2 * (y_hat - y) / float(batch_size) # gradient of the loss function
        if return_grads:
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

            self.layers[i].weights -= learning_rate * dw
            self.layers[i].bias -= learning_rate * db

            if return_grads:
                db_table[i] = db
                dw_table[i] = dw

        if return_grads:
            return dw_table, db_table
    

    def train(self, input, y, learning_rate = 0.01, epochs = 100):
        """
        trains a neural network, returns a list of losses for each epoch
        """
        losses = []
        for epoch in range(epochs):
            self.full_backward_propagation(input, y, learning_rate)
            losses.append(self.loss(input, y))
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.loss(input, y)}")
        return losses
    

    def minibatch_train(self, input, y, learning_rate = 0.01, epochs = 100, batch_size = 32):
        """
        trains a neural network using mini-batch gradient descent, returns a list of losses for each epoch
        """
        losses = []
        

        for epoch in range(epochs):
            permutation = np.random.permutation(input.shape[1])
            input = input[:, permutation]
            y = y[:, permutation]

            for i in range(0, input.shape[1], batch_size):
                input_batch = input[:, i:i+batch_size]
                y_batch = y[:, i:i+batch_size]
                self.full_backward_propagation(input_batch, y_batch, learning_rate)
            losses.append(self.loss(input, y))
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.loss(input, y)}")
        return losses
    

    def minibatch_stochastic_train(self, input, y, learning_rate = 0.01, epochs = 100, batch_size = 32):
        """
        trains a neural network using stochastic gradient descent, returns a list of losses for each epoch
        """
        losses = []
        

        for epoch in range(epochs):
            minibatch = np.random.choice(input.shape[1], batch_size, replace=False)
            input_batch = input[:, minibatch]
            y_batch = y[:, minibatch]
            
            self.full_backward_propagation(input_batch, y_batch, learning_rate)
            losses.append(self.loss(input, y))
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.loss(input, y)}")
        return losses
    
    def get_weights(self):
        return [(layer.get_weights(), layer.get_biases()) for layer in self.layers]
    


    
    
