from abc import ABC, abstractmethod
import numpy as np
from .layers import FCLayer


class MLP:
    __slots__ = ["layers", "seed"]

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
    
    def loss(self, input, y):
        return np.mean((self.full_forward_pass(input) - y) ** 2)

    def full_backward_propagation(self, input: np.ndarray, y: np.ndarray, learning_rate = 0.01):
        """
        input: an input matrix X
        y: the target matrix
        learning_rate: the learning rate
        """
        a_s = [input]  # list to store all the layer outputs (after activation function)
        z_s = [] # list of all weighted inputs into activation function

        for layer in self.layers:
            z = np.dot(layer.weights, a_s[-1]) + layer.bias
            z_s.append(z)
            a = layer.activation(z)
            a_s.append(a)


        dw = []  # dC/dW
        db = []  # dC/dB

        deltas = [None] * len(self.layers)  # error for each layer

        deltas[-1] = ((y-a_s[-1])*(self.layers[-1].activation_prime(z_s[-1])))


        for i in reversed(range(len(deltas)-1)):
            deltas[i] = self.layers[i+1].weights.T.dot(deltas[i+1])*(self.layers[i].activation_prime(z_s[i]))  
        
        a = [d.shape for d in deltas]
        batch_size = y.shape[1]
        db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
        dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
        # return the derivitives respect to weight matrix and biases

        for i, layer in enumerate(self.layers):
            layer.weights += learning_rate * dw[i]
            layer.bias += learning_rate * db[i]

        return dw, db
    

    def train(self, input, y, learning_rate = 0.01, epochs = 100):
        """
        trains a neural network, returns a list of losses for each epoch
        """
        losses = []
        for epoch in range(epochs):
            self.full_backward_propagation(input, y, learning_rate)
            losses.append(self.loss(input, y))
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {self.loss(input, y)}")
        return losses
    


# TODO backpropagation
# TODO tests
    
    
