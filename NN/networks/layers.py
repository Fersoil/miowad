import numpy as np
from .activations import relu, sigmoid, linear, prelu, relu_prime, sigmoid_prime, linear_prime, tanh_prime, prelu_prime
import matplotlib.pyplot as plt

class Layer:
    def __init__(self):
        self.input_dim = None
        self.output_dim = None
        self.activation = None

    def forward_pass(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    

class FCLayer(Layer):
    __slots__ = ["input_dim", "output_dim", "activation", "weights", "bias"]

    def __init__(self, input_dim, output_dim, activation="sigmoid", weights=None, biases=None, init="uniform"):
            """
            Initializes a layer of a neural network.

            Args:
                input_dim (int): The dimension of the input to the layer.
                output_dim (int): The dimension of the output from the layer.
                activation (str, optional): The activation function to be used. Defaults to "sigmoid".
                weights (ndarray, optional): The weights for the layer. Defaults to None.
                biases (ndarray, optional): The biases for the layer. Defaults to None.
                init (str, optional): The initialization method for the weights. Defaults to "uniform". Options are "uniform", "Xavier" and "He" or None. If None, the weights and biases will be taken from the function parameter.

            Raises:
                ValueError: If the activation function is not supported.
            """
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
    
            if activation == "sigmoid":
                self.activation = sigmoid
            elif activation == "relu":
                self.activation = relu
            elif activation == "linear":
                self.activation = linear
            else:
                raise ValueError("Activation function not supported")
            
            # derivative
            if activation == "relu":
                self.activation_prime = relu_prime
            elif activation == "sigmoid":
                self.activation_prime = sigmoid_prime
            elif activation == "linear":
                self.activation_prime = linear_prime
            elif activation == "tanh":
                self.activation_prime = tanh_prime
            elif activation == "prelu":
                self.activation_prime = prelu_prime
            else:
                raise ValueError("Activation function not supported")

            if init not in [None, "uniform", "Xavier", "He"]:
                raise ValueError("Initialization method not supported")

            if init is None:
                if weights is not None and biases is not None:
                
                    if weights.shape != (output_dim, input_dim):
                        raise ValueError("The shape of the weights provided does not match the layer's dimensions")
                    if biases.shape != (output_dim, 1):
                        raise ValueError("The shape of the biases provided does not match the layer's dimensions")
                    
                    self.weights = weights
                    self.bias = biases
                else:
                    raise ValueError("The weights and biases must be provided if the initialization method is None")
            if init == "uniform":
                self.random_init_weights()
            elif init == "Xavier":
                self.xavier_init_weights()
            elif init == "He":
                self.he_init_weights()

    def __str__(self) -> str:
        return "Fully connected layer with input dimension {} and output dimension {} and {} activation function".format(self.input_dim, self.output_dim, self.activation.__name__)

    # initialize weights and biases
    def random_init_weights(self):
        self.weights = np.random.uniform(0, 1, (self.output_dim, self.input_dim))
        self.bias = np.zeros((self.output_dim, 1))

    def xavier_init_weights(self):
        """
        xavier uniform initialization
        """
        x = np.sqrt(6/(self.input_dim + self.output_dim))        
        self.weights = np.random.uniform(-x, x, (self.output_dim, self.input_dim)) * np.sqrt(1/self.input_dim)
        self.bias = np.zeros((self.output_dim, 1))

    def he_init_weights(self):
        """
        he normal initialization https://arxiv.org/abs/1502.01852v1
        """
        if self.activation is not relu or self.activation is not prelu:
            raise Warning("He initialization is recommended for ReLU or PReLU activation function")
        
        std = np.sqrt(2/self.input_dim)
        self.weights = np.random.normal(0, std, (self.output_dim, self.input_dim))
        self.bias = np.zeros((self.output_dim, 1))


    def forward_pass(self, input):
        input = input
        output = self.activation(np.dot(self.weights, input) + self.bias)
        return output
    
    def backward_propagation(self, output_error):
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)
        
        pass


    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.bias
    
    def plot_weights(self):
        plt.figure(figsize=(10, 6))
        for i in range(self.weights.shape[1]):
            plt.bar(len(self.weights[:, i]), self.weights[:, i])
        plt.xlabel('Column Index')
        plt.ylabel('Weight Value')
        plt.title('Weights of FCLayer')
        plt.show()