import numpy as np
from .activations import relu, sigmoid, linear, prelu, tanh, relu_prime, sigmoid_prime, linear_prime, tanh_prime, prelu_prime, softmax, softmax_prime
import matplotlib.pyplot as plt
import pandas as pd

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
    __slots__ = ["input_dim", "output_dim", "activation", "weights", "bias", "output", "linear_output", "input", "dZ"]
    
    
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
            elif activation == "tanh":
                self.activation = tanh
            elif activation == "linear":
                self.activation = linear
            elif activation == "prelu":
                self.activation = prelu
            elif activation == "softmax":
                self.activation = softmax
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
            elif activation == "softmax":
                self.activation_prime = softmax_prime
            else:
                raise ValueError("Activation function not supported")

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
            elif init == "xavier":
                self.xavier_init_weights()
            elif init == "he":
                self.he_init_weights()
            elif init == "normal":
                self.normal_init_weights()
            else: 
                raise ValueError("Initialization method not supported")
                
            self.bias_change = None
            self.weights_change = None
            self.moving_average_weights = None
            self.moving_average_bias = None

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
        self.normal_init_weights()

        std = np.sqrt(1/self.input_dim)
        self.weights = self.weights * std

    def he_init_weights(self):
        """
        he normal initialization https://arxiv.org/abs/1502.01852v1
        """
        if self.activation is not relu and self.activation  is not prelu:
            print(f"He initialization is recommended for ReLU or PReLU activation function, used {self.activation.__name__} instead.")
        self.normal_init_weights()

        std = np.sqrt(2/self.input_dim)
        self.weights = self.weights * std

    def normal_init_weights(self):
        self.weights = np.random.randn(self.output_dim, self.input_dim)
        self.bias = np.zeros((self.output_dim, 1))

    
    def forward_pass_without_activation(self, input):

    
        self.input = input
        
        self.linear_output = np.matmul(self.weights,  input) + self.bias
        return self.linear_output


    def forward_pass(self, input):

        self.batch_size = input.shape[1]
        self.output = self.activation(self.forward_pass_without_activation(input))
        return self.output
    
    def backward_propagation(self, upstream_gradient):

        self.dZ = upstream_gradient * self.activation_prime(self.get_linear_output())

        try:
            db = np.sum(self.dZ, axis=1, keepdims=True)
        except:
            db = np.sum(self.dZ, axis=1)
        # db = np.sum(self.dZ, axis=1, keepdims=True) this line threw an error - but the keepdims parameter looks like not necessary
        dw = np.matmul(self.dZ, self.get_input().T) #/ float(self.batch_size)


        # print("input shape ", self.get_input().shape)

        g = np.matmul(self.weights.T, self.dZ) # * self.activation_prime(self.get_linear_output())

        self.input = None
        self.output = None
        self.linear_output = None

        return dw, db, g
    


    # getters for temporary variables

    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.bias
    
    def get_linear_output(self):
        if self.linear_output is None:
            return self.forward_pass_without_activation(self.input)
        return self.linear_output
    
    def get_output(self):
        if self.output is None and self.linear_output is not None:
            return self.activation(self.linear_output)
        if self.output is None and self.linear_output is None and self.input is not None:
            return self.forward_pass(self.input)
        if self.output is None:
            return None
        return self.output
    
    def get_input(self):
        return self.input
    
    def get_batch_size(self):
        if self.batch_size is None:
            return self.get_input().shape[1]
        return self.batch_size
    
    
    def plot_weights(self):
        plt.figure(figsize=(3 * self.input_dim, 3 * self.output_dim))
        for i in range(self.weights.shape[1]):
            print(self.weights[:, i])
            plt.subplot(self.weights.shape[1], 1, i+1)
            plt.bar(range(len(self.weights[:, i])), self.weights[:, i])
            plt.title(f'weights of Neuron {i}')

        plt.xlabel('Column Index')
        plt.ylabel('Weight Value')
        plt.suptitle('Weights of FCLayer')
        plt.show()       
        

    
        
    def update_params(self, dw: np.ndarray, db: np.ndarray):
        """function updates parameters of a single layer in a neural network

        Args:
            dw (np.ndarray): array of derivatives of the loss function with respect to the weights
            db (np.ndarray): array of derivatives of the loss function with respect to the biases
        """
        
        self.weights = self.weights - dw
        self.bias = self.bias - db