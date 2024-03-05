import numpy as np
from .layers import FCLayer


class MLP:
    __slots__ = ["layers", "depth"]

    def __init__(self, init_layers, input, weights=None, biases=None):
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

            if input.ndim == 1:
                x_size = 1
            else:
                x_size = input.shape[0]
    
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
                        print(f"initializing layer {i} using predefined weights and biases")
                        layer = FCLayer(input_dim, output_dim, activation, weights[i], biases[i], )
                    except IndexError:
                        raise ValueError("The number of weights and biases provided does not match the number of layers")
                else:
                    print(f"initializing layer {i} using {init_method} initialization")
                    layer = FCLayer(input_dim, output_dim, activation=activation, init=init_method)
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
        
        batch_size = input.shape[1]
        if verbose:
            print("y_hat: ", input.shape, y_hat.shape, y.shape)
        
        g = 2 * (y_hat - y) / float(batch_size) # gradient of the loss function
        db_table = [None] * self.depth # derivatives of output in respect to weights
        dw_table = [None] * self.depth # derivatives of output in respect to biases
        
        if verbose:
            print("g mean: ", g.mean())

        for i in reversed(range(self.depth)):
            # print("shape of g: ", g.shape)


            dw, db, g = self.layers[i].backward_propagation(g)
            
            # print("shape of g: ", g.shape)
            # print("shape of dw: ", dw.shape)
            # print("shape of db: ", db.shape)
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
    

    def train(self, input, y, max_epochs = 100, batch_size = 32, learning_rate = 0.01, stochastic_descent = False, momentum = False, 
              momentum_decay = 0.9, rms_prop = False, squared_gradient_decay = 0.99, adam = False, epsilon=1e-8, early_loss_stop = 1e-5, *args, **kwargs):
        """
        trains a neural network, returns a list of losses for each epoch, a wrapper function
        """
        
        losses = []
        counter = 0

        m = input.shape[1]

        if momentum or adam:
            momentum_gradients = [np.zeros_like(layer.weights) for layer in self.layers]
        if rms_prop or adam:
            squared_gradients = [np.zeros_like(layer.weights) for layer in self.layers]

        for epoch in range(max_epochs):
            if stochastic_descent:
                permutation = np.random.permutation(input.shape[1])
                input = input[:, permutation]
                y = y[:, permutation]
            
            for batch_number in range(0, m, batch_size):
                input_batch = input[:, batch_number:batch_number+batch_size]
                y_batch = y[:, batch_number:batch_number+batch_size]
                
                dw, db = self.full_backward_propagation(input_batch, y_batch)
                
                # modify the weights and biases
                # add momentum
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
                
                
                #print("dw: ", [dw[i].mean() for i in range(self.depth)])

                dw = [dw[i] * learning_rate for i in range(self.depth)]                
                db = [db[i] * learning_rate for i in range(self.depth)]

                
                self.update_params(dw, db)

            loss = self.loss(input, y)
            losses.append(loss)
            if loss < early_loss_stop:
                print("Early stop at epoch: ", epoch)
                print("Loss for training set:", loss)
                return losses
                
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {self.loss(input, y)}")
        return losses
           
    
    def get_weights(self):
        return [(layer.get_weights(), layer.get_biases()) for layer in self.layers]
    


    
    
