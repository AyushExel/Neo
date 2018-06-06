"""
Class containing functionality to build and train neural networks
Contains all activation and loss functions some other utility function.
"""

import numpy as np


class nn:
    def __init__(self, layer_dimensions, activations, cost_function):
        """
        Initializes networks's weights and other useful variables.

        Parameters contains weights of the layer in form {'Wi':[],'bi':[]}
        Cache contains intermediate results as [[A[i-1],Wi,bi],[Zi]], where i
        is layer number.

        :param layer_dimensions:
        :param activations: To store the activation for each layer
        """
        self.parameters = {}
        self.cache = []
        self.activations = activations
        self.cost_function = cost_function

        self.initialize_parameters(layer_dimensions)

    def initialize_parameters(self, layer_dimensions):
        """
        Random initialization of weights of a network described by given layer
        dimensions.

        :param layer_dimensions: Dimensions to layers of the network
        :return: None
        """
        for i in range(1, len(layer_dimensions)):
            self.parameters["W" + str(i)] = (
                np.random.randn(layer_dimensions[i],
                                layer_dimensions[i - 1]) * 0.01
            )
            self.parameters["b" + str(i)] = np.zeros((layer_dimensions[i], 1))

    def __activate(self, Z, n_layer=1):
        """
        Activate the given layer(Z) using the activation function specified by
        'type'.

        Note: This function treats 1 as starting index!
              First layer's index is 1.

        :param Z: Layer to activate
        :param n_layer: Layer's index
        :return: Activated layer and activation cache
        """
        act_cache = [Z]
        act = None
        if (self.activations[n_layer - 1]).lower() == "relu":
            act = Z * (Z > 0)
        if (self.activations[n_layer - 1]).lower() == "tanh":
            act = np.tanh(Z)
        if (self.activations[n_layer - 1]).lower() == "sigmoid":
            act = 1 / (1 + np.exp(-Z))

        # assert(act!=None)

        return act, act_cache

    def forward(self, net_input):
        """
        To forward propagate the entire Network.

        :param net_input: Contains the input to the Network
        :return: Output of the network
        """
        A = net_input

        for i in range(1, int(len(self.parameters) / 2)):
            W = self.parameters["W" + str(i)]
            b = self.parameters["b" + str(i)]
            Z, linear_cache = self.__linear_forward(A, W, b)

            A, act_cache = self.__activate(Z, i)
            self.cache.append([linear_cache, act_cache])

        # For Last Layer
        W = self.parameters["W" + str(int(len(self.parameters) / 2))]
        b = self.parameters["b" + str(int(len(self.parameters) / 2))]
        Z, linear_cache = self.__linear_forward(A, W, b)
        if len(self.activations) == len(self.parameters) / 2:
            A, act_cache = self.__activate(Z, len(self.activations))
            self.cache.append([linear_cache, act_cache])
        else:
            A = Z
            self.cache.append([linear_cache, [None]])

        return A

    def forward_upto(self, net_input, layer_num):
        """
        Calculates forward prop upto layer_num.

        :param net_input: Contains the input to the Network
        :param layer_num: Layer up to which forward prop is to be calculated
        :return: Activations of layer layer_num
        """
        if layer_num == int(len(self.parameters) / 2):
            return self.forward(net_input)
        else:
            A = net_input
            for i in range(1, layer_num):
                W = self.parameters["W" + str(i)]
                b = self.parameters["b" + str(i)]
                Z, linear_cache = self.__linear_forward(A, W, b)

                A, act_cache = self.__activate(Z, i)
                self.cache.append([linear_cache, act_cache])
            return A

    @staticmethod
    def __linear_forward(A_prev, W, b):
        """
        Linear forward to the current layer using previous activations.

        :param A_prev: Previous Layer's activation
        :param W: Weights for current layer
        :param b: Biases for current layer
        :return: Linear cache and current calculated layer
        """
        Z = W.dot(A_prev) + b
        linear_cache = [A_prev, W, b]
        return Z, linear_cache

    @staticmethod
    def cross_entropy_loss(prediction, mappings):
        """
        Calculates the cross entropy loss between output of the network and
        the real mappings of a function.

        Changes cost_function to appropriate value.

        :param prediction: Output of the neural net
        :param mappings: Real outputs of a function
        :return: Mean squared error b/w output and mappings
        """
        a = -(1 / output.shape[1])
        b = mappings * np.log(prediction)
        c = (b + (1 - mappings) * np.log(1 - prediction))
        return a * c.sum()
    
    @staticmethod
    def mse_loss(prediction, mappings):
        """
        Calculates the Mean Squared error between output of the network and
        the real mappings of a function.

        Changes cost_function to appropriate value.

        :param prediction: Output of the neural net
        :param mappings: Real outputs of a function
        :return: Mean squared error b/w output and mappings
        """
        return np.square(prediction-mappings).mean()


def test_run():
    """
    Sample test run.

    :return: None
    """
    # test run:
    data = np.random.randn(2, 100)
    net = nn([2, 15, 2], ["tanh", "relu"])
    A = net.forward(data)
    print(A.shape)


if __name__ == "__main__":
    test_run()
