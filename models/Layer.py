import numpy as np


class Layer(object):
    def __init__(self, units_count, activation_func):
        self.current_layer_dim = units_count
        self.activation_func = activation_func

        self.weights = None
        self.bias = None
        self.pre_activation = None
        self.activation = None

    def __if_params_not_initialized(self):
        return (self.weights is None) or (self.bias is None)

    def __init_parameters(self, size_of_previous_layer):
        self.weights = np.random.randn(self.current_layer_dim, size_of_previous_layer) / np.sqrt(self.current_layer_dim)
        self.bias = np.zeros((size_of_previous_layer, 1))

    def forward(self, activation_from_previous_layer):
        if self.__if_params_not_initialized():
            self.__init_parameters(activation_from_previous_layer.shape[0])

        self.pre_activation = self.weights.dot(activation_from_previous_layer) + self.bias
        self.activation = self.activation_func(self.pre_activation)

        return self.activation

    def backward(self):
        pass
