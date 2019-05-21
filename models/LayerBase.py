import numpy as np


class LayerBase(object):
    def __init__(self, units_count, activation_func):
        self.current_layer_dim = units_count
        self.activation_func = activation_func

        self.weights = None
        self.bias = None
        self.pre_activation = None
        self.activation = None
        self.activation_previous = None

    def __if_params_not_initialized(self):
        return (self.weights is None) or (self.bias is None)

    def __init_parameters(self, size_of_previous_layer):
        self.weights = np.random.randn(self.current_layer_dim, size_of_previous_layer) / np.sqrt(self.current_layer_dim)
        self.bias = np.zeros((size_of_previous_layer, 1))

    def __forward_linear(self, activation_previous):
        self.activation_previous = activation_previous
        if self.__if_params_not_initialized():
            self.__init_parameters(activation_previous.shape[0])

        self.pre_activation = self.weights.dot(activation_previous) + self.bias

    def forward(self, activation_from_previous_layer):
        self.__forward_linear(activation_from_previous_layer)

    def backward(self):
        pass
