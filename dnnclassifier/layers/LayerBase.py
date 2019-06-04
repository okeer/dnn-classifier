import numpy as np


class LayerBase(object):
    def __init__(self, units_count, activation_func):
        self.current_layer_dim = units_count
        self.activation_func = activation_func

        self.weights = None
        self.bias = None
        self.pre_activation = None
        self.activation_layer = None
        self.activation = None
        self.d_weights = None
        self.d_bias = None
        self.d_activation = None

    def __if_params_not_initialized(self):
        return (self.weights is None) or (self.bias is None)

    def __init_parameters(self, size_of_previous_layer):
        self.weights = np.random.randn(self.current_layer_dim, size_of_previous_layer) \
                       * np.sqrt(2. / size_of_previous_layer)
        self.bias = np.zeros((self.current_layer_dim, 1))

    def __forward_linear(self):
        if self.__if_params_not_initialized():
            self.__init_parameters(self.activation.shape[0])

        self.pre_activation = self.weights.dot(self.activation) + self.bias

    def forward(self, activation):
        self.activation = activation
        self.__forward_linear()
        self.activation_layer = self.activation_func.forward(self.pre_activation)
        return self.activation_layer

    def __backward_linear(self, d_pre_activation):
        m = self.activation.shape[1]

        self.d_weights = 1. / m * np.dot(d_pre_activation, self.activation.T)
        self.d_bias = 1. / m * np.sum(d_pre_activation, axis=1, keepdims=True)
        self.d_activation = np.dot(self.weights.T, d_pre_activation)

    def backward(self, dA):
        dZ = self.activation_func.backward(self.pre_activation, dA)
        self.__backward_linear(dZ)
        return self.d_activation

    def update(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.bias -= learning_rate * self.d_bias
