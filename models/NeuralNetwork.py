import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers
        self.layers_count = len(self.layers)
        self.propagation_features = None

    def __forward_propagation(self, features):
        self.propagation_features = features

        for index in range(self.layers_count - 1):
            self.propagation_features = self.layers[index].forward(self.propagation_features)

    def __backward_propagation(self, classes):
        d_activation = (np.divide(classes, self.propagation_features) - np.divide(1 - classes, 1 - self.propagation_features))
        for index in range(len(self.layers) - 1, 0):
            d_activation = self.layers[index].backward(d_activation)

    def train(self, features, classes):
        self.__forward_propagation(features)
        self.__backward_propagation(classes)

    def predict(self, features):
        self.__forward_propagation(features)
        return self.propagation_features
