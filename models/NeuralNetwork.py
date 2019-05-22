import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers, learning_rate, iterations):
        self.layers = layers
        self.layers_count = len(self.layers)
        self.propagation_features = None
        self.learning_rate = learning_rate
        self.iterations = iterations

    def __forward_propagation(self, features):
        self.propagation_features = features

        for index in range(self.layers_count - 1):
            self.propagation_features = self.layers[index].forward(self.propagation_features)

    def __backward_propagation(self, classes):
        d_activation = (np.divide(classes, self.propagation_features) - np.divide(1 - classes, 1 - self.propagation_features))
        for index in reversed(range(len(self.layers) - 1)):
            d_activation = self.layers[index].backward(d_activation)
            self.layers[index].update(self.learning_rate)

    def train(self, features, classes):
        for index in range(self.iterations):
            self.__forward_propagation(features)
            self.__backward_propagation(classes)

    def predict(self, features):
        self.__forward_propagation(features)
        return self.propagation_features
