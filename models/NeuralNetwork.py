import numpy as np


class NeuralNetwork(object):
    def __init__(self, layers):
        self.layers = layers
        self.layers_count = len(self.layers)

    def train(self, features, classes):
        propagation_features = features

        for index in range(self.layers_count - 1):
            propagation_features = self.layers[index].forward(propagation_features)

        d_activation = (np.divide(classes, propagation_features) - np.divide(1 - classes, 1 - propagation_features))

        for index in range(len(self.layers) - 1, 0):
            d_activation = self.layers[index].backward(d_activation)

    def predict(self, features):
        propagation_features = features

        for index in range(self.layers_count - 1):
            propagation_features = self.layers[index].forward(propagation_features)

        return propagation_features
