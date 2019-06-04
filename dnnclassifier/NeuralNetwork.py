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

        for index in range(self.layers_count):
            self.propagation_features = self.layers[index].forward(self.propagation_features)

    def __backward_propagation(self, classes):
        d_activation = - (np.divide(classes, self.propagation_features) - np.divide(1 - classes,
                                                                                    1 - self.propagation_features))
        for index in reversed(range(len(self.layers))):
            d_activation = self.layers[index].backward(d_activation)
            self.layers[index].update(self.learning_rate)

    def __compute_loss(self, activation_layer, classes):
        m = classes.shape[1]
        return 1. / m * np.nansum(
            np.multiply(-np.log(activation_layer), classes) + np.multiply(-np.log(1 - activation_layer), 1 - classes))

    def train(self, features, classes, chunk_size=None):
        if chunk_size is not None and chunk_size > classes.shape[1]:
            raise Exception("Chunk size should be less than number of examples")
        if chunk_size is None:
            chunk_size = classes.shape[1]

        chunks = np.round(features.shape[1] / chunk_size).astype(int)
        split_indices = np.arange(start=chunk_size, step=chunk_size, stop=chunk_size * chunks)

        for epoch in range(self.iterations):
            rand_perm = np.random.permutation(features.shape[1])

            X_chunks = np.hsplit(features[:, rand_perm], split_indices)
            Y_chunks = np.hsplit(classes[:, rand_perm], split_indices)

            for index in range(len(X_chunks)):
                self.__forward_propagation(X_chunks[index])
                self.__backward_propagation(Y_chunks[index])

            if epoch % 100 == 0:
                print(f"Epoch {epoch} loss is {self.__compute_loss(self.propagation_features, Y_chunks[index])}")

    def predict(self, features):
        self.__forward_propagation(features)
        return self.propagation_features
