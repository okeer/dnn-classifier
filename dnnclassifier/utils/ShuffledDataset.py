import numpy as np


class ShuffledDataset(object):
    def __init_params(self):
        if self.batch_size is None:
            self.batch_size = self.X.shape[1]
        elif self.batch_size > self.X.shape[1]:
            raise Exception("Chunk size should be less than number of examples")

        self.chunks = np.round(self.X.shape[1] / self.batch_size).astype(int)
        self.indicies = np.arange(start=self.batch_size, step=self.batch_size, stop=self.batch_size * self.chunks)
        self.shuffle()

    def __init__(self, X, Y, mini_batch_size=None):
        self.X = X
        self.Y = Y

        self.random_permutation = None
        self.X_random = None
        self.Y_random = None

        self.batch_size = mini_batch_size
        self.index = 0
        self.__init_params()

    def shuffle(self):
        self.random_permutation = np.random.permutation(self.X.shape[1])
        self.X_random = np.hsplit(self.X[:, self.random_permutation], self.indicies)
        self.Y_random = np.hsplit(self.Y[:, self.random_permutation], self.indicies)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            res = self.X_random[self.index], self.Y_random[self.index]
            self.index += 1
            return res
        except IndexError:
            raise StopIteration
