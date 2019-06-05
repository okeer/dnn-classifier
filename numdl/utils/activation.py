import numpy as np


class ActivationFunc(object):
    def forward(self, Z):
        pass

    def backward(self, Z, dA):
        pass


class SigmoidFunc(ActivationFunc):
    @staticmethod
    def __sigmoid(Z):
        return 1 / (1 + np.exp(- Z))

    @staticmethod
    def forward(Z):
        A = SigmoidFunc.__sigmoid(Z)
        return A

    @staticmethod
    def backward(Z, dA):
        s = SigmoidFunc.__sigmoid(Z)
        dZ = dA * s * (1 - s)
        return dZ


class ReluFunc(ActivationFunc):
    @staticmethod
    def __relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def forward(Z):
        A = ReluFunc.__relu(Z)
        return A

    @staticmethod
    def backward(Z, dA):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ
